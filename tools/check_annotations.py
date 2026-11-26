"""Check that every newly declared variable carries a type annotation (mypy checks function arguments and return values).

A name needs an annotation (`x: T = value`, or a bare `x: T` before other bindings) where it is first bound in its
scope, including loop, unpacking, `with ... as` and walrus targets and attributes of `self` / `cls` (in any method, or
at class level).  Exempt: names bound by import, def, class, global / nonlocal, comprehensions, lambdas and except
clauses, `_`, and module- or class-level CamelCase type aliases (`FloatArray = npt.NDArray[...]`).

    python tools/check_annotations.py FILE_OR_DIR ...
"""

from __future__ import annotations

import ast
import dataclasses
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Union

EXEMPT: frozenset[str] = frozenset({"_", "__all__", "__version__"})
Scope = Union[ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]
CAMEL: re.Pattern[str] = re.compile(r"[A-Z][a-z0-9]+[A-Za-z0-9]*$")


def _type_alias(node: ast.AST) -> bool:
    """a CamelCase name assigned a type expression (subscript, name, attribute, X | Y, TypeVar / NewType call)"""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
        return False
    if not CAMEL.match(node.targets[0].id):
        return False
    v: ast.expr = node.value
    if isinstance(v, ast.Call):
        return isinstance(v.func, ast.Name) and v.func.id in ("TypeVar", "NewType")
    return isinstance(v, (ast.Subscript, ast.Name, ast.Attribute)) or (
        isinstance(v, ast.BinOp) and isinstance(v.op, ast.BitOr)
    )


@dataclasses.dataclass(frozen=True)
class Problem:
    path: Path
    line: int
    col: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col + 1}: {self.message}"


def _names(target: ast.expr) -> Iterator[ast.Name]:
    """the plain names bound by an assignment target (recursing into tuples, lists and starred targets)"""
    if isinstance(target, ast.Name):
        yield target
    elif isinstance(target, (ast.Tuple, ast.List)):
        elt: ast.expr
        for elt in target.elts:
            yield from _names(elt)
    elif isinstance(target, ast.Starred):
        yield from _names(target.value)


def _attrs(target: ast.expr, owners: frozenset[str]) -> Iterator[ast.Attribute]:
    """attributes `self.x` (owners: the first argument of the method) bound by an assignment target"""
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in owners:
        yield target
    elif isinstance(target, (ast.Tuple, ast.List)):
        elt: ast.expr
        for elt in target.elts:
            yield from _attrs(elt, owners)
    elif isinstance(target, ast.Starred):
        yield from _attrs(target.value, owners)


def _scope_nodes(scope: Scope) -> Iterator[ast.AST]:
    """the nodes of a scope in source order, without descending into nested scopes (functions, classes, lambdas, comprehensions)"""
    stack: list[ast.AST] = list(reversed(scope.body))
    while stack:
        node: ast.AST = stack.pop()
        yield node
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            continue
        children: list[ast.AST] = list(ast.iter_child_nodes(node))
        stack.extend(reversed(children))


def _bindings(node: ast.AST) -> tuple[list[ast.expr], bool]:
    """(binding targets of a statement or expression, whether the binding is annotated)"""
    if isinstance(node, ast.AnnAssign):
        return [node.target], True
    if isinstance(node, ast.Assign):
        return list(node.targets), False
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return [node.target], False
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [item.optional_vars for item in node.items if item.optional_vars is not None], False
    if isinstance(node, ast.NamedExpr):
        return [node.target], False
    return [], False


class Checker:
    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self.problems: list[Problem] = []

    def report(self, node: ast.AST, message: str) -> None:
        self.problems.append(Problem(self.path, getattr(node, "lineno", 1), getattr(node, "col_offset", 0), message))

    def scope(self, scope: Scope, declared_attrs: set[str] | None = None, owners: frozenset[str] = frozenset()) -> None:
        """check one scope; declared_attrs / owners: the attributes of the enclosing class declared so far and the names of the
        method's first argument"""
        declared: set[str] = set()
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args: ast.arguments = scope.args
            declared.update(a.arg for a in args.posonlyargs + args.args + args.kwonlyargs)
            if args.vararg is not None:
                declared.add(args.vararg.arg)
            if args.kwarg is not None:
                declared.add(args.kwarg.arg)
        node: ast.AST
        for node in _scope_nodes(scope):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                declared.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                declared.update(node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declared.add(node.name)
                method_owners: frozenset[str] = frozenset()
                if isinstance(scope, ast.ClassDef) and node.args.args:
                    is_static: bool = any(
                        isinstance(d, ast.Name) and d.id == "staticmethod" for d in node.decorator_list
                    )
                    if not is_static:
                        method_owners = frozenset({node.args.args[0].arg})
                self.scope(node, declared_attrs if isinstance(scope, ast.ClassDef) else None, method_owners)
            elif isinstance(node, ast.ClassDef):
                declared.add(node.name)
                self.scope(node, set(), frozenset())
            elif isinstance(scope, (ast.Module, ast.ClassDef)) and _type_alias(node):
                declared.add(node.targets[0].id)  # type: ignore[attr-defined]
            else:
                targets: list[ast.expr]
                annotated: bool
                targets, annotated = _bindings(node)
                target: ast.expr
                for target in targets:
                    name: ast.Name
                    for name in _names(target):
                        if name.id in declared or name.id in EXEMPT:
                            continue
                        if not annotated:
                            self.report(name, f'"{name.id}" is first bound without a type annotation')
                        declared.add(name.id)
                        if isinstance(scope, ast.ClassDef) and declared_attrs is not None:
                            declared_attrs.add(name.id)
                    if declared_attrs is not None:
                        attr: ast.Attribute
                        for attr in _attrs(target, owners):
                            if attr.attr in declared_attrs:
                                continue
                            if not annotated:
                                self.report(attr, f'attribute "{attr.attr}" is first bound without a type annotation')
                            declared_attrs.add(attr.attr)


def check_file(path: Path) -> list[Problem]:
    tree: ast.Module = ast.parse(path.read_text(), filename=str(path))
    checker: Checker = Checker(path)
    checker.scope(tree)
    return sorted(checker.problems, key=lambda p: (p.line, p.col))


def python_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    p: str
    for p in paths:
        q: Path = Path(p)
        out.extend(sorted(q.rglob("*.py")) if q.is_dir() else [q])
    return out


def main(argv: list[str] | None = None) -> int:
    args: list[str] = sys.argv[1:] if argv is None else argv
    problems: list[Problem] = []
    f: Path
    for f in python_files(args or ["."]):
        problems.extend(check_file(f))
    p: Problem
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
