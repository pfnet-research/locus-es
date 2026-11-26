"""pysen plugin: runs tools/check_annotations.py on the lint sources as part of `pysen run lint` (target "lint")."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any

from pysen.command import CommandBase
from pysen.component import ComponentBase
from pysen.diagnostic import Diagnostic
from pysen.plugin import PluginBase
from pysen.pyproject_model import Config, PluginConfig
from pysen.reporter import Reporter
from pysen.runner_options import PathContext, RunOptions
from pysen.source import Source, extension_filter


def _checker() -> ModuleType:
    path: pathlib.Path = pathlib.Path(__file__).with_name("check_annotations.py")
    spec: Any = importlib.util.spec_from_file_location("check_annotations", path)
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules["check_annotations"] = module  # dataclasses resolve the module of their class
    spec.loader.exec_module(module)
    return module


class AnnotationCommand(CommandBase):
    def __init__(self, base_dir: pathlib.Path, source: Source) -> None:
        self._base_dir: pathlib.Path = base_dir
        self._source: Source = source

    @property
    def name(self) -> str:
        return "annotations"

    @property
    def has_side_effects(self) -> bool:
        return False

    def __call__(self, reporter: Reporter) -> int:
        files: set[pathlib.Path] = self._source.resolve_files(
            self._base_dir, extension_filter({".py"}), use_git=True, reporter=reporter
        )
        return self._check(reporter, sorted(files))

    def run_files(self, reporter: Reporter, files: Sequence[pathlib.Path]) -> int:
        return self._check(reporter, [f for f in files if f.suffix == ".py"])

    def _check(self, reporter: Reporter, files: list[pathlib.Path]) -> int:
        mod: ModuleType = _checker()
        reporter.logger.info(f"Checking {len(files)} files")
        problems: list[Any] = []
        f: pathlib.Path
        for f in files:
            problems.extend(mod.check_file(f))
        p: Any
        for p in problems:
            reporter.process_output.info(str(p))
        reporter.report_diagnostics(
            [
                Diagnostic(file_path=p.path.resolve(), start_line=p.line, start_column=p.col + 1, message=p.message)
                for p in problems
            ]
        )
        return 1 if problems else 0


class AnnotationComponent(ComponentBase):
    def __init__(self, source: Source) -> None:
        self._source: Source = source

    @property
    def name(self) -> str:
        return "annotations"

    @property
    def targets(self) -> Sequence[str]:
        return ["lint"]

    def create_command(self, target: str, paths: PathContext, options: RunOptions) -> CommandBase:
        return AnnotationCommand(paths.base_dir, self._source)


class AnnotationPlugin(PluginBase):
    def load(self, file_path: pathlib.Path, config: PluginConfig, root: Config) -> Sequence[ComponentBase]:
        source: Source = (
            root.lint.source if root.lint is not None and root.lint.source is not None else Source(includes=["."])
        )
        return [AnnotationComponent(source)]


def plugin() -> PluginBase:
    return AnnotationPlugin()
