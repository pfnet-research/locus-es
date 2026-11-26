"""Free-atom data shared with the GPAW exporter (from the neutral ground-state configurations of GPAW's atom table).

NV_SP: electrons in the outer s and p shells (principal number of the highest occupied s shell), for the valence sp density of
the localized-carrier screening.  SHELL_N: principal number of the s, p and d shells used as the atomic shell of a band edge.
"""

from __future__ import annotations

NV_SP: dict[str, int] = {
    "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8, "Na": 1, "Mg": 2, "Al": 3, "Si": 4,
    "P": 5, "S": 6, "Cl": 7, "Ar": 8, "K": 1, "Ca": 2, "Sc": 2, "Ti": 2, "V": 2, "Cr": 1, "Mn": 2, "Fe": 2, "Co": 2, "Ni": 2,
    "Cu": 1, "Zn": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8, "Rb": 1, "Sr": 2, "Y": 2, "Zr": 2, "Nb": 1,
    "Mo": 1, "Tc": 2, "Ru": 1, "Rh": 1, "Pd": 8, "Ag": 1, "Cd": 2, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8,
    "Cs": 1, "Ba": 2, "La": 2, "Ce": 2, "Pr": 2, "Nd": 2, "Pm": 2, "Sm": 2, "Eu": 2, "Gd": 2, "Tb": 2, "Dy": 2, "Ho": 2,
    "Er": 2, "Tm": 2, "Yb": 2, "Lu": 2, "Hf": 2, "Ta": 2, "W": 2, "Re": 2, "Os": 2, "Ir": 2, "Pt": 1, "Au": 1, "Hg": 2,
    "Tl": 3, "Pb": 4, "Bi": 5, "Po": 6, "At": 7, "Rn": 8,
}  # fmt: skip

SHELL_N: dict[str, tuple[int, int, int]] = {
    "H": (1, 2, 3), "He": (1, 2, 3), "Li": (2, 2, 3), "Be": (2, 2, 3), "B": (2, 2, 3), "C": (2, 2, 3), "N": (2, 2, 3),
    "O": (2, 2, 3), "F": (2, 2, 3), "Ne": (2, 2, 3), "Na": (3, 2, 3), "Mg": (3, 2, 3), "Al": (3, 3, 3), "Si": (3, 3, 3),
    "P": (3, 3, 3), "S": (3, 3, 3), "Cl": (3, 3, 3), "Ar": (3, 3, 3), "K": (4, 3, 3), "Ca": (4, 3, 3), "Sc": (4, 3, 3),
    "Ti": (4, 3, 3), "V": (4, 3, 3), "Cr": (4, 3, 3), "Mn": (4, 3, 3), "Fe": (4, 3, 3), "Co": (4, 3, 3), "Ni": (4, 3, 3),
    "Cu": (4, 3, 3), "Zn": (4, 3, 3), "Ga": (4, 4, 3), "Ge": (4, 4, 3), "As": (4, 4, 3), "Se": (4, 4, 3), "Br": (4, 4, 3),
    "Kr": (4, 4, 3), "Rb": (5, 4, 3), "Sr": (5, 4, 3), "Y": (5, 4, 4), "Zr": (5, 4, 4), "Nb": (5, 4, 4), "Mo": (5, 4, 4),
    "Tc": (5, 4, 4), "Ru": (5, 4, 4), "Rh": (5, 4, 4), "Pd": (4, 4, 4), "Ag": (5, 4, 4), "Cd": (5, 4, 4), "In": (5, 5, 4),
    "Sn": (5, 5, 4), "Sb": (5, 5, 4), "Te": (5, 5, 4), "I": (5, 5, 4), "Xe": (5, 5, 4), "Cs": (6, 5, 4), "Ba": (6, 5, 4),
    "La": (6, 5, 5), "Ce": (6, 5, 5), "Pr": (6, 5, 4), "Nd": (6, 5, 4), "Pm": (6, 5, 4), "Sm": (6, 5, 4), "Eu": (6, 5, 4),
    "Gd": (6, 5, 5), "Tb": (6, 5, 4), "Dy": (6, 5, 4), "Ho": (6, 5, 4), "Er": (6, 5, 4), "Tm": (6, 5, 4), "Yb": (6, 5, 4),
    "Lu": (6, 5, 5), "Hf": (6, 5, 5), "Ta": (6, 5, 5), "W": (6, 5, 5), "Re": (6, 5, 5), "Os": (6, 5, 5), "Ir": (6, 5, 5),
    "Pt": (6, 5, 5), "Au": (6, 5, 5), "Hg": (6, 5, 5), "Tl": (6, 6, 5), "Pb": (6, 6, 5), "Bi": (6, 6, 5), "Po": (6, 6, 5),
    "At": (6, 6, 5), "Rn": (6, 6, 5),
}  # fmt: skip
