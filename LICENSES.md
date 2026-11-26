# Licences

Each top-level directory has its own `LICENSE` file, which governs everything under the directory. The `LICENSE` file at the top
covers the files outside these directories (`docs/`, `tools/`, `examples/` and the files at the top).

| Directory | Contents | Licence | Imports GPAW? |
|---|---|---|---|
| `es_core/` | `locus_es`: the spectral correction E_S. It reads a code-neutral input file (`docs/ES_INPUT_FORMAT.md`) and returns corrected band edges and levels. | GPL-2.0-or-later (see `es_core/LICENSE`) | no |
| `shell_term/` | `locus_shell`: the shell term of LOCUS. It contains the gate, the energy density and its derivatives, the shell descriptor, and a PySCF implementation of LOCUS. | GPL-2.0-or-later (see `shell_term/LICENSE`) | no |
| `qe_plugin/` | `locus_es_qe`: writes the E_S input file from a Quantum ESPRESSO `pw.x` PAW calculation. It reads only the files of the run and the UPF datasets. | GPL-2.0-or-later (see `qe_plugin/LICENSE`) | no |
| `gpaw_plugin/` | `locus_gpaw`: LOCUS and rVV10 for GPAW 24.1.0 (a subclass of GPAW's meta-GGA class, registered with GPAW's xc factory on import; GPAW's source is not changed) and the PAW setups. `locus_es_gpaw`: writes the E_S input file from a converged GPAW calculation. | GPL-3.0-or-later (see `gpaw_plugin/LICENSE`) | yes |
| `data/` | Structures, k-meshes and calculation protocol of every system in the paper. | GPL-3.0-or-later (see `data/LICENSE`) | no |
| other files | `docs/`, `tools/`, `examples/` and the files at the top | GPL-3.0-or-later (see `LICENSE`) | no |

The packages in `es_core/`, `shell_term/` and `qe_plugin/` are free software; you can redistribute them and/or modify them under the
terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your
option) any later version. Their `LICENSE` files contain the text of version 2.

The contents of `gpaw_plugin/`, `data/` and the other files are free software; you can redistribute them and/or modify them under the
terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your
option) any later version. Their `LICENSE` files contain the text of version 3.

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

## Why two versions

GPAW is distributed under the GNU GPL, version 3 or later. Code that extends GPAW, patches it or runs inside it is a work based on
GPAW, so `gpaw_plugin/` is distributed under version 3 or later.

The packages in `es_core/`, `shell_term/` and `qe_plugin/` are independent programs:

- They do not import GPAW.
- They contain no code taken from GPAW or from Quantum ESPRESSO.
- They exchange data with a DFT code only through documented files and arrays.

Under version 2 or later they can be combined both with programs under version 2 or later, such as Quantum ESPRESSO, and with GPAW
(the combination with GPAW is then distributed under version 3).

Rules for contributors:

- Do not import `gpaw` or copy GPAW code into `es_core/`, `shell_term/` or `qe_plugin/`; do not copy Quantum ESPRESSO code into `qe_plugin/`. A feature that needs GPAW internals belongs in `gpaw_plugin/`.
- Code under version 3 only cannot enter `es_core/`, `shell_term/` or `qe_plugin/`.
- The input format of `es_core` is the only interface between `gpaw_plugin/` and the other packages. Change it through `docs/ES_INPUT_FORMAT.md`, with a version bump.
