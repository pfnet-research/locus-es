# LOCUS and the spectral correction E_S

This repository contains the code for "Band edges and d levels approaching GW accuracy at semilocal cost".

- **E_S** is a one-shot correction of Kohn–Sham band edges and localized levels. It is evaluated after a self-consistent calculation with a semilocal functional.
- **LOCUS** is the base functional: r²SCAN + rVV10 + a shell term for closed d shells.

| Directory | What it is | Licence |
|---|---|---|
| `es_core/` | E_S on a code-neutral input file (NumPy only) | GPL-2.0-or-later |
| `shell_term/` | the LOCUS shell term (pointwise kernel) and LOCUS for PySCF | GPL-2.0-or-later |
| `data/` | structures, k-meshes and protocol for reproducing the paper | GPL-3.0-or-later |
| `gpaw_plugin/` | LOCUS for GPAW, the GPAW exporter of the E_S input, PAW setups | GPL-3.0-or-later |
| `qe_plugin/` | exporter of the E_S input from Quantum ESPRESSO PAW calculations with PBE (prototype; `qe_plugin/SPEC.md`) | GPL-2.0-or-later |

`es_core/`, `shell_term/` and `qe_plugin/` are under GPL-2.0-or-later, the rest under GPL-3.0-or-later; see `LICENSES.md`, which also gives the rule that only `gpaw_plugin/` imports GPAW. See `docs/DESIGN.md` for how the parts fit together.

## Installation

Tested on Ubuntu 22.04 (libxc 5.1.7) and 24.04 (libxc 5.2.3) with CPython 3.11. The versions below are the tested ones.

**System packages.** GPAW 24.1.0 is compiled from source. It needs a C compiler, BLAS and libxc 5 (it does not build with libxc 7):

    sudo apt-get install build-essential libxc-dev libopenblas-dev curl

**Python packages.** GPAW 24.1.0 must be compiled against NumPy 1.x. A plain `pip install gpaw==24.1.0` builds it in an isolated
environment with NumPy 2 and fails (`'PyArray_Descr' has no member named 'elsize'` in `c/lfc.c`). Install NumPy first and build GPAW
without build isolation, so that it uses that NumPy:

    git clone https://github.com/pfnet-research/locus-es.git
    cd locus-es
    uv venv -p 3.11 .venv                            # uv (https://docs.astral.sh/uv/) fetches CPython 3.11 if needed
    source .venv/bin/activate
    uv pip install numpy==1.26.4 scipy==1.13.1 ase==3.23.0 setuptools wheel
    uv pip install --no-build-isolation gpaw==24.1.0
    uv pip install ./gpaw_plugin ./es_core           # LOCUS for GPAW, the exporter, E_S
    uv pip install "./shell_term[pyscf]"             # optional: LOCUS for PySCF (molecules)

Without uv: create the environment with `python3.11 -m venv .venv` (on Ubuntu this needs the packages `python3.11-venv` and
`python3.11-dev`) and write `pip` for `uv pip`.

No change to GPAW itself is needed: `import locus_gpaw` registers LOCUS with GPAW's xc factory.

**Molecules only (PySCF).** LOCUS for PySCF and E_S need neither GPAW nor NumPy 1.x:

    cd locus-es
    uv venv -p 3.11 .venv
    source .venv/bin/activate
    uv pip install ./es_core "./shell_term[pyscf]"

Tested with PySCF 2.3.0 (NumPy 1.26.4, SciPy 1.13.1) and PySCF 2.15.1 (NumPy 2.4.6, SciPy 1.17.1); both reproduce the paper's values
for Cl2 (`shell_term/tests`).

**PAW setups.** `gpaw_plugin/setups/` holds the setups made for this work for 36 elements (`data/PROTOCOL.md`). All other elements use
the standard GPAW setups 24.11.0. Download them once and put both directories on the setup path, the setups of this work first:

    mkdir -p ~/gpaw-setups
    curl -fL https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-24.11.0.tar.gz | tar -xz -C ~/gpaw-setups
    export GPAW_SETUP_PATH="$PWD/gpaw_plugin/setups:$HOME/gpaw-setups/gpaw-setups-24.11.0"

**Check.** GaAs uses only setups of this work. On one core it takes about 10 minutes and must give a LOCUS gap of 1.1606 eV and a
corrected gap of 1.3562 eV (last line of the output):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 examples/run_es.sh data/structures/crystals.json GaAs_107946

**MPI (optional).** For parallel runs, build GPAW with MPI and FFTW instead. Replace the `uv pip install --no-build-isolation
gpaw==24.1.0` step above with:

    sudo apt-get install libopenmpi-dev openmpi-bin libfftw3-dev
    printf "mpi = True\ncompiler = 'mpicc'\nlibraries = ['xc', 'openblas', 'fftw3']\nfftw = True\nscalapack = False\n" > siteconfig.py
    GPAW_CONFIG=$PWD/siteconfig.py uv pip install --no-build-isolation --no-binary gpaw gpaw==24.1.0

The check then runs with `MPI="mpiexec -n 4" examples/run_es.sh data/structures/crystals.json GaAs_107946` and gives 1.1605 and
1.3560 eV: in parallel the shell term uses a linear sphere-to-grid adjoint (see Parallel runs below).

## Reproducing a result of the paper

    examples/run_es.sh data/structures/crystals.json GaAs_107946

This runs the ground state of the paper's protocol (PBE, then LOCUS; `data/PROTOCOL.md`), the export and E_S. For GaAs it gives a
LOCUS gap of 1.1606 eV and a corrected gap of 1.3562 eV.

Molecules (PySCF, def2-TZVPP, grid level 4):

    from pyscf import gto
    from locus_shell.pyscf_locus import run_locus
    e, mf, info = run_locus(gto.M(atom="Cl 0 0 0; Cl 0 0 1.9878", basis="def2-tzvpp"))

Set `OMP_NUM_THREADS` to the number of cores you have; otherwise PySCF starts one thread per CPU it sees, which in a container can
be far more than its CPU limit. On 8 cores the Cl2 example takes about 30 s.

Tests: `pytest es_core/tests shell_term/tests` (the PySCF test is skipped if PySCF is not installed).

## Quick start (GPAW)

    python -m locus_es_gpaw points gs.gpw -o points.npz                     # GPL side: integration points of the ground state
    locus-es fields points.npz -o fields.npz                                # E_S: bare-shift fields on the points
    python -m locus_es_gpaw export gs.gpw --fields fields.npz -o es_input.npz   # GPL side: empty bands, state integrals, SOC
    locus-es run es_input.npz -o es.json                                    # E_S itself

`examples/run_es.sh` lists the options used for each class of system in the paper.

## Interfaces: a screening g(r)

E_S uses one screening g per calculation. A cell that contains two materials with different dielectric constants, such as a
superlattice, needs g to vary in space. `locus-es interface` does this:

- It mixes the bulk g of each material with Hirshfeld-type layer weights into g(r).
- It forms V_S = (1 − P) W_C (1 − P) − P W_V P, with W_X = s kX g(r) u_X.
- It diagonalizes H + V_S among the highest occupied and the lowest empty states.
- It reports the band edges of each layer and the band offsets.

For a constant g it reduces to E_S. Steps:

    locus-es run AlAs_bulk.es_input.npz -o AlAs.es.json                       # the bulk g of each material at the lattice of the cell
    python -m locus_es_gpaw points sl.gpw -o sl.points.npz
    locus-es fields sl.points.npz --layers layers.json -o sl.fields.npz      # layers.json: atoms and g (or es.json) of each layer
    python -m locus_es_gpaw window sl.gpw --fields sl.fields.npz -o sl.window.npz --occ 24 --emp 16
    locus-es interface sl.window.npz --layers layers.json -o sl.iface.json [--g-one sl.es.json]

See `es_core/src/locus_es/interface.py` for the format of `layers.json` and `examples/run_interface.sh` for a complete run.

The layer edges are the highest occupied and the lowest empty states with weight above 0.7 on the layer. An edge counts as defined
only if its weight is at least 0.75 both before and after the diagonalization. Otherwise the state is resonant with a band of the
other material, as for the conduction band of ZnSe against GaAs or Ge.

Spin–orbit edge shifts are not included; add the bulk spin–orbit shifts of each material if they are needed.

Parallel runs.  With a GPAW built with MPI, the two GPAW steps run in parallel (`mpiexec -n N python -m locus_es_gpaw ...`). GPAW
distributes the k-points over the ranks and splits each k-point over real-space domains, so a Γ-only supercell is split into N
domains. Each rank then holds 1/N of the plane-wave coefficients and of the grid points. Over k-points the output does not depend on N.
With domains, it agrees with a serial run to within the convergence of the empty bands that the exporter builds. `window` runs over
k-points only. LOCUS ground states use GPAW's k-point and domain parallelisation (not band parallelisation). In parallel, the potential of the
shell term uses a linear instead of a cubic sphere-to-grid adjoint, which changes converged gaps by up to about 1 meV. One-time setup
tables are cached in `~/.cache/gpaw_dconv` (`DCONV_CACHE_DIR`).

Large cells (Γ-only supercells of a few hundred atoms with a band gap).  Run the ground state with `--large-cell`: only the occupied
bands + 8 are converged (`data/PROTOCOL.md`), and the exporter builds the empty bands itself. With a gap the empty bands carry no
charge, so the density does not depend on them; the option is not meant for metals. With `examples/run_es.sh`, set
`MPI="mpiexec -n N"` and `GSOPT=--large-cell`. Export molecular-dynamics snapshots and other supercells with `--soc-window` equal to
the number of atoms, and run E_S with `--no-loc`, as in the paper. The band extension keeps the projector functions of one k-point
expanded in plane waves (per rank: local plane waves × projectors; above `LOCUS_ES_PROJ_GIB`, default 8 GiB, GPAW's own routines
are used).

Forces.  LOCUS forces include the explicit dependence of the shell term's one-centre evaluation on the atomic positions and agree
with finite differences of the energy (GaAs with a displaced Ga atom: 2e-4 eV/A) in serial runs and under k-point MPI with
`DCONV_SPLAT=cubic`; the default distributed layout omits this term (a warning is printed; error about 2e-3 eV/A).  Because the
shell term reads grid fields at the atomic one-centre points, the energy depends slightly on the position of the atoms relative to
the grid: under a rigid translation of GaAs the slope of the energy is 0.015 eV/A at 500 eV and 0.0015 eV/A at 800 eV.  Use a
higher cutoff for tight relaxations.

*Status: pre-release.*

## Quantum ESPRESSO (prototype)

`qe_plugin/` writes the E_S input from a `pw.x` PAW calculation with PBE, reading only the files of the run and the UPF datasets. E_S then runs with the factor of the PBE base, `locus-es run es_input.npz --s 0.7084`. Usage and scope: `qe_plugin/README.md`. Specification: `qe_plugin/SPEC.md`. Comparison with the GPAW exporter on six crystals: `qe_plugin/VALIDATION.md`.

## Development

Lint and format with [pysen](https://github.com/pfnet/pysen); the settings are in `pysen.toml`:

    pip install "pysen==0.12.1" black flake8 "isort<8" "mypy<2" numpy scipy
    pysen run format
    pysen run lint

`pysen run lint` runs five checks:

- black, isort and flake8;
- mypy in strict mode, which requires the types of all function arguments and return values;
- `annotations` (`tools/check_annotations.py`, loaded by `tools/pysen_annotations.py`).

The `annotations` check requires every newly declared variable to carry a type annotation at its first binding:

- local, module and class variables;
- loop, unpacking and `with` targets, declared just before they are bound;
- attributes of `self`.

Comprehension and lambda variables, exception names, `_` and type aliases are exempt.

Objects of GPAW, ASE and PySCF are typed `Any` (these packages ship no type information). Array types are in each package's `_types.py`.
