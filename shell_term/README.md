# locus_shell

The LOCUS shell term: the pointwise kernel (`kernel.py`) and LOCUS for PySCF (`pyscf_locus.py`). NumPy, SciPy and (for molecules) PySCF.

    pip install .[pyscf]
    python -c "from pyscf import gto; from locus_shell.pyscf_locus import run_locus; print(run_locus(gto.M(atom='Cl 0 0 0; Cl 0 0 1.9878', basis='def2-tzvpp'))[0])"

`tests/test_pyscf_cl2.py` reproduces the LOCUS energy and HOMO of Cl2 in the paper's molecular records.
