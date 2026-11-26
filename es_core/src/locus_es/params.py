"""Parameters of E_S as used in the paper.  No coefficient is fitted to an experimental gap.

Levers (eV bohr; dimensionless k_d): least squares through the origin of the exact-exchange corrections <Sigma_x - v_x^PBE> of the PBE
states of ten semiconductors (Si, Ge, GaAs, GaP, InP, InSb, GaN, ZnO, ZnSe, CdTe).  s: least squares of the HOMOs of the 66 GW100
molecules with Z <= 18 and at most 12 atoms against CCSD(T), for the base functional (LOCUS by default).
"""

from __future__ import annotations

from typing import Any

HA: float = 27.211386024367243  # eV per hartree (ase.units.Hartree, as in GPAW)
BOHR: float = 0.5291772105638411  # angstrom per bohr (ase.units.Bohr)

S_BASE: dict[str, float] = {"LOCUS": 0.6575674354312077, "PBE": 0.7084, "r2SCAN": 0.6523}

DEFAULT: dict[str, Any] = {
    "s": S_BASE["LOCUS"],
    "kV": 2.8966287887185795,  # occupied lever: -kV <k_F exp(-alpha) G(s)>
    "kC": 0.6512915837216264,  # empty lever:    +kC <k_F G(s)>
    "k_d": 1.4933882264291105,  # scale of the lever for localized d states
    "mu_d": 0.0,  # d states: fraction of the reference functional's own first-order shift removed (0 in the paper)
    "mu_cap": True,  # cap the removed part at the E_S term of the state
    "screening": {
        "omega_min_ev": 0.1,  # floor of the transition energies
        "kp_cell": True,  # k.p cell average of the 1/omega^3 integrand at direct-gap minima
        "extrap": True,
        "extrap_gamma": 2.0,  # extrapolation of the band sum in the number of empty bands
        "finite": {"g_fin": 0.5521, "eps1": 1.15, "w1": 0.03, "gamma_guard": True, "vac_min_A": 5.0},
        "metal": {"L": 6.0, "sigma_ev": 0.05},  # Thomas-Fermi term 4 pi D_F L^2 (bohr), D_F with a fixed broadening
        "slab": None,  # monolayers: {"delta_A": 3.1, "axis": 2}
    },
    "loc_screen": {
        "on": True,
        "c": 2.0,
        "w": 0.3,
        "m_c": 2.5,
        "w_m": 0.25,
        "ls": "sp",
        "gap_ev": 1.0,
        "complex_tol_ev": 0.1,
    },
    "state_switch": {"f_c": 0.4, "w": 0.05},  # SQS alloys: switch per state from its edge uniformity
}

HB2M: float = 7.619964  # hbar^2 / m_e in eV A^2
