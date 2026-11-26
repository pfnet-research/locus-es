# Validation of `locus_es_qe` against the GPAW exporter

Systems: crystals of the paper's release data (`data/structures/crystals.json`), with the structure and k-mesh stored there.

**QE side**

- Quantum ESPRESSO 7.6 (conda-forge build, HDF5 wave functions).
- PBE with the pslibrary 1.0.0 PAW datasets. The `-dn` variant is used where it exists; otherwise the `-n` variant.
- `ecutwfc` 70 Ry (GaAs 60 Ry) and `ecutrho` 8 × `ecutwfc`.
- `scf`, then `nscf` with `nosym` and nocc + max(16, 2 nocc) bands.

**GPAW side**

- The paper's PBE calculations: GPAW 24.1.0, its PAW setups, 600 eV, the same k-mesh.
- E_S on the QE input uses the same s as these calculations (0.7256), so that the two exporters are compared directly.

**What is compared**

Spin–orbit coupling is left out on both sides.

- KS gap: the base PBE gap. It differs by the datasets, not by the exporter.
- g: the screening. It follows the KS gap.
- E_S correction: the shift of the gap, ΔCBM − ΔVBM.
- Bare shift VBM and bare shift CBM: the relative difference of the bare-shift integrals ∫dn k_F e^{−α} G(s) (VBM) and ∫dn k_F G(s) (CBM). These isolate the point set and the state densities.
- d: the shift of the d-band centre.

| System | KS gap GPAW / QE (eV) | g GPAW / QE | E_S correction GPAW / QE (eV) | bare shift VBM | bare shift CBM | d shift GPAW / QE (eV) |
|---|---|---|---|---|---|---|
| GaAs | 0.515 / 0.537 | 0.0552 / 0.0570 | 0.217 / 0.226 | +1.1 % | +0.2 % | -0.583 / -0.602 |
| Si | 0.611 / 0.616 | 0.0697 / 0.0691 | 0.201 / 0.202 | +2.3 % | -0.8 % | – |
| GaP | 1.636 / 1.641 | 0.0869 / 0.0885 | 0.305 / 0.310 | +0.1 % | -0.5 % | -0.916 / -0.934 |
| ZnO | 0.770 / 0.773 | 0.1876 / 0.1927 | 1.300 / 1.335 | +0.1 % | -0.1 % | -1.676 / -1.808 |
| MgO | 4.711 / 4.707 | 0.3135 / 0.3178 | 1.872 / 1.898 | +0.1 % | +0.1 % | – |
| LiF | 9.071 / 9.104 | 0.4928 / 0.5029 | 3.942 / 3.993 | +0.0 % | -1.4 % | – |

## Summary

**Bare-shift integrals.** They depend only on the point set and the state densities. They agree within 2.3 % at the VBM (1.1 % or less except Si) and within 1.4 % at the CBM.

**Other checks.**

- The valence electrons integrated over the points: 17.99997–18.0 for GaAs and GaP.
- ⟨ψ|S|ψ⟩ from the projections: 1 to 1.2e-5 (GaAs).
- The localized-carrier switch t of LiF: 0.505 (GPAW) and 0.507 (QE).

**E_S correction of the gap.** QE is larger by 1–51 meV (1–4 %).

The main part of this comes from g, which is 1–3 % larger with QE. This holds even where the KS gaps agree (MgO 4.711 / 4.707 eV, g +1.4 %; LiF +2.1 %). The cause is not yet identified. Candidates:

- the momentum matrix elements (the PAW correction of the two codes);
- the high empty bands of the `nscf` run.

A direct comparison of `p2_kp` with the GPAW exporter on one system would settle it.

**d bands.** The shifts of the d bands are 2–8 % larger (ZnO −1.68 / −1.81 eV). The pslibrary Zn dataset is generated in a slightly ionic configuration (3d^9.7 4p^0.3), which enters the free-atom data only, not the d states.

**Effect of the pseudo core.** Leaving the pseudo core out of the grid and of the pseudo sphere points (`SPEC.md` §6) brought the VBM bare shift of GaAs from +4.8 % to +1.1 %.

## Run notes

- The apt build of QE 6.7 in the test environment aborts at start-up with a buffer overflow; QE 7.6 (conda-forge) was used.
- The `nscf` run of GaP stopped in the Davidson diagonalization ("S matrix not positive definite"). `diagonalization = 'cg'` was used for the `nscf` runs of GaP, ZnO, MgO and LiF.
- Exporter time on one process: 1.5–3 min per system, for 2–4 atoms and 260–666 k-points.
