# Calculation protocol

These are the settings behind every number in the paper. Each structure file records the k-mesh, plane-wave cutoff and PAW setups used for that system (`kpts`, `ecut`, `setups`).

## Ground states (GPAW, solids)

- **Code and basis.** GPAW 24.1.0 with LOCUS from `locus_gpaw` (`gpaw_plugin/`; GPAW's source is not changed). Plane waves, 600 eV (molecular crystals: 500 eV).
- **Occupations and bands.** Fermi–Dirac smearing of 0.01 eV. Symmetry `full`. Number of bands `-max(8, nocc // 2)`, i.e. nocc + max(8, nocc/2).
- **k-points.** Γ-centred mesh with `n_i = max(3, round(30 Å / |a_i|))`.
  - Systems whose band edges lie at zone-boundary points use the smallest mesh that contains them: even sizes, and multiples of 6 in the hexagonal plane.
  - The mesh actually used is stored with each structure.
- **PAW setups.**
  - GPAW setups 24.11.0.
  - Setups named `new` (`<El>.new.PBE`) for the 36 elements listed in `gpaw_plugin/setups/`.
  - Setups named `fix` for Ca (3d projector) and Ga (3d in the valence).
- **SCF stages.** The calculation runs in two stages, PBE and then LOCUS, each with up to 333 and 300 iterations:

  | Stage | Energy | Density | Eigenstates | Bands converged | Mixer |
  |---|---|---|---|---|---|
  | PBE | 5e-4 | 1e-6 | 1e-9 | all but the 4 highest | GPAW default |
  | LOCUS | 1e-6 | 1e-5 | 1e-7 | all but the 4 highest | plain, β = 0.02, nmaxold = 8, weight = 100 |

  Spin-polarized calculations (open-shell oxides) use the mixer of the summed density (`MixerSum`) with the same parameters and up to 600 LOCUS iterations.

  Energy is in eV per valence electron, density in electrons per valence electron, and eigenstates in eV² per valence electron (GPAW units).
- **Large cells (Γ-only supercells with a band gap).** All stages use a domain decomposition. Only the occupied bands + 8 are converged; the E_S step builds the empty bands itself.
- **Open-shell oxides.** Collinear antiferromagnetic order, from the initial moments stored with each structure.

## E_S

- **Parameters.** Those of `es_core` (`locus_es.params.DEFAULT`). No coefficient is fitted to an experimental gap.
- **Empty bands.** The sum for `eps_KS` uses 2 × nocc empty bands. The exporter adds them by a locked band extension at the converged density.
- **Spin–orbit coupling.**
  - Second variation with a window of bands on each side of the gap: 8 bands in the benchmark cells, and max(8, number of atoms) in supercells (molecular-dynamics snapshots, amorphous cells, SQS).
  - Omitted for elements whose setups carry f projectors.
- **Monolayers.**
  - C2DB structures as distributed (`structures/external_ids.json`), with the k-mesh listed there.
  - Effective layer dielectric constant `eps_s = 1 + (eps_cell - 1) L / d`, with `d` = atomic thickness + 3.1 Å.

## Surfaces

- **Slabs.** Symmetric slabs with 16 Å of vacuum, relaxed with PBE with the central layers fixed (`structures/surfaces.json`).
- **Ionization potential.** IP = [V_vac − ⟨V⟩_centre]_slab + [⟨V⟩ − ε_VBM]_bulk, with the PBE slab dipole, and EA = IP − gap.
  - V_vac is the planar average in the middle of the vacuum.
  - ⟨V⟩_centre is the macroscopic average over one interlayer period at the slab centre.

## Molecular dynamics (halide perovskites)

- **Method.** Langevin dynamics with PFP v9.0.0 in its r²SCAN mode (Matlantis), at fixed cell.
  - The exact parameters of each snapshot are in `structures/perovskites.json` under `source.alloy`: temperature, time step, friction, hydrogen mass, seed, equilibration time, snapshot interval, time of the snapshot, instantaneous temperature and parent cell.
- **Supercells.**
  - 2×2×2 cubic supercells (40 atoms).
  - Cells of 80 atoms (`mdP16`) and 96 atoms.
  - The orthorhombic cell of MAPbI3 and the γ cell of CsPbI3.
  - Three 320-atom cells (`md4`) of CsPbI3: the cell-size check and the cost of the largest cell.
- **Snapshots.** Taken every 2 ps after 6 ps of equilibration.
  - Caesium compounds: time step 2 fs, friction 0.02 (ASE units).
  - Methylammonium and formamidinium compounds: 1 fs, hydrogen mass 4 u, friction 0.1. The hydrogen mass does not change the classical configurational distribution.
  - Exception: a first FAPbI3 trajectory (seed 1) used 0.5 fs and the standard hydrogen mass; it supplies three of the nine FAPbI3 snapshots and the polymorphous FAPbI3 structure.
  - Entries without a recorded friction or hydrogen mass used 0.02 and the standard mass.
  - Seeds 3–5 of 40-atom CsPbI3 at 300 K are the sampling test.
- **Polymorphous structures (`*_poly*`).** Relaxed at fixed cell from the last 300 K snapshot, to residual forces below 0.02 eV/Å.

## Alloys and amorphous cells

- **SQS.**
  - InGaN and AlGaN: 2×2×2 (32 atoms) and 3×3×2 (72 atoms) wurtzite supercells at the Vegard cell.
  - Cu(In,Ga)Se2: 2×2×2 supercells of the primitive chalcopyrite cell (64 atoms).
  - The positions are relaxed. The cell and method of each structure are in `source.structure`.
- **Amorphous cells.**
  - Melt-quenched with PFP v9.0.0 (r²SCAN mode) at the experimental density: liquid at 3,000–4,000 K, quenched within 13.5 ps.
  - Relaxed to residual forces below 0.02 eV/Å.
  - The protocol of each cell is in `source.structure`.
  - Mobility gaps are taken between extended states, i.e. states whose edge-uniformity index is at least 0.4. Every state of the
    window of 40 bands below and 12 above the gap is corrected with the global screening, without spin-orbit coupling.
- SQS alloys: every state of the window of 40 bands below and 16 above the gap is corrected; states with an edge uniformity below
  0.4 are screened as localized carriers (switch per state), and the reported gap is that of the corrected spectrum plus the
  spin-orbit shifts of the edges.
