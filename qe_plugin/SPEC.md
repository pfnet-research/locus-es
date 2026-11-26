# `locus_es_qe`: processing specification (prototype)

`locus_es_qe` writes the E_S input files (`docs/ES_INPUT_FORMAT.md`, version 1) from a Quantum ESPRESSO `pw.x` calculation with PAW datasets.

It reads only the files that `pw.x` writes, together with the UPF datasets. It does not import or contain Quantum ESPRESSO or GPAW code.

This document fixes what the exporter computes and how it computes it.

## 1. Scope of the prototype

**Supported**

| Item | Supported |
|---|---|
| Base functional | PBE. In E_S, use `--s 0.7084`, the factor fixed for PBE in the paper. |
| Datasets | PAW in UPF v2 written by `ld1.x`, e.g. pslibrary 1.0.0 (`*.pbe-*-kjpaw_psl.1.0.0.UPF`). The file needs the generation input (`PP_INPUTFILE`), the GIPAW core orbitals and the all-electron core density. |
| Systems | Spin-paired, collinear insulators and semiconductors. |
| k-points | An automatic Monkhorst–Pack mesh (`K_POINTS automatic`). |
| Wave functions | HDF5 files (`wfc<k>.hdf5`, from a `pw.x` built with HDF5). |

**Not implemented**

- spin polarization, metals, noncollinear or spin–orbit runs;
- the Γ-point trick;
- wave functions in the Fortran binary format (`wfc<k>.dat`);
- norm-conserving and ultrasoft pseudopotentials;
- the window groups of SQS alloys and amorphous cells (`--window`);
- the `window` step for interfaces;
- spin–orbit coupling computed by the exporter (§9).

## 2. Calculations to run

1. An `scf` run. Symmetry may be used.
2. An `nscf` run in the same `outdir`, with:
   - `nosym = .true.`, so that the k-points are the full mesh reduced only by time reversal;
   - `nbnd = nocc + max(16, 2 nocc)`, where `nocc` is half the number of valence electrons.

   E_S sums the screening over 2 × nocc empty bands, as in the paper.

The exporter reads `<outdir>/<prefix>.save` of the nscf run:

    python -m locus_es_qe points SAVE -o points.npz
    locus-es fields points.npz -o fields.npz
    python -m locus_es_qe export SAVE --fields fields.npz -o es_input.npz [--soc VBM CBM]
    locus-es run es_input.npz -o es.json --s 0.7084

The `export` step rebuilds the point set. It then checks that the fields were computed on the same points, by comparing the number of points and the weighted sums of n, σ and τ to a relative 1e-8.

## 3. Files read

**`data-file-schema.xml` (`output` section)**

- Lattice vectors, atomic positions and species (bohr), with `alat`.
- The UPF file of each species.
- `nelec`, `nbnd`, and the Monkhorst–Pack mesh and shift.
- The dense FFT grid (`fft_grid`), `ecutwfc`, and the functional.
- The flags `lsda`, `noncolin`, `spinorbit` and `nosym`.
- For every k-point:
  - the Cartesian coordinates, in units of 2π/alat (converted to bohr⁻¹);
  - the weight, which includes the spin degeneracy (the weights sum to 2);
  - the eigenvalues, in hartree.

**`wfc<k>.hdf5`**

The Miller indices G of the plane waves and the coefficients c_nG of all bands, normalized so that ⟨ψ|S|ψ⟩ = 1.

**UPF datasets**

| Quantity | Stored as |
|---|---|
| Radial mesh r and weights rab = dr/di | as given |
| Projectors | r β |
| All-electron and pseudo partial waves | r φ, r φ̃ |
| All-electron and pseudo core densities | n_c, ñ_c (not multiplied by 4πr²) |
| GIPAW core orbitals | r ψ_c |
| Generation input of `ld1.x` | text |

## 4. Datasets

**Projectors.** The states of the generation input are matched in order to the projector labels. States that are not projectors are skipped; `ld1.x` lists them with a negative occupation.

A projector is **bound** if it was generated at the eigenvalue of an atomic state, i.e. with energy 0.00 in the generation input. Its occupation is the one in that input.

**Sphere radius.** The sphere ends 3 mesh points beyond the last point where any all-electron and pseudo partial wave, or the all-electron and pseudo core density, differ by more than 1e-12 of their maximum.

**Core orbitals.**

1. Start from the GIPAW core orbitals.
2. Drop those whose (n, l) is a valence state of the dataset. Some pslibrary files list valence orbitals there and omit one core shell; for As, 4s is listed and 3d is missing.
3. If the remaining orbitals do not hold all the core electrons of n_c, the rest of n_c must be exactly one closed shell of 2(2l + 1) electrons, within 1e-3. It is then added as that shell, with R = √(4π n_rest / (2(2l + 1))).

The core density rebuilt from the orbitals equals n_c to 1e-14 electrons for all 17 pslibrary 1.0.0 datasets tested.

**Core kinetic-energy density.** For a closed shell of f electrons with radial function R:

    τ_c = Σ f / (8π) [R'² + l(l + 1) R² / r²]

## 5. Projections and density matrices

The channels p = (i, m) use real spherical harmonics up to l = 3 (`harmonics.py`). For atom a at R_a and a state with coefficients c_G at k, with q = k + G:

    ⟨p̃_p|ψ̃⟩ = 4π / √V · i^l · Σ_G c_G Y_lm(q̂) β̃_i(|q|) exp(i q·R_a)
    β̃_i(q) = ∫ (r β_i)(r) j_l(qr) r dr

β̃_i is tabulated every 0.005 bohr⁻¹ and interpolated with a cubic spline.

**Check of the projections.** The exporter computes

    ⟨ψ|S|ψ⟩ = Σ_G |c_G|² + Σ_a P^† q^a P,   q^a_pp' = ⟨φ_p|φ_p'⟩ − ⟨φ̃_p|φ̃_p'⟩

It warns if this deviates from 1 by more than 1e-4. For GaAs the deviation is 1.2e-5.

**Ground-state density matrix.**

    D^a_pp' = Σ_k w_k Σ_{n occupied} Re(P*_np P_np')

Taking the real part accounts for the time-reversal partners that are left out of the k-points.

## 6. Integration points and ground-state fields

The points are written in this order:

1. The dense FFT grid of the run, with weight V/N. Point j is at (j/N)·cell, in C order.
2. For every atom, its sphere points twice: first the all-electron points (weight +), then the pseudo points (weight −).

On every point the exporter writes the spin-summed density n, σ = |∇n|² and τ.

**Grid.**

    ñ = Σ_k w_k Σ_occ |u_nk|² / V
    τ̃ = ½ Σ_k w_k Σ_occ |∇ψ_nk|² / V

The derivatives are taken in reciprocal space with (k + G). σ comes from the reciprocal-space gradient of ñ.

**Spheres.** The points are the dataset's radial mesh up to the sphere radius, times a Lebedev rule of degree 11 (50 directions). The weights are r² dr (Simpson in the mesh index) times the Lebedev weights, as in the GPAW exporter.

    Φ_p = φ_i(r) Y_lm(r̂)
    ∇Φ_p = φ_i' Y_lm r̂ + (φ_i / r) (∇S_lm(r̂) − l Y_lm r̂)

where S_lm is the polynomial of Y_lm. Then:

    n = Σ D Φ Φ
    ∇n = 2 Σ_pp' D_pp' Φ_p' ∇Φ_p
    τ = ½ Σ D ∇Φ·∇Φ

The all-electron points add n_c, its gradient and τ_c.

**No pseudo core.** The pseudo core (ñ_c, and a pseudo core τ) is left out of both the grid and the pseudo sphere points, so the core enters only through the all-electron sphere points. This is still a valid PAW decomposition: the grid and the pseudo spheres carry the same fields, and the core density outside the spheres is negligible (0.05 of 64 electrons in GaAs).

The reason is accuracy. The bare-shift fields are strongly nonlinear in α = (τ − τ_W)/τ_unif, and the dense FFT grid does not resolve the pseudized core. GaAs shows the effect on ⟨k_F e^{−α} G(s)⟩, against the GPAW record:

| Variant | VBM | CBM |
|---|---|---|
| Pseudo core density only, no pseudo core τ | +4.8 % | +4.0 % |
| Pseudo core density and τ̃_c = τ_c (ñ_c/n_c)^{5/3} | +4.9 % | +10.7 % |
| No pseudo core (adopted) | +1.1 % | +0.7 % |

The values that do not depend on α (⟨k_F⟩, ⟨k_F G(s)⟩) agree with GPAW within 1 % in all three variants.

**Check.** The valence electrons integrated over the points equal `nelec` (GaAs: 17.99998).

## 7. State densities and bare-shift integrals

For a group of bands at k-point k, the state density is one electron spread evenly over the bands:

- on the grid: the mean of |u_n|²/V over the bands;
- on the spheres: the group's density matrix Re(P* P) / n_bands, with the all-electron and the pseudo partial waves (no core).

There is no symmetrization over the star of k. The weight fields are invariant under the symmetry of the ground state, so the integrals are the same.

    lev = (Σ_p w_p dn_p u_V,p, Σ_p w_p dn_p u_C,p)
    norm = Σ_p w_p dn_p        (≈ 1)

## 8. The other entries of the input file

**Momentum matrix elements.** For every pair n < m with n occupied and m empty:

    M_v = Σ_G c*_nG (k + G)_v c_mG − i Σ_a P_n^a† N^a_v P_m^a
    N^a_v,pp' = ⟨φ_p|∂_v φ_p'⟩ − ⟨φ̃_p|∂_v φ̃_p'⟩

N is computed on the sphere points. The file stores p2 = Σ_v |M_v|² / 3.

**Occupations.** The first nocc = nelec / 2 bands are occupied (1) and the others are empty (0). A run without a gap is rejected.

**Band edges.** The VBM is band nocc − 1 and the CBM band nocc. At each edge the exporter takes the extremum over the k-points; if several k-points share it, the first one is used. The bands within 1 meV of the edge at that k-point form the edge.

**d groups.** For every k-point and element: the occupied bands whose weight on the occupied, bound d projectors of that element is at least 0.5 of Σ|P|² over all projectors.

**Channels.** A channel is a pair (element, l) for which the element has a bound projector of that l. `chanw` is |P|² summed over those bound projectors.

**Free-atom shells.** These are written for the dominant channel of each edge, as the localized-carrier screening of E_S needs them.

- The atom is the dataset's generation configuration: the core orbitals plus the bound valence partial waves with their occupations.
- The shell (n, l) uses the same n as the GPAW exporter (`tables.SHELL_N`). It is taken from a bound partial wave or a core orbital of the dataset.
- If the dataset has no such state, the shell is not written and its name is listed in `meta.missing_shells`.

**Other entries.**

- `nv_bohr3`: the outer s and p electrons of the neutral atoms (`tables.NV_SP`, as in the GPAW exporter), divided by the volume.
- `atom_class`: the chemical species.

**k-point arrays.**

- `ibzk_kc`: the k-points of the run, in scaled coordinates.
- `bzk_kc`: the full mesh.
- `bz2ibz_k`: maps each mesh point to the k-point of the run that equals it or its negative. If a mesh point has no such k-point, the run used symmetry and is rejected.

## 9. Spin–orbit coupling

The exporter does not compute spin–orbit coupling. `--soc VBM CBM` stores given edge shifts in eV, for example from a separate fully relativistic run. Note that the paper's GPAW protocol uses a second-variation window instead.

## 10. Differences from the GPAW exporter

| Item | GPAW exporter | QE exporter |
|---|---|---|
| Ground state | GPAW calculator in memory | rebuilt from the occupied wave functions of the nscf run |
| Empty bands | locked band extension at the converged density | `nscf` run of `pw.x` |
| k-points | irreducible wedge with symmetry | full mesh reduced by time reversal (`nosym`) |
| Pseudo core on the grid | ñ_c and τ̃_c of the setup | none (§6) |
| Core τ | from the setup | from the core orbitals (§4) |
| Spin–orbit coupling | second variation in a window | given shifts only |
| Free atom | GPAW all-electron atom, neutral ground state | generation configuration of the dataset (Zn and Cd in pslibrary are slightly ionic) |

## 11. Validation

See `VALIDATION.md` of this directory.
