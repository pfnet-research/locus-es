# Files exchanged between a DFT code and E_S (format version 1)

An E_S calculation runs in four steps. Steps 1 and 3 run in the DFT code's exporter (for GPAW: `gpaw_plugin/locus_es_gpaw`); steps 2 and 4 run in E_S (`es_core/locus_es`).

    1. exporter  points  -> points.npz   integration points and the density on them
    2. E_S       fields  -> fields.npz   bare-shift fields u_V, u_C on these points (without k_V, k_C)
    3. exporter  export  -> input.npz    everything else, including the integrals of u_V and u_C over the state densities
    4. E_S       run     -> es.json      corrected levels

For interfaces, a screening g(r) is built from the bulk g of each material (`es_core/locus_es/interface.py`). Step 2 then also writes the layer fields (`locus-es fields --layers layers.json`). The exporter's `window` step writes their matrices between the states near the gap, and `locus-es interface` gives the band edges of each layer.

The forms of the bare shifts belong to E_S, so they never appear in the exporter. The exporter only integrates the weight fields it receives.

All files are NumPy `.npz` archives. Each holds a JSON string under the key `meta`, which contains `format` and `version` (1). Quantities are in Hartree atomic units unless their name says otherwise (`_ev`, `_A`).

## points.npz (`format: locus-es-points`)

| Key | Shape | Meaning |
|---|---|---|
| `w` | (P,) | Integration weights. PAW codes give the all-electron sphere points weight + and the pseudo sphere points weight −. |
| `n`, `sigma`, `tau` | (P,) | Spin-summed density, \|∇n\|² of the total density and kinetic-energy density, core included. |
| `meta.fingerprint` | | [P, Σw, Σw·n, Σw·sigma, Σw·tau]. Step 3 checks it against its own points to within 1e-8. |

A norm-conserving code supplies grid points only.

Optional entries, needed only for a screening g(r) at interfaces (`locus-es fields --layers`):

| Key | Shape | Meaning |
|---|---|---|
| `atom_p` | (P,) | −1 on the grid points, which come first; the atom index on the sphere points. |
| `meta.grid_N`, `meta.grid_beg`, `meta.grid_n` | | The grid points are the points i ∈ [beg, beg + n) of a periodic grid of N points per lattice vector, at (i / N) · cell, in C order. |
| `meta.cell_bohr`, `meta.spos`, `meta.symbols` | | Cell, scaled atomic positions and symbols. |
| `meta.atom_orbit` | (N,) | Symmetry orbit of every atom, or `null` without symmetry. |

## fields.npz (`format: locus-es-fields`)

`u_v`, `u_c` (P,) in bohr⁻¹. The points fingerprint is copied into `meta.points_fingerprint`.

With `--layers`, the file also holds, for every layer M, the weight `h:<M>` and the products `uv:<M>` = h_M u_V and `uc:<M>` = h_M u_C. Their names are listed in `meta.matrix_fields`, the layers in `meta.layers`.

## window.npz (`format: locus-es-window`)

This file is written by the exporter's `window` step from a fields file with `matrix_fields`. For every k-point / spin entry u, it holds the `n_occ` highest occupied and the `n_emp` lowest empty bands.

| Key | Shape | Meaning |
|---|---|---|
| `w_k`, `spin_k`, `ibz_of_u` | (K,) | As in the input file. |
| `bands_occ`, `eps_occ` | (K, n_occ) | Band indices and eigenvalues (Ha) of the occupied block. |
| `bands_emp`, `eps_emp` | (K, n_emp) | The same for the empty block. |
| `M:<field>:occ`, `M:<field>:emp` | (K, n, n) complex | <m\|F\|n> over the points, including the PAW one-centre parts. |

The empty bands of the window are converged at the ground-state density before the matrices are formed.

## input.npz (`format: locus-es-input`)

### meta

| Field | Meaning |
|---|---|
| `n_electrons` | number of valence electrons |
| `spin_paired` | whether the calculation is spin-paired |
| `symbols`, `positions_A`, `cell_A`, `cell_bohr`, `pbc` | geometry |
| `nbzkpts` | number of k-points in the full Brillouin zone |
| `nv_bohr3` | free-atom valence sp electron density, Σ(outer s + p) / V |
| `channels` | list of `[element, l]` of the bound projectors |
| `edges` | `{vbm, cbm: {u, k, bands, eps, occupation}}`. `u` indexes the k-point/spin entries; `bands` are the bands degenerate with the edge within 1 meV. |
| `groups` | list of `{kind: "d" \| "window", u, k, bands, occupied, eps, weight, occupation, element?, d_ref_ev?}` |
| `window` | `[below, above]` for the window groups, or `null` |
| `soc` | `{vbm, cbm}`: spin–orbit edge shifts in Ha, or `null` |

**Groups.**
- `d` groups are the occupied bands with bound-d character ≥ 0.5 on one element, at one k-point. `d_ref_ev` (written only with `export --ref-xc`) is the first-order level shift of a reference functional relative to PBE, <H_ref − H_PBE>, at the base density; it enters only with `mu_d > 0` (the paper uses `mu_d = 0`).
- `window` groups are every degenerate set of bands in [nocc − below, nocc + above), at every k-point.

### Arrays

| Key | Shape | Meaning |
|---|---|---|
| `eps_kn`, `occ_kn` | (K, B) | Eigenvalues and occupation fractions of the irreducible k-point / spin entries. The bands include 2 × nocc empty ones. |
| `w_k`, `w_k_raw` | (K,) | k-point weights, including the spin degeneracy. |
| `spin_k`, `ibz_of_u` | (K,) | Spin and irreducible k-point index of each entry. |
| `ibzk_kc`, `bzk_kc`, `bz2ibz_k` | | k-points in scaled coordinates, and the map from the full zone to the irreducible wedge. |
| `pair_n`, `pair_m`, `p2_kp` | (Np,), (Np,), (K, Np) | Band pairs n < m with max_k f_n (1 − f_m) > 1e-8, and \|<n\|p\|m>\|² averaged over the three directions, including the PAW correction. |
| `lev` | (2 + G, 2) | ∫ dn u_V and ∫ dn u_C for the VBM, the CBM and the groups, where dn is the density of one electron spread over the group's bands. |
| `norm` | (2 + G,) | ∫ dn |
| `chanw_knc` | (K, B, C) | Projection weight of every state on every channel. |
| `atom_class` | (N,) | Classes of equivalent atoms (symmetry orbits, or species when the cell has no symmetry). |
| `atomw_gi` | (G, N) | Atomic projection weights of the window groups. |
| `shell:<el>_<l>:r_g`, `w_g`, `rho_g`, `n_g`, `sigma_g`, `tau_g` | | Free neutral atom (PBE, scalar-relativistic) for the dominant channel of each band edge: radial grid, weights 4πr²dr, normalized shell density, and the atomic n, \|∇n\|² and τ. |
