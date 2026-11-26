# Design: separating the model from the DFT code

## E_S

E_S is a post-processing step. It does not change the self-consistent density, energy or forces of the base calculation. It shifts each one-electron level n by

    eps'_n = eps_n + s g Dhat_n + Delta^SOC_n

Each ingredient comes from data that any converged Kohn–Sham calculation can provide.

The paper calls `Dhat_n` the signed bare (exchange) shift of state n and `u_V`, `u_C` (with the coefficients `kV`, `kC`) the bare-shift fields. The code keeps the earlier name *lever* in identifiers, comments and help texts (for example `lever_fields`); it means the same quantity.

| Quantity | Computed by | Data required |
|---|---|---|
| bare shifts `Dhat_n` (`-kV <k_F e^-alpha G(s)>_n`, `+kC <k_F G(s)>_n`, d states x `k_d`) | `es_core` | Integration points with weights; the density, `\|grad n\|^2` and `tau` at the points; the density of each requested state at the points. In PAW the points include the atom-centred sphere points with signed weights (all-electron +, pseudo −). |
| screening `g = 1/(eps_KS + eps_TF)` | `es_core` | Eigenvalues, occupations, k-point weights, isotropic `\|<m\|p\|n>\|^2` of the band pairs, k-point lists and the cell. |
| finite-system switch, Gamma-only guard, monolayer `eps_s` | `es_core` | Eigenvalues, k-points, cell, atomic positions and periodicity. |
| localized-carrier screening | `es_core` | For each band edge: the projection weights of the states on atomic channels (element, l); the free-atom shell density and atomic density, gradient and `tau` on a radial grid; the valence sp electron density of the free atoms. |
| `Delta^SOC_n` | DFT-code side | Spin–orbit coupling in second variation within the band window, computed by the DFT code and written as edge shifts. |
| additional empty bands (the `eps_KS` sum uses 2 x nocc empty bands) | DFT-code side | Written into the eigenvalue and matrix-element arrays. |
| screening g(r) at interfaces: layer weights `h_M`, `g(r) = sum_M g_M h_M`, the operator `V_S` and its diagonalization in the occupied and empty blocks, the band edges of each layer | `es_core` (`interface.py`) | The grid geometry, atomic positions and the atom of every sphere point. The matrices <m\|F\|n> of the layer fields between the states of a window around the gap, with converged empty bands. |

Everything in the left column that is part of the model lives in `es_core`. This covers:

- the forms and coefficients of the bare shifts;
- the screening formula, with its k·p cell average and band extrapolation;
- the switches;
- the localized-screening criterion.

`es_core` is plain NumPy.

The exporter, `gpaw_plugin/locus_es_gpaw`, only extracts data. It:

- evaluates densities on the PAW points;
- computes matrix elements;
- extends the empty bands;
- runs the spin–orbit step;
- solves the free atom.

Another code can produce the same file:

- A norm-conserving code supplies only grid points.
- A PAW code adds sphere points.
- A code without spin–orbit coupling leaves `Delta^SOC` at zero.

## Shell term

The shell term is

    E_sh = c ∫ G(n, p, x) e_x^LDA(n) dr,    x = (G_w * n)/n

It has two parts:

1. **Pointwise part.** The gate `G`, the energy density and the derivatives with respect to `n`, `p` and `x`. This part is independent of the code. It lives in `shell_term/locus_shell` with a NumPy reference implementation.
2. **Descriptor and potential.** These need the code's real-space machinery. The descriptor `x` is a Gaussian convolution of the density whose width varies in space. The potential needs the adjoint of that convolution. In PAW the descriptor and the potential must also be carried into the augmentation spheres.
   - PySCF: implemented on molecular grids in `shell_term/` (PySCF is Apache-2.0).
   - GPAW: needs its own meta-GGA PAW corrections and plane-wave density machinery. `locus_gpaw` (in `gpaw_plugin/`) implements them in a subclass of GPAW 24.1.0's meta-GGA class, together with the rVV10 kernel used by LOCUS, and registers LOCUS with GPAW's xc factory on import. GPAW's source is not changed.

## Reproducing the paper

The published numbers are reproduced with an unmodified GPAW 24.1.0 and `locus_gpaw`. Tests compare `es_core` (on files written by `locus_es_gpaw`) with the reference records of the paper:

- band-edge shifts within 1 meV;
- the screening `g` within 1e-4.
