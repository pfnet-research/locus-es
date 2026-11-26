# Reproduction data

- `PROTOCOL.md`: the calculation settings.
- `MANIFEST.csv`: one line per structure, giving its file, number of atoms, k-mesh, where it is used in the paper, and its database identifier.
- `structures/*.json`: the structures. Each file has the form `{units: "Angstrom", n, entries: {key: {...}}}`. Each entry holds:
  - `symbols`, `positions_A`, `cell_A`, `pbc`;
  - `kpts`, `ecut` and `setups` of the calculation;
  - `initial_magnetic_moments` where needed;
  - `source`: the database identifiers and how the structure was made;
  - `used_in`.

| File | Contents |
|---|---|
| `crystals.json` | Every crystal of the gap benchmark: Borlido et al. plus the added crystals. Materials Project structures rescaled to the ICSD volume where available, then standardised to the primitive cell. |
| `perovskites.json` | Ideal, polymorphous and molecular-dynamics structures of the halide perovskites, including the 320-atom cells of the cost scaling. |
| `alloys.json` | SQS of InGaN, AlGaN and Cu(In,Ga)Se2. |
| `amorphous.json` | Melt-quenched a-SiO2, a-Al2O3, a-HfO2 and a-IGZO. |
| `surfaces.json` | Relaxed symmetric slabs of the ionization-potential benchmark. |
| `molecular_crystals.json` | Molecular crystals. |
| `open_shell_oxides.json` | Open-shell oxides with their magnetic order. |
| `cell_representation.json` | Supercells of the cell-representation test. |
| `external_ids.json` | Systems used exactly as distributed by a database, listed by identifier only: C2DB monolayers, the Materials Project entries of the independent validation set, GW100 and GMTKN55. |

Read an entry with ASE:

    from ase import Atoms
    e = json.load(open("structures/perovskites.json"))["entries"]["CsPbI3_cub_md2_T300_s1_1"]
    atoms = Atoms(e["symbols"], positions=e["positions_A"], cell=e["cell_A"], pbc=e["pbc"])
