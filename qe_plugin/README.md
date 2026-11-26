# locus_es_qe (prototype)

Writes the E_S input (`docs/ES_INPUT_FORMAT.md`) from a Quantum ESPRESSO `pw.x` calculation with PAW datasets and the PBE functional. Only the files written by `pw.x` and the UPF datasets are read; no Quantum ESPRESSO or GPAW code is used.

    pip install ./es_core ./qe_plugin
    # pw.x: scf, then nscf in the same outdir with nosym = .true. and nbnd = nocc + max(16, 2 nocc)
    python -m locus_es_qe points out/prefix.save -o points.npz
    locus-es fields points.npz -o fields.npz
    python -m locus_es_qe export out/prefix.save --fields fields.npz -o es_input.npz
    locus-es run es_input.npz -o es.json --s 0.7084        # s of the PBE base

What is computed and how: `SPEC.md`. Comparison with the GPAW exporter: `VALIDATION.md`.

Requirements of the prototype: PAW datasets in UPF v2 from `ld1.x` with the GIPAW section (pslibrary 1.0.0); spin-paired insulators; `K_POINTS automatic`; wave functions in HDF5 files. Spin–orbit coupling is not computed (`--soc VBM CBM` stores given edge shifts in eV).
