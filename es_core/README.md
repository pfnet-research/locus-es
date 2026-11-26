# locus_es

E_S on a code-neutral input file. NumPy only; no DFT code is imported.

    pip install .
    locus-es fields points.npz -o fields.npz
    locus-es run es_input.npz -o es.json

Interfaces (a screening g(r) from the bulk g of each material; `src/locus_es/interface.py`):

    locus-es fields points.npz --layers layers.json -o fields.npz
    locus-es interface window.npz --layers layers.json -o iface.json

The input is written by an exporter of the DFT code (GPAW: `gpaw_plugin/`); format in `docs/ES_INPUT_FORMAT.md`.
