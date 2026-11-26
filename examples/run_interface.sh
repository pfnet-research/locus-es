#!/bin/bash
# E_S with a screening g(r) for a cell with two or more materials (superlattice, heterojunction), from its LOCUS ground state.
#   examples/run_interface.sh sl.gpw layers.json [sl.es.json]
# layers.json (es_core/src/locus_es/interface.py): the atoms of every layer and its g, a number or the es.json of the bulk material
# computed at the lattice of the cell (examples/run_es.sh on the bulk).  The ground state needs no empty bands: the window step converges
# them.  The optional es.json of the cell itself (E_S with one g; run_es.sh) adds the first-order levels with that g for comparison.
# Large cells: set MPI="mpiexec -n N" (GPAW with MPI) to run the exporter steps in parallel over k-points.
set -e
G=$1; L=$2; ONE=$3; K=${G%.gpw}
$MPI python -m locus_es_gpaw points "$G" -o "$K.points.npz"
locus-es fields "$K.points.npz" --layers "$L" -o "$K.fields_layers.npz"
$MPI python -m locus_es_gpaw window "$G" --fields "$K.fields_layers.npz" -o "$K.window.npz" --occ 24 --emp 16
locus-es interface "$K.window.npz" --layers "$L" -o "$K.iface.json" ${ONE:+--g-one "$ONE"}
