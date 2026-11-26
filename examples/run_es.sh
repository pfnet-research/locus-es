#!/bin/bash
# E_S of one system of the paper, from its structure: LOCUS ground state (GPAW), export, E_S.
#   examples/run_es.sh data/structures/crystals.json GaAs_107946 [extra export options] [-- extra E_S options]
# Needs: GPAW 24.1 with gpaw_plugin installed (pip install ./gpaw_plugin), es_core installed (pip install ./es_core), and
# GPAW_SETUP_PATH = gpaw_plugin/setups:<GPAW setups 24.11.0>.
# As in the paper:  supercells (molecular-dynamics snapshots, amorphous cells, SQS): export with --soc-window nsites;
#   molecular-dynamics snapshots: E_S with --no-loc;  SQS alloys: export with --window 40 16 (the gap with the switch per state);
#   amorphous cells: export with --window 40 12 --no-soc, E_S with --no-loc (the mobility gap between states of uniformity >= 0.4);
#   monolayers: E_S with --slab;  setups with f projectors (La, Ce, ...): export with --no-soc.
# Parallel: set MPI="mpiexec -n N" (GPAW with MPI) to run the ground state and the two exporter steps in parallel over k-points and
# real-space domains.  Large Gamma-only cells with a gap: also set GSOPT=--large-cell (only the occupied bands + 8 are converged; data/PROTOCOL.md).
set -e
S=$1; K=$2; shift 2
XOPT=(); EOPT=(); sep=0
for a in "$@"; do if [ "$a" = "--" ]; then sep=1; elif [ $sep = 0 ]; then XOPT+=("$a"); else EOPT+=("$a"); fi; done
$MPI python -m locus_gpaw.protocol "$S" "$K" -o "$K.gpw" --txt "$K.txt" ${GSOPT:-}
$MPI python -m locus_es_gpaw points "$K.gpw" -o "$K.points.npz"
locus-es fields "$K.points.npz" -o "$K.fields.npz"
$MPI python -m locus_es_gpaw export "$K.gpw" --fields "$K.fields.npz" -o "$K.es_input.npz" "${XOPT[@]}"
locus-es run "$K.es_input.npz" -o "$K.es.json" "${EOPT[@]}"
