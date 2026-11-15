#!/bin/bash
# One run of a Mac configuration: cpu, gpu, or irl-cpu / irl-gpu (all cores).
# Usage: scripts/run_mac.sh cpu|gpu|irl-cpu|irl-gpu
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

PY=${PYTHON:-.venv/bin/python}
# measure_/ is a package in src/; run its modules from the project root, so out/ is here
export PYTHONPATH=src
CORES=$(sysctl -n hw.ncpu)

measure() { # base name, then extra flags for peaks and benchmark
  local base=$1
  shift
  local name
  name=$(next_run "$base" peaks/peaks benchmark/bench)
  echo "== $name"
  env $RUN_OVERRIDES "$PY" -m measure_.peaks --env "$name" "$@"
  env $RUN_OVERRIDES "$PY" -m measure_.benchmark --env "$name" "$@"
}

case ${1:-} in
cpu)
  measure apple-cpu
  # E1: where the compute peak comes from
  for t in 1 2 4 8; do
    env $RUN_OVERRIDES "$PY" -m measure_.peaks --env "$(next_run "apple-cpu-t$t" peaks/peaks)" --threads "$t"
  done
  mkdir -p out/versions
  write_once out/versions/cpu_apple-cpu.txt sh -c "sysctl -n machdep.cpu.brand_string hw.ncpu; $PY -m measure_.versions"
  write_once out/versions/torch_config_apple-cpu.txt "$PY" -m measure_.versions --config
  write_once out/versions/sysctl_apple-cpu.txt sysctl machdep.cpu.brand_string hw.ncpu \
    hw.perflevel0.name hw.perflevel0.physicalcpu hw.perflevel0.l2cachesize \
    hw.perflevel1.name hw.perflevel1.physicalcpu hw.perflevel1.l2cachesize \
    hw.optional.arm.FEAT_SME hw.optional.arm.FEAT_SME2 hw.memsize
  ;;
gpu) measure apple-gpu --device mps ;;
irl-cpu) measure irl-apple-cpu --threads "$CORES" ;;
irl-gpu) measure irl-apple-gpu --device mps --threads "$CORES" ;;
*)
  echo "usage: $0 cpu|gpu|irl-cpu|irl-gpu" >&2
  exit 2
  ;;
esac
