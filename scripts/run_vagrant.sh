#!/bin/bash
# One run in the local VM: start it, measure, and always halt it again, even on failure.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

PY='~/venv/bin/python'

halt_vm() { (cd envs/vagrant && vagrant halt) || echo "warning: vagrant halt failed" >&2; }
trap halt_vm EXIT

# a VM that failed to boot often comes up on the next try
for try in 1 2 3; do
  (cd envs/vagrant && vagrant up) && break
  echo "vagrant up failed (try $try)" >&2
  [ "$try" = 3 ] && exit 1
  (cd envs/vagrant && vagrant halt --force) || true
done

# run names are chosen here, on the shared out/ folder, before running in the VM
name=$(next_run apple-cpu-vm peaks/peaks benchmark/bench)
mkdir -p out/versions
cmds="$PY -m measure_.peaks --env $name; $PY -m measure_.benchmark --env $name"
for t in 1 2 4; do
  cmds+="; $PY -m measure_.peaks --env $(next_run "apple-cpu-vm-t$t" peaks/peaks) --threads $t"
done
[ -e out/versions/cpu_apple-cpu-vm.txt ] || cmds+="; (lscpu; $PY -m measure_.versions) > out/versions/cpu_apple-cpu-vm.txt"
[ -e out/versions/torch_config_apple-cpu-vm.txt ] || cmds+="; $PY -m measure_.versions --config > out/versions/torch_config_apple-cpu-vm.txt"
echo "== $name"

# the project folder is shared, so the VM writes straight into out/
(cd envs/vagrant && vagrant ssh -c "set -e; cd ~/proj1; export PYTHONPATH=src ENV_FILE=envs/vagrant/.env ${RUN_OVERRIDES}; $cmds")
