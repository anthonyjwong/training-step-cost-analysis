#!/bin/bash
# One run on a GCP instance: create it, set it up, measure, copy the output back, and always
# delete it, even on failure, since a running instance keeps billing.
# Usage: scripts/run_gcp.sh BASE MACHINE ENV_DIR THREADS "ZONES" ["SWEEP THREADS"]
#   e.g. scripts/run_gcp.sh gcp-cpu n2-standard-4 gcp-cpu 4 "us-central1-a us-central1-b" "1 2 4"
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

base=$1 machine=$2 env_dir=$3 threads=$4 zones=$5 sweep=${6:-}
vm="proj1-$base"
PY='~/venv/bin/python'

case $env_dir in
gcp-cpu) create_flags="--image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud --boot-disk-size 30GB" ;;
# GPU instances need an image with the NVIDIA driver and cannot live-migrate
gcp-gpu) create_flags="--image-family common-cu129-ubuntu-2404-nvidia-580 --image-project deeplearning-platform-release --boot-disk-size 60GB --maintenance-policy TERMINATE" ;;
*)
  echo "unknown env dir $env_dir" >&2
  exit 2
  ;;
esac

delete_vm() { # delete $vm in whatever zone it is in
  local z
  for z in $(gcloud compute instances list --format='value(name,zone)' | awk -v vm="$vm" '$1 == vm {print $2}'); do
    gcloud compute instances delete "$vm" --zone "$z" --quiet
  done
}
zone=""
fetched=$(mktemp -d)
cleanup() {
  delete_vm || echo "WARNING: could not delete $vm; delete it by hand, it is billing" >&2
  rm -rf "$fetched"
}
trap cleanup EXIT

# an instance left over from a failed run would keep billing and block the name
delete_vm

# GPUs are often out of stock and come back within minutes, so try each zone in turn,
# then wait and try them all again, for up to CREATE_MINUTES
create_minutes=${CREATE_MINUTES:-30}
deadline=$((SECONDS + create_minutes * 60))
while [ -z "$zone" ]; do
  for z in $zones; do
    echo "creating $vm ($machine) in $z"
    if gcloud compute instances create "$vm" --zone "$z" --machine-type "$machine" $create_flags; then
      zone=$z
      break
    fi
  done
  [ -n "$zone" ] && break
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "no zone had capacity for $machine in $create_minutes minutes: $zones" >&2
    exit 1
  fi
  echo "no capacity in $zones; trying again in 2 minutes"
  sleep 120
done

vm_ssh() { gcloud compute ssh "$vm" --zone "$zone" --command "$1"; }
# a new instance refuses SSH until it has booted
for i in $(seq 30); do vm_ssh true 2>/dev/null && break || sleep 10; done
vm_ssh 'mkdir -p ~/proj1/src ~/proj1/out/versions'
gcloud compute scp --recurse --zone "$zone" src "$vm:~/proj1/"
gcloud compute scp --zone "$zone" "envs/$env_dir/setup.sh" "$vm:~/proj1/"
gcloud compute scp --zone "$zone" "envs/$env_dir/.env" "$vm:~/proj1/.env.local"
vm_ssh 'sudo bash ~/proj1/setup.sh'

# the project's quota, as the report cites it
mkdir -p out/versions
[ -e out/versions/gcp_quota.txt ] || {
  gcloud compute project-info describe --flatten=quotas \
    --format="value(quotas.metric,quotas.limit,quotas.usage)" |
    awk '$1=="CPUS_ALL_REGIONS"||$1=="GPUS_ALL_REGIONS"'
  gcloud compute regions describe us-central1 --flatten=quotas \
    --format="value(quotas.metric,quotas.limit,quotas.usage)" |
    awk '$1=="NVIDIA_L4_GPUS"{print "us-central1", $0}'
} >out/versions/gcp_quota.txt

# run names are chosen from the local out/, which is what must not be overwritten
name=$(next_run "$base" peaks/peaks benchmark/bench)
cmds="$PY -m measure_.peaks --env $name --threads $threads; $PY -m measure_.benchmark --env $name --threads $threads"
for t in $sweep; do
  cmds+="; $PY -m measure_.peaks --env $(next_run "$base-t$t" peaks/peaks) --threads $t"
done
info="(lscpu; command -v nvidia-smi >/dev/null && nvidia-smi; $PY -m measure_.versions)"
[ -e "out/versions/cpu_$base.txt" ] || cmds+="; $info > out/versions/cpu_$base.txt"
[ -e "out/versions/torch_config_$base.txt" ] || cmds+="; $PY -m measure_.versions --config > out/versions/torch_config_$base.txt"
echo "== $name on $machine in $zone"
vm_ssh "set -e; cd ~/proj1; export PYTHONPATH=src; ${RUN_OVERRIDES:+export $RUN_OVERRIDES;} $cmds"

gcloud compute scp --recurse --zone "$zone" "$vm:~/proj1/out/*" "$fetched/"
move_new "$fetched"
