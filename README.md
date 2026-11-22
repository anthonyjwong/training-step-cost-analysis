# Roofline Analysis of DNN Training on Bare Metal, Virtualized and Cloud CPUs/GPUs

The full write-up, with results and charts can be read at [REPORT.pdf](REPORT.pdf).

## What is this?

An experiment that times a training step of three torchvision CNNs on local and cloud hardware.

It asks three questions:
1. What does one training run cost, in hours and dollars, in each environment?
2. Which hardware limit (compute or memory bandwidth) sets each model's throughput?
3. When does buying a GPU beat renting one?

## Environments

| Name | Where it runs | Run by |
| --- | --- | --- |
| `apple-cpu` | Mac CPU, 4 threads | `scripts/run_mac.sh cpu` |
| `apple-gpu` | Mac GPU (MPS) | `scripts/run_mac.sh gpu` |
| `apple-cpu-vm` | VirtualBox VM on the Mac, 4 CPUs | `scripts/run_vagrant.sh` |
| `gcp-cpu` | GCP `n2-standard-4` | `scripts/run_gcp.sh` |
| `gcp-gpu` | GCP `g2-standard-4` (NVIDIA L4) | `scripts/run_gcp.sh` |
| `irl-apple-cpu`, `irl-apple-gpu` | Mac, all cores | `scripts/run_mac.sh irl-cpu`, `irl-gpu` |
| `irl-gcp-cpu`, `irl-gcp-gpu` | GCP `n2-standard-16`, `g2-standard-16` | `scripts/run_gcp.sh` |


## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The Makefile expects the interpreter at `.venv/bin/python`. Override it with
`make PYTHON=/path/to/python`.

## Running

```sh
make           # the whole experiment
make RUNS=3    # measure every environment three times
```

**`make` and `make smoke` create real GCP instances, which cost money.** `smoke` is
designed around minimum cost to test the workflow.

`make dry-run` is free.

## Settings

Run settings are read from `.env.local` (set `ENV_FILE` to use another file). A
command-line flag overrides the file.

| Variable | Meaning |
| --- | --- |
| `RUN_ENV` | Name of the run, used in output file names |
| `DEVICE` | `cpu`, `mps` or `cuda` |
| `THREADS` | CPU threads for PyTorch |
| `BATCH_SIZES` | Batch sizes to time, e.g. `8 16 32` |
| `STEPS`, `WARMUP` | Timed steps, and untimed warm-up steps before them |
| `SEED` | Random seed. Unset means a new one each run; each run prints its seed |

The VM and the GCP instances use their own copies in `envs/<name>/.env`.

## Layout

```
main.py        runs the whole experiment and statistical analysis
Makefile
src/           one file per stage
scripts/       infra setup
envs/          per-environment settings
REPORT.pdf
```
