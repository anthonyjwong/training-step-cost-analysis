import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

from utils import getenv_flag, project_dir

# zones to try in order
CPU_ZONES = "us-central1-a us-central1-b us-central1-c us-central1-f"
GPU_ZONES = "us-central1-a us-central1-b us-central1-c"

# smoke: the whole workflow shrunk to its smallest size
SMOKE_OVERRIDES = "STEPS=1 WARMUP=0 BATCH_SIZES=8"


@dataclass
class Environment:
    name: str
    command: list[str]

    @classmethod
    def from_mac(cls, name: str, configuration: str) -> "Environment":
        return cls(name, ["scripts/run_mac.sh", configuration])

    @classmethod
    def from_gcp(
        cls,
        name: str,
        machine_type: str,
        env_dir: str,
        threads: int,
        zones: str,
        sweep_threads: str = "",
    ) -> "Environment":
        command = ["scripts/run_gcp.sh", name, machine_type, env_dir, str(threads)]
        command += [zones] + ([sweep_threads] if sweep_threads else [])
        return cls(name, command)


# in the order they run: the study, then the real-world runs outside it
ENVIRONMENTS = [
    Environment.from_mac("apple-cpu", "cpu"),
    Environment.from_mac("apple-gpu", "gpu"),
    Environment("apple-cpu-vm", ["scripts/run_vagrant.sh"]),
    Environment.from_gcp("gcp-cpu", "n2-standard-4", "gcp-cpu", 4, CPU_ZONES, "1 2 4"),
    Environment.from_gcp("gcp-gpu", "g2-standard-4", "gcp-gpu", 4, GPU_ZONES),
    Environment.from_mac("irl-apple-cpu", "irl-cpu"),
    Environment.from_mac("irl-apple-gpu", "irl-gpu"),
    Environment.from_gcp("irl-gcp-cpu", "n2-standard-16", "gcp-cpu", 16, CPU_ZONES),
    Environment.from_gcp("irl-gcp-gpu", "g2-standard-16", "gcp-gpu", 16, GPU_ZONES),
]


@dataclass
class Args(argparse.Namespace):
    envs: list[str]
    smoke: bool
    dry_run: bool


def get_environments(names: list[str]) -> list[Environment]:
    envs = {e.name: e for e in ENVIRONMENTS}
    unknown = [n for n in names if n not in envs]
    if unknown:
        raise ValueError(
            f"unknown environment {', '.join(unknown)}; choose from {', '.join(envs)}"
        )

    return [envs[n] for n in names] if names else ENVIRONMENTS


def measure(envs: list[Environment], smoke: bool, dry_run: bool) -> list[str]:
    shell_vars = dict(os.environ, PYTHON=sys.executable)
    if smoke:
        shell_vars["RUN_OVERRIDES"] = SMOKE_OVERRIDES

    failed: list[str] = []
    for env in envs:
        print(
            f"== {env.name}: {subprocess.list2cmdline(env.command)}",
            flush=True,
        )
        if dry_run:
            continue
        result = subprocess.run(env.command, cwd=project_dir, env=shell_vars)
        if result.returncode != 0:
            failed.append(env.name)

    return failed


def write_output(failed: list[str]):
    if failed:
        print(f"failed: {', '.join(failed)}; run them again with:")
        print(f"  python src/measure.py {' '.join(failed)}")


def main():
    # default to env; the flags turn each on
    args = Args(envs=[], smoke=getenv_flag("SMOKE"), dry_run=getenv_flag("DRY_RUN"))
    parser = argparse.ArgumentParser()
    parser.add_argument("envs", nargs="*", help="default: all of them")
    parser.add_argument("--smoke", action="store_true", help=SMOKE_OVERRIDES)
    parser.add_argument(
        "--dry-run", action="store_true", help="print each command, run nothing"
    )
    args = parser.parse_args(namespace=args)

    failed = measure(get_environments(args.envs), args.smoke, args.dry_run)

    write_output(failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
