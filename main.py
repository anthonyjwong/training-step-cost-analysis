import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from utils import getenv, getenv_flag  # noqa: E402


@dataclass
class Args(argparse.Namespace):
    smoke: bool
    dry_run: bool
    runs: int


def run_stage(stage: str, flags: list[str]) -> int:
    print(f"\n### {stage}", flush=True)
    command = [sys.executable, os.path.join("src", f"{stage}.py")] + flags
    return subprocess.run(command, cwd=PROJECT_DIR).returncode


def run_stage_cached(stage: str, output: str) -> int:
    if os.path.exists(os.path.join(PROJECT_DIR, output)):
        print(f"\n### {stage}: {output} exists, skipped (delete it to rerun)")
        return 0
    return run_stage(stage, [])


def check_gcloud_login():
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit("gcloud login expired: run gcloud auth login")


def keep_mac_awake():
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


def main():
    # default to env
    args = Args(
        smoke=getenv_flag("SMOKE"),
        dry_run=getenv_flag("DRY_RUN"),
        runs=getenv("RUNS", 1, int),
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="the whole experiment, but shrunk to its smallest size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="the whole experiment, but mock each environment's script call",
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="how many times to measure every environment (default 1)",
    )
    args = parser.parse_args(namespace=args)
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if not args.dry_run:
        check_gcloud_login()
    keep_mac_awake()

    # setup stages
    for stage, output in (
        ("prices", "out/prices/prices.csv"),
        ("estimate", "out/estimate/arithmetic_intensity.csv"),
    ):
        if run_stage_cached(stage, output) != 0:
            sys.exit(f"{stage} failed")

    # experiment stage
    measure_flags = ["--smoke"] * args.smoke + ["--dry-run"] * args.dry_run
    success = 0
    for run in range(args.runs):
        print(f"\n## run {run} of {args.runs}", flush=True)
        success |= run_stage("measure", measure_flags)

    # stats stage
    for stage in ("summarize", "costs", "charts"):
        if run_stage(stage, []) != 0:
            sys.exit(f"{stage} failed")

    if success != 0:
        sys.exit("some environments failed")


if __name__ == "__main__":
    main()
