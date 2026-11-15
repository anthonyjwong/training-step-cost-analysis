import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import torch

from utils import (
    MachineInfo,
    get_seed,
    getenv,
    wait_for_device,
    write_csv,
)


@dataclass
class Roof:
    env: str
    seed: int
    machine: MachineInfo
    peak_gflops: float
    peak_gbs: float
    ridge: float

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> Self:
        return cls(
            env=row["env"],
            seed=int(row["seed"]),
            machine=MachineInfo.from_csv_row(row),
            peak_gflops=float(row["peak_gflops"]),
            peak_gbs=float(row["peak_gbs"]),
            ridge=float(row["ridge"]),
        )


def get_best_time(
    run: Callable[[], object], device: torch.device, retries: int
) -> float:
    run()  # warm up
    wait_for_device(device)

    best = float("inf")
    for _ in range(retries):
        start = time.perf_counter()
        run()
        wait_for_device(device)
        best = min(best, time.perf_counter() - start)

    return best


def get_system_peak_computation_capability(device: torch.device, retries: int) -> float:
    """Math-heavy operation to test CPU boundary"""
    MATRIX_SIZE = 4096

    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    seconds = get_best_time(lambda: a @ b, device, retries)

    return 2 * MATRIX_SIZE**3 / seconds


def get_system_peak_bandwidth(device: torch.device, retries: int) -> float:
    """Data-heavy operation to test memory boundary"""
    ARRAY_BYTES = 1024 * 2**20  # 1 GB
    BYTES_PER_FLOAT = 4
    ALPHA = 3.0

    x = torch.randn(ARRAY_BYTES // BYTES_PER_FLOAT, device=device)
    y = torch.randn(ARRAY_BYTES // BYTES_PER_FLOAT, device=device)
    z = torch.empty_like(x)
    seconds = get_best_time(
        lambda: torch.add(x, y, alpha=ALPHA, out=z), device, retries
    )

    return 3 * x.nbytes / seconds  # read x and y, write z


@dataclass
class Args(argparse.Namespace):
    env: str
    device: str
    threads: int
    seed: int


PEAKS_CSV = "out/peaks/peaks_{env}.csv"


def write_output(roof: Roof):
    out = PEAKS_CSV.format(env=roof.env)
    write_csv(out, [roof])

    print(roof.machine)
    print(f"seed           {roof.seed} (replay with SEED={roof.seed})")
    print(f"peak           {roof.peak_gflops:10.1f} GFLOP/s")
    print(f"bandwidth      {roof.peak_gbs:10.1f} GB/s")
    print(f"ridge          {roof.ridge:10.2f} FLOP/byte")
    print(f"wrote {out}")


def main():
    RETRIES = 10

    # default to env
    args = Args(
        env=getenv("RUN_ENV", "", str),
        device=getenv("DEVICE", "cpu", str),
        threads=getenv("THREADS", 0, int),
        seed=get_seed(),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env")
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(namespace=args)
    if not args.env:
        parser.error("set RUN_ENV in .env or pass --env")

    out = PEAKS_CSV.format(env=args.env)
    if os.path.exists(out):
        raise FileExistsError(
            f"{out} already exists, delete it manually first if you would like to overwrite it"
        )

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    peak_cc = get_system_peak_computation_capability(device, RETRIES)
    peak_bandwidth = get_system_peak_bandwidth(device, RETRIES)

    roof = Roof(
        env=args.env,
        seed=args.seed,
        machine=MachineInfo.from_device(device),
        peak_gflops=peak_cc / 1e9,
        peak_gbs=peak_bandwidth / 1e9,
        ridge=peak_cc / peak_bandwidth,
    )

    write_output(roof)


if __name__ == "__main__":
    main()
