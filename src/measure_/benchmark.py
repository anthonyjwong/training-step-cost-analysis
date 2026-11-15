import argparse
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torchvision

from utils import (
    MODELS,
    MachineInfo,
    get_seed,
    getenv,
    getenv_list,
    wait_for_device,
    write_csv,
)
from estimate import get_step_complexity


@dataclass
class Benchmark:
    env: str
    seed: int
    machine: MachineInfo
    model_name: str
    batch_size: int
    steps: int
    median_seconds: float
    mean_seconds: float
    stdev_seconds: float
    images_per_second: float
    arithmetic_intensity: float
    achieved_gflops: float


def get_train_step(
    model_name: str, batch_size: int, device: torch.device
) -> Callable[[], None]:
    IMAGE_SIZE = 224
    NUM_CLASSES = 1000

    model = torchvision.models.get_model(model_name, weights=None).to(device).train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )

    # one random batch kept on the device, so only the training step is timed
    images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)

    def train_step():
        optimizer.zero_grad()
        scores = cast(torch.Tensor, model(images))  # torchvision has no type hints
        loss = nn.functional.cross_entropy(scores, labels)
        loss.backward()
        optimizer.step()

    return train_step


def get_step_times(
    train_step: Callable[[], None], device: torch.device, warmup: int, steps: int
) -> list[float]:
    for _ in range(warmup):
        train_step()
    wait_for_device(device)

    times: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        train_step()
        wait_for_device(device)
        times.append(time.perf_counter() - start)

    return times


@dataclass
class Args(argparse.Namespace):
    env: str
    device: str
    threads: int
    models: list[str]
    batch_sizes: list[int]
    steps: int
    warmup: int
    seed: int


BENCHMARK_CSV = "out/benchmark/bench_{env}.csv"


def write_output(env: str, benchmarks: list[Benchmark]):
    # the newest result, then every result so far, so a crash keeps what finished
    newest = benchmarks[-1]
    spread = newest.stdev_seconds / newest.mean_seconds
    print(
        f"{newest.model_name:14} batch {newest.batch_size:<4}"
        + f" {newest.median_seconds * 1e3:9.1f} ms/step"
        + f" {newest.images_per_second:8.1f} img/s"
        + f" {newest.achieved_gflops:9.1f} GFLOP/s"
        + f" (spread {spread:.1%})"
    )
    write_csv(BENCHMARK_CSV.format(env=env), benchmarks)


def main():
    # defaults; command-line flags override them
    args = Args(
        env=getenv("RUN_ENV", "", str),
        device=getenv("DEVICE", "cpu", str),
        threads=getenv("THREADS", 0, int),
        models=getenv_list("MODELS", MODELS, str),
        batch_sizes=getenv_list("BATCH_SIZES", [8, 16, 32], int),
        steps=getenv("STEPS", 10, int),
        warmup=getenv("WARMUP", 2, int),
        seed=get_seed(),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env")
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(namespace=args)
    if not args.env:
        parser.error("set RUN_ENV in .env or pass --env")

    out = BENCHMARK_CSV.format(env=args.env)
    if os.path.exists(out):
        raise FileExistsError(
            f"{out} already exists, delete it manually first if you would like to overwrite it"
        )

    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    machine = MachineInfo.from_device(device)
    print(machine)
    print(f"seed {args.seed}")

    benchmarks: list[Benchmark] = []
    for model_name in args.models:
        for batch_size in args.batch_sizes:
            torch.manual_seed(args.seed)
            try:
                train_step = get_train_step(model_name, batch_size, device)
                times = get_step_times(train_step, device, args.warmup, args.steps)
            except Exception as e:  # e.g. out of memory; skip this run, keep going
                print(
                    f"{model_name} batch {batch_size}: failed ({str(e).splitlines()[0]})"
                )
                continue

            complexity = get_step_complexity(model_name, batch_size)
            median = statistics.median(times)

            benchmark = Benchmark(
                env=args.env,
                seed=args.seed,
                machine=machine,
                model_name=model_name,
                batch_size=batch_size,
                steps=len(times),
                median_seconds=median,
                mean_seconds=statistics.mean(times),
                stdev_seconds=statistics.stdev(times) if len(times) > 1 else 0.0,
                images_per_second=batch_size / median,
                arithmetic_intensity=complexity.arithmetic_intensity,
                achieved_gflops=complexity.train_flops / median / 1e9,
            )
            benchmarks.append(benchmark)
            write_output(args.env, benchmarks)

    print(f"wrote {BENCHMARK_CSV.format(env=args.env)}")


if __name__ == "__main__":
    main()
