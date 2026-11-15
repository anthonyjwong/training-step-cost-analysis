import math
import re
import statistics
from dataclasses import dataclass
from typing import Self

from measure_.benchmark import Benchmark
from measure_.peaks import Roof
from utils import read_stage_csv, write_csv


@dataclass
class RoofSummary:
    env: str
    runs: int
    peak_gflops: float
    peak_gflops_cv: float
    peak_gbs: float
    peak_gbs_cv: float
    ridge: float


@dataclass
class BenchSummary:
    env: str
    model_name: str
    batch_size: int
    runs: int
    median_seconds: float
    median_seconds_cv: float
    mean_seconds: float
    mean_seconds_cv: float
    step_stdev_seconds: float
    step_cv: float
    images_per_second: float
    arithmetic_intensity: float
    achieved_gflops: float

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> Self:
        return cls(
            env=row["env"],
            model_name=row["model_name"],
            batch_size=int(row["batch_size"]),
            runs=int(row["runs"]),
            median_seconds=float(row["median_seconds"]),
            median_seconds_cv=float(row["median_seconds_cv"]),
            mean_seconds=float(row["mean_seconds"]),
            mean_seconds_cv=float(row["mean_seconds_cv"]),
            step_stdev_seconds=float(row["step_stdev_seconds"]),
            step_cv=float(row["step_cv"]),
            images_per_second=float(row["images_per_second"]),
            arithmetic_intensity=float(row["arithmetic_intensity"]),
            achieved_gflops=float(row["achieved_gflops"]),
        )


def get_base_env(env: str) -> str:
    return re.sub(r"-r\d+$", "", env)


def get_coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    return statistics.stdev(values) / statistics.mean(values)


def get_roof_summaries(roofs: list[Roof]) -> list[RoofSummary]:
    runs_by_env: dict[str, list[Roof]] = {}
    for roof in roofs:
        runs_by_env.setdefault(get_base_env(roof.env), []).append(roof)

    summaries: list[RoofSummary] = []
    for env, runs in runs_by_env.items():
        gflops = [r.peak_gflops for r in runs]
        gbs = [r.peak_gbs for r in runs]
        peak_gflops, peak_gbs = statistics.mean(gflops), statistics.mean(gbs)
        summaries.append(
            RoofSummary(
                env=env,
                runs=len(runs),
                peak_gflops=peak_gflops,
                peak_gflops_cv=get_coefficient_of_variation(gflops),
                peak_gbs=peak_gbs,
                peak_gbs_cv=get_coefficient_of_variation(gbs),
                ridge=peak_gflops / peak_gbs,
            )
        )

    return summaries


def get_bench_summaries(benchmarks: list[Benchmark]) -> list[BenchSummary]:
    if not benchmarks:
        raise ValueError(
            "no benchmarks: out/benchmark/bench_<env>.csv is missing (run python src/measure.py)"
        )
    runs_by_key: dict[tuple[str, str, int], list[Benchmark]] = {}
    for b in benchmarks:
        key = (get_base_env(b.env), b.model_name, b.batch_size)
        runs_by_key.setdefault(key, []).append(b)

    summaries: list[BenchSummary] = []
    for (env, model_name, batch_size), runs in runs_by_key.items():
        medians = [r.median_seconds for r in runs]
        median_seconds = statistics.mean(medians)
        means = [r.mean_seconds for r in runs]
        mean_seconds = statistics.mean(means)
        # the runs time the same number of steps, so pool their variances equally
        step_stdev_seconds = math.sqrt(
            statistics.mean(r.stdev_seconds**2 for r in runs)
        )
        # train FLOPs is constant
        train_flops = runs[0].achieved_gflops * runs[0].median_seconds
        summaries.append(
            BenchSummary(
                env=env,
                model_name=model_name,
                batch_size=batch_size,
                runs=len(runs),
                median_seconds=median_seconds,
                median_seconds_cv=get_coefficient_of_variation(medians),
                mean_seconds=mean_seconds,
                mean_seconds_cv=get_coefficient_of_variation(means),
                step_stdev_seconds=step_stdev_seconds,
                step_cv=step_stdev_seconds / mean_seconds,
                images_per_second=batch_size / median_seconds,
                arithmetic_intensity=runs[0].arithmetic_intensity,
                achieved_gflops=train_flops / median_seconds,
            )
        )

    return summaries


def write_output(
    roof_summaries: list[RoofSummary], bench_summaries: list[BenchSummary]
):
    write_csv("out/summarize/summary_peaks.csv", roof_summaries)
    write_csv("out/summarize/summary_bench.csv", bench_summaries)

    for r in roof_summaries:
        print(
            f"{r.env:12} runs {r.runs}  {r.peak_gflops:7.1f} GFLOP/s"
            + f" (cv {r.peak_gflops_cv:.1%})  {r.peak_gbs:6.1f} GB/s"
            + f" (cv {r.peak_gbs_cv:.1%})  ridge {r.ridge:5.2f}"
        )
    for b in bench_summaries:
        print(
            f"{b.env:12} {b.model_name:13} b{b.batch_size:<3} runs {b.runs}"
            + f"  {b.median_seconds * 1e3:8.1f} ms"
            + f" (runs cv {b.median_seconds_cv:.1%}, steps cv {b.step_cv:.1%})"
            + f"  {b.achieved_gflops:7.1f} GFLOP/s"
        )

    print("wrote out/summarize/summary_peaks.csv, summary_bench.csv")


def main():
    roofs = [
        Roof.from_csv_row(row)
        for row in read_stage_csv("out/peaks/peaks_*.csv", "measure")
    ]
    benchmarks = [
        Benchmark.from_csv_row(row)
        for row in read_stage_csv("out/benchmark/bench_*.csv", "measure")
    ]

    roof_summaries = get_roof_summaries(roofs)
    bench_summaries = get_bench_summaries(benchmarks)

    write_output(roof_summaries, bench_summaries)


if __name__ == "__main__":
    main()
