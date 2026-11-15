"""Generates every chart under out/charts/

Styled after the lecture 7 roofline slides.
"""

import math
import os
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from costs import GpuBuyOrRentCutoff
from summarize import BenchSummary, RoofSummary
from utils import read_stage_csv

# lecture 7 slide colors
BANDWIDTH_BLUE = "#4472C4"
PEAK_RED = "#FF0000"
UNATTAINABLE_GRAY = "#D9D9D9"
BANDWIDTH_FILL = "#DAE3F3"
COMPUTE_FILL = "#FBE5D6"
TEXT_COLOR = "#000000"

# one side-by-side chart per comparison in the report, all on the same axes
CHARTS = {
    "all": [
        "apple-cpu",
        "apple-cpu-vm",
        "gcp-cpu",
    ],  # the standardized CPU environments
    "gpu": ["apple-gpu", "gcp-gpu"],
    "scale": ["gcp-cpu", "irl-gcp-cpu"],
    "irl": [
        "irl-apple-cpu",
        "irl-apple-gpu",
        "irl-gcp-cpu",
        "irl-gcp-gpu",
    ],  # outside the study
}
ENV_TITLES = {
    "apple-cpu": "Bare metal: Apple M5 Pro",
    "apple-cpu-vm": "Local VM: VirtualBox (Apple M5 Pro)",
    "gcp-cpu": "Cloud VM: GCP n2-standard-4 (Intel Xeon)",
    "apple-gpu": "Bare metal GPU: Apple M5 Pro",
    "gcp-gpu": "Cloud VM: GCP g2-standard-4 (NVIDIA L4)",
    "irl-apple-cpu": "Mac CPU, all 15 cores",
    "irl-apple-gpu": "Mac GPU, 15 threads",
    "irl-gcp-cpu": "GCP n2-standard-16, all 16 vCPUs",
    "irl-gcp-gpu": "GCP g2-standard-16 (NVIDIA L4), 16 vCPUs",
}
MODEL_NAMES = {
    "resnet18": "ResNet-18",
    "mobilenet_v2": "MobileNetV2",
    "vgg16": "VGG-16",
}
MODEL_COLORS = {"resnet18": "#00B050", "mobilenet_v2": "#7030A0", "vgg16": "#262626"}
MODEL_MARKERS = {"resnet18": "o", "mobilenet_v2": "s", "vgg16": "^"}
BATCH_SIZES = {8: 25, 16: 50, 32: 85}  # batch size -> marker area

plt.rcParams["font.family"] = [
    "Arial",
    "DejaVu Sans",
]  # Calibri, the slides' font, is not installed


def get_tick_label(value: float) -> str:
    if value >= 1:
        return f"{value:,.0f}" if value >= 10 else f"{value:.3g}"
    return f"{value:g}"


def get_decades(low: float, high: float) -> list[float]:
    first, last = math.ceil(math.log10(low)), math.floor(math.log10(high))
    return [10.0**d for d in range(first, last + 1)]


def get_log_ticks(low: float, high: float) -> list[float]:
    # decades, or 1-2-5 steps when the range spans too few decades to read
    decades = get_decades(low, high)
    if len(decades) >= 3:
        return decades
    first, last = math.floor(math.log10(low)), math.ceil(math.log10(high))
    steps = [m * 10.0**d for d in range(first, last + 1) for m in (1, 2, 5)]
    return [t for t in steps if low <= t <= high]


def draw_arrow_axes(ax: Axes):
    # arrow axes, no box or grid
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.5)
    ax.plot(1, 0, ">", color=TEXT_COLOR, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^", color=TEXT_COLOR, transform=ax.transAxes, clip_on=False)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)


def draw_axes(ax: Axes, x_range: tuple[float, float], y_range: tuple[float, float]):
    # log-log as on the slides, but with ticks, since the numbers are real
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    draw_arrow_axes(ax)

    y_ticks = get_decades(*y_range)
    ax.set_yticks(y_ticks, [get_tick_label(y) for y in y_ticks])
    ax.minorticks_off()
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=11)


def draw_roofline(
    ax: Axes,
    roof: RoofSummary,
    benchmarks: list[BenchSummary],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> Text:
    """Draws one roofline; returns the bandwidth label, to be aligned after layout."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    ridge, peak, bandwidth = roof.ridge, roof.peak_gflops, roof.peak_gbs

    draw_axes(ax, x_range, y_range)

    # the five regions, simplified: unattainable above the roof, bound below it
    slope_start = max(x_min, y_min / bandwidth)
    ax.fill_between([x_min, x_max], y_max, y_min, color=UNATTAINABLE_GRAY, lw=0)
    ax.fill_between(
        [slope_start, ridge],
        [slope_start * bandwidth, peak],
        y_min,
        color=BANDWIDTH_FILL,
        lw=0,
    )
    ax.fill_between([ridge, x_max], peak, y_min, color=COMPUTE_FILL, lw=0)

    # attainable = min(AI x peak GB/s, peak GFLOP/s)
    ax.plot(
        [slope_start, ridge],
        [slope_start * bandwidth, peak],
        color=BANDWIDTH_BLUE,
        lw=2.5,
    )
    ax.plot([ridge, x_max], [peak, peak], color=PEAK_RED, lw=2.5)

    # dotted guides to the transition point, its value on the x axis
    ax.plot([x_min, ridge], [peak, peak], color=PEAK_RED, lw=1, ls=":")
    ax.plot([ridge, ridge], [y_min, peak], color=BANDWIDTH_BLUE, lw=1, ls=":")
    x_ticks = [d for d in get_decades(*x_range) if max(d / ridge, ridge / d) > 1.8]
    ax.set_xticks(
        [*x_ticks, ridge], [*(get_tick_label(x) for x in x_ticks), f"{ridge:.3g}"]
    )
    ax.get_xticklabels()[-1].set_color(BANDWIDTH_BLUE)

    # roof labels
    ax.text(
        ridge * 1.15,
        peak * 1.12,
        f"Peak {peak:,.0f} GFLOP/s",
        color=PEAK_RED,
        fontsize=11,
        va="bottom",
    )
    # the full label needs about 1.2 decades of slope; shorten it on a short slope
    is_short_slope = math.log10(ridge / slope_start) < 1.2
    bandwidth_text = f"{bandwidth:,.0f} GB/s"
    label_x = math.sqrt(slope_start * ridge)
    bandwidth_label = ax.text(
        label_x,
        label_x * bandwidth * 1.3,
        bandwidth_text if is_short_slope else f"Bandwidth {bandwidth_text}",
        color=BANDWIDTH_BLUE,
        fontsize=11,
        ha="center",
        va="bottom",
        rotation_mode="anchor",
    )
    ax.text(
        math.sqrt(ridge * x_max),
        math.sqrt(peak * y_max),
        "Unattainable",
        color="white",
        fontsize=11,
        ha="center",
        va="center",
    )
    ax.text(
        math.sqrt(ridge * x_max),
        y_min * 1.4,
        "Computation-Bound",
        color=TEXT_COLOR,
        fontsize=10,
        ha="center",
    )
    ax.text(
        ridge / 1.15,
        y_min * 1.4,
        "Bandwidth-Bound",
        color=TEXT_COLOR,
        fontsize=10,
        ha="right",
    )

    # one dot per model and batch size: achieved GFLOP/s at the model's AI
    for benchmark in benchmarks:
        model = benchmark.model_name
        ax.scatter(
            benchmark.arithmetic_intensity,
            benchmark.achieved_gflops,
            s=BATCH_SIZES.get(benchmark.batch_size, 50),
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_title(ENV_TITLES.get(roof.env, roof.env), fontsize=12, loc="left")
    return bandwidth_label


def align_to_slope(ax: Axes, label: Text):
    # the sloped roof rises one decade per decade, so its on-screen angle
    # depends only on how many pixels a decade takes on each axis
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    box = ax.get_window_extent()
    x_pixels_per_decade = box.width / math.log10(x_max / x_min)
    y_pixels_per_decade = box.height / math.log10(y_max / y_min)
    label.set_rotation(
        math.degrees(math.atan2(y_pixels_per_decade, x_pixels_per_decade))
    )


def add_legend(fig: Figure):
    model_handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=MODEL_MARKERS[m],
            color=MODEL_COLORS[m],
            ms=8,
            label=MODEL_NAMES[m],
        )
        for m in MODEL_COLORS
    ]
    batch_handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            color="#7F7F7F",
            ms=math.sqrt(area),
            label=f"batch {batch_size}",
        )
        for batch_size, area in BATCH_SIZES.items()
    ]
    # the legend fills column by column; interleave so models share the top row
    handles = [h for pair in zip(model_handles, batch_handles) for h in pair]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(model_handles),  # models on one row, batch sizes on the next
        frameon=False,
        fontsize=11,
        title="Dots: achieved GFLOP/s",
        title_fontsize=10,
    )


def get_ranges(
    roofs: list[RoofSummary], benchmarks: list[BenchSummary]
) -> tuple[tuple[float, float], tuple[float, float]]:
    ais = [r.arithmetic_intensity for r in benchmarks]
    ridges = [p.ridge for p in roofs]
    gflops = [r.achieved_gflops for r in benchmarks] + [p.peak_gflops for p in roofs]
    x_range = (min(ais + [1]) / 4, max(ais + ridges) * 4)
    # room above the highest roof for its Peak label and, above that, "Unattainable"
    y_range = (min(gflops) / 4, max(gflops) * 10)
    return x_range, y_range


def save_side_by_side(
    envs: list[str],
    roof_by_env: dict[str, RoofSummary],
    benchmarks: list[BenchSummary],
    ranges: tuple[tuple[float, float], tuple[float, float]],
    out: str,
):
    x_range, y_range = ranges
    fig = plt.figure(figsize=(6 * len(envs), 5.4), facecolor="white")
    fig.subplots(1, len(envs), sharey=True)
    row = fig.axes  # one Axes per environment, left to right
    labels = [
        draw_roofline(
            ax,
            roof_by_env[env],
            [r for r in benchmarks if r.env == env],
            x_range,
            y_range,
        )
        for ax, env in zip(row, envs)
    ]
    row[0].set_ylabel("Attainable GFLOP/s", fontsize=11)
    add_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    for ax, label in zip(row, labels):
        align_to_slope(ax, label)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def save_intensity(out: str):
    # each model's arithmetic intensity by batch size, against lecture 7's typical
    # transition point; from estimate.py, so no environment is involved
    rows = read_stage_csv("out/estimate/arithmetic_intensity.csv", "estimate")
    X_RANGE, Y_RANGE = (6.5, 40), (1, 300)
    TRANSITION = (5, 10)  # FLOP/byte

    fig, ax = plt.subplots(figsize=(7, 5.6), facecolor="white")
    draw_axes(ax, X_RANGE, Y_RANGE)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size", fontsize=11)
    ax.set_ylabel("Arithmetic Intensity (FLOP/byte)", fontsize=11)
    ax.set_xticks(list(BATCH_SIZES), [str(b) for b in BATCH_SIZES])
    ax.minorticks_off()

    ax.axhspan(*TRANSITION, color=UNATTAINABLE_GRAY, alpha=0.6, lw=0)
    ax.text(
        X_RANGE[0] * 1.03,
        TRANSITION[1] * 1.1,
        "Typical transition point, 5–10 FLOP/byte",
        fontsize=9,
        color=TEXT_COLOR,
        va="bottom",
    )
    ax.text(
        X_RANGE[0] * 1.03,
        2.6,
        "Bandwidth-bound below",
        fontsize=9,
        color=BANDWIDTH_BLUE,
    )
    ax.text(
        X_RANGE[0] * 1.03, 20, "Computation-bound above", fontsize=9, color=PEAK_RED
    )

    for model_name, name in MODEL_NAMES.items():
        points = sorted(
            (int(r["batch_size"]), float(r["arithmetic_intensity"]))
            for r in rows
            if r["model_name"] == model_name
        )
        batches, ais = zip(*points)
        color = MODEL_COLORS[model_name]
        ax.plot(batches, ais, color=color, lw=2)
        ax.scatter(
            batches,
            ais,
            s=[BATCH_SIZES[b] for b in batches],
            marker=MODEL_MARKERS[model_name],
            color=color,
            zorder=3,
        )
        # direct labels: the model at its last point, each value beside its point,
        # VGG-16's above and the others' below, clear of the line above them
        ax.text(
            batches[-1] * 1.08, ais[-1], name, color=color, fontsize=10, va="center"
        )
        is_top = model_name == "vgg16"
        for b, ai in points:
            ax.text(
                b,
                ai * 1.2 if is_top else ai / 1.2,
                f"{ai:.3g}",
                color=TEXT_COLOR,
                fontsize=8,
                ha="center",
                va="bottom" if is_top else "top",
            )

    ax.set_title("Arithmetic intensity per training step", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def draw_gpu_buy_or_rent_cutoff(cutoffs: list[GpuBuyOrRentCutoff], out: str):
    first = cutoffs[0]  # every row shares the bought GPU and the rented machine
    PRICE = first.bought_price
    X_RANGE, Y_RANGE = (10, 100_000), (10, 100_000)

    fig, ax = plt.subplots(figsize=(8, 5.6), facecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # buying: the card's price, whatever the use
    ax.plot(X_RANGE, [PRICE, PRICE], color=PEAK_RED, lw=2.5)
    ax.text(
        X_RANGE[0] * 1.2,
        PRICE * 1.15,
        f"Buy the {first.bought_gpu}: ${PRICE:,.0f}",
        color=PEAK_RED,
        va="bottom",
        fontsize=10,
    )

    # renting: one line per model, solid while it is the cheaper option
    for c in cutoffs:
        color = MODEL_COLORS[c.model_name]
        ax.plot(
            [X_RANGE[0], c.cutoff_epochs],
            [X_RANGE[0] * c.rent_per_epoch, PRICE],
            color=color,
            lw=2.5,
        )
        ax.plot(
            [c.cutoff_epochs, X_RANGE[1]],
            [PRICE, X_RANGE[1] * c.rent_per_epoch],
            color=color,
            lw=1,
            ls=":",
        )
        ax.plot(c.cutoff_epochs, PRICE, "o", color=color, markersize=8, mec="white")
        # a guide down to the axis, as at a roofline's transition point; MobileNetV2's
        # label goes left of its guide, since ResNet-18's is close on the right
        ax.plot(
            [c.cutoff_epochs, c.cutoff_epochs],
            [Y_RANGE[0], PRICE],
            color=color,
            lw=1,
            ls=":",
        )
        is_left = c.model_name == "mobilenet_v2"
        ax.text(
            c.cutoff_epochs / 1.1 if is_left else c.cutoff_epochs * 1.1,
            40 if c.model_name != "vgg16" else 150,
            f"{c.cutoff_epochs:,.0f} epochs\n≈ {c.cutoff_hours_per_day:.1f} h/day",
            color=color,
            ha="right" if is_left else "left",
            va="center",
            fontsize=9,
        )

    ticks = get_decades(*X_RANGE)
    ax.set_xticks(ticks, [f"{t:,.0f}" for t in ticks])
    ax.set_yticks(get_decades(*Y_RANGE), [f"${t:,.0f}" for t in get_decades(*Y_RANGE)])
    ax.minorticks_off()
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.set_xlabel(
        f"Epochs trained over the card's {first.bought_life_years:g}-year life",
        fontsize=11,
    )
    ax.set_ylabel("Total cost", fontsize=11)
    ax.set_title(
        f"{first.bought_gpu} vs a rented {first.rented_gpu}"
        + f" ({first.rented_instance_name}, on demand)",
        loc="left",
        fontsize=11,
    )
    handles = [
        plt.Line2D([], [], color=MODEL_COLORS[c.model_name], lw=2.5) for c in cutoffs
    ]
    labels = [f"Rent: {MODEL_NAMES[c.model_name]}" for c in cutoffs]
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9)
    fig.text(
        0.01,
        0.01,
        f"h/day: {first.bought_gpu} hours a day, estimated from the spec sheets"
        + " (not measured)",
        fontsize=8,
        color="#595959",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def save_gpu_buy_or_rent_cutoff(out: str):
    # cost over the card's life against epochs trained (costs.py): buying is flat at
    # its price, renting grows with every epoch, so the cheaper option is
    # min(rent, buy), a roof whose ridge is the cutoff point. The g2-standard-4 is the
    # chart, since it rents the same L4 for less and trained as fast
    rows = read_stage_csv("out/costs/gpu_buy_or_rent_cutoff.csv", "costs")
    cutoffs = [
        c
        for c in (GpuBuyOrRentCutoff.from_csv_row(row) for row in rows)
        if c.rented_instance_name == "g2-standard-4"
    ]
    if not cutoffs:
        raise ValueError(
            "no g2-standard-4 rows in out/costs/gpu_buy_or_rent_cutoff.csv"
        )
    order = list(MODEL_NAMES)
    draw_gpu_buy_or_rent_cutoff(
        sorted(cutoffs, key=lambda c: order.index(c.model_name)), out
    )


STUDY_ENVS = ["apple-cpu", "apple-cpu-vm", "gcp-cpu", "apple-gpu", "gcp-gpu"]
IRL_ENVS = CHARTS["irl"]
# the thread sweep's machines, as the report names them
SWEEP_NAMES = {"apple-cpu": "Mac", "apple-cpu-vm": "Local VM", "gcp-cpu": "GCP"}
SWEEP_STYLES = {"apple-cpu": "-", "apple-cpu-vm": "--", "gcp-cpu": ":"}
SWEEP_MARKERS = {"apple-cpu": "o", "apple-cpu-vm": "s", "gcp-cpu": "^"}
# each model's column beside its environment's tick
MODEL_OFFSETS = {"resnet18": -0.22, "mobilenet_v2": 0.0, "vgg16": 0.22}
NOTE_GRAY = "#595959"


def get_log_range(values: list[float], margin: float = 2.5) -> tuple[float, float]:
    return min(values) / margin, max(values) * margin


def set_log_axis(ax: Axes, axis: str, value_range: tuple[float, float], unit: str = ""):
    ticks = get_log_ticks(*value_range)
    labels = [f"{unit}{get_tick_label(t)}" for t in ticks]
    if axis == "y":
        ax.set_yscale("log")
        ax.set_ylim(*value_range)
        ax.set_yticks(ticks, labels)
    else:
        ax.set_xscale("log")
        ax.set_xlim(*value_range)
        ax.set_xticks(ticks, labels)
    ax.minorticks_off()


def save_figure(fig: Figure, out: str):
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def get_model_handles(marker_size: float = 8) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            ls="",
            marker=MODEL_MARKERS[m],
            color=MODEL_COLORS[m],
            ms=marker_size,
            label=MODEL_NAMES[m],
        )
        for m in MODEL_COLORS
    ]


def get_spot_handles() -> list[Line2D]:
    return [
        Line2D([], [], ls="", marker="o", color="#7F7F7F", ms=8, label="on demand"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            mfc="white",
            mec="#7F7F7F",
            mew=1.5,
            ms=8,
            label="spot",
        ),
    ]


def draw_env_columns(ax: Axes, envs: list[str]):
    # environments along the x axis, each with one column per model
    ax.set_xlim(-0.6, len(envs) - 0.4)
    ax.set_xticks(range(len(envs)), envs, fontsize=9)
    draw_arrow_axes(ax)
    # faint separators between environments, so each one's dots read as a group
    for i in range(len(envs) - 1):
        ax.axvline(i + 0.5, color=UNATTAINABLE_GRAY, lw=0.8, zorder=0)


def save_model_dots(
    benchmarks: list[BenchSummary],
    envs: list[str],
    get_value: Callable[[BenchSummary], float],
    y_label: str,
    title: str,
    out: str,
):
    """One dot per environment, model and batch size, styled as the roofline dots."""
    shown = [b for b in benchmarks if b.env in envs]
    y_range = get_log_range([get_value(b) for b in shown])

    fig, ax = plt.subplots(figsize=(1.6 * len(envs) + 1.5, 5.6), facecolor="white")
    draw_env_columns(ax, envs)
    set_log_axis(ax, "y", y_range)
    for b in shown:
        ax.scatter(
            envs.index(b.env) + MODEL_OFFSETS[b.model_name],
            get_value(b),
            s=BATCH_SIZES.get(b.batch_size, 50),
            color=MODEL_COLORS[b.model_name],
            marker=MODEL_MARKERS[b.model_name],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, loc="left", fontsize=11)
    add_legend(fig)
    fig.legends[0].set_title(None)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, out)


def get_sweep_roofs(
    roofs: list[RoofSummary],
) -> dict[str, list[tuple[int, RoofSummary]]]:
    # the thread-sweep runs are named <env>-t<threads>
    sweeps: dict[str, list[tuple[int, RoofSummary]]] = {}
    for roof in roofs:
        base, _, threads = roof.env.rpartition("-t")
        if base in SWEEP_NAMES and threads.isdigit():
            sweeps.setdefault(base, []).append((int(threads), roof))
    return {env: sorted(points) for env, points in sweeps.items()}


def save_thread_sweep(roofs: list[RoofSummary], out: str):
    # E1: each machine's two ceilings against threads; compute in the roof's red,
    # bandwidth in its blue, one panel each since their units differ
    sweeps = get_sweep_roofs(roofs)
    if not sweeps:
        raise ValueError("no thread-sweep runs (<env>-t<threads>) in summary_peaks.csv")
    panels = [
        ("Peak GFLOP/s", PEAK_RED, lambda r: r.peak_gflops),
        ("Peak GB/s", BANDWIDTH_BLUE, lambda r: r.peak_gbs),
    ]
    all_threads = sorted({t for points in sweeps.values() for t, _ in points})

    fig, row = plt.subplots(1, 2, figsize=(11, 5.2), facecolor="white")
    for ax, (label, color, get_value) in zip(row, panels):
        values = [get_value(r) for points in sweeps.values() for _, r in points]
        draw_arrow_axes(ax)
        set_log_axis(ax, "y", get_log_range(values, 2))
        ax.set_xscale("log", base=2)
        ax.set_xlim(all_threads[0] / 1.4, all_threads[-1] * 1.4)
        ax.set_xticks(all_threads, [str(t) for t in all_threads])
        ax.minorticks_off()
        ax.set_xlabel("Threads", fontsize=11)
        ax.set_title(label, loc="left", fontsize=11, color=color)
        for env, points in sweeps.items():
            threads = [t for t, _ in points]
            ys = [get_value(r) for _, r in points]
            ax.plot(
                threads,
                ys,
                color=color,
                lw=2,
                ls=SWEEP_STYLES[env],
                marker=SWEEP_MARKERS[env],
                ms=7,
                mec="white",
            )
            for t, y in zip(threads, ys):
                # below the point when another machine's value sits just above it
                is_below = any(
                    y < get_value(r) < y * 1.5
                    for other, points in sweeps.items()
                    if other != env
                    for other_t, r in points
                    if other_t == t
                )
                ax.text(
                    t,
                    y / 1.12 if is_below else y * 1.12,
                    f"{y:,.0f}",
                    fontsize=8,
                    ha="center",
                    va="top" if is_below else "bottom",
                )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#7F7F7F",
                lw=2,
                ls=SWEEP_STYLES[env],
                marker=SWEEP_MARKERS[env],
                label=name,
            )
            for env, name in SWEEP_NAMES.items()
            if env in sweeps
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(sweeps),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Thread sweep (E1): where each CPU's ceilings come from",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, out)


def save_ceilings(roofs: list[RoofSummary], envs: list[str], title: str, out: str):
    # each environment's two measured ceilings as columns, one panel per unit;
    # a linear axis from zero, since a column's height is its value
    by_env = {r.env: r for r in roofs}
    envs = [e for e in envs if e in by_env]
    if not envs:
        raise ValueError(f"no ceilings for {out} in summary_peaks.csv")
    # GB/s keeps a decimal: gcp-cpu's 26.5 would otherwise print as 26
    panels = [
        ("Peak GFLOP/s", PEAK_RED, lambda r: r.peak_gflops, ",.0f"),
        ("Peak GB/s", BANDWIDTH_BLUE, lambda r: r.peak_gbs, ",.1f"),
    ]
    xs = list(range(len(envs)))

    fig, row = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="white")
    for ax, (label, color, get_value, value_format) in zip(row, panels):
        values = [get_value(by_env[e]) for e in envs]
        draw_arrow_axes(ax)
        ax.bar(xs, values, width=0.55, color=color, zorder=2)
        ax.set_ylim(0, max(values) * 1.12)  # room for the value on the tallest
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.set_xticks(xs, envs, fontsize=9, rotation=20, ha="right")
        ax.set_title(label, loc="left", fontsize=11)
        for x, value in zip(xs, values):
            ax.text(
                x,
                value,
                f"{value:{value_format}}",
                fontsize=8,
                ha="center",
                va="bottom",
            )
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out)


def get_epoch_costs() -> list[dict[str, str]]:
    return read_stage_csv("out/costs/costs.csv", "costs")


def save_epoch_costs(envs: list[str], title: str, out: str):
    # one epoch at each environment's fastest batch size: hours, then dollars,
    # with spot as a hollow dot where it differs from on demand
    rows = [r for r in get_epoch_costs() if r["env"] in envs]
    panels = [
        ("Hours per epoch", "", [float(r["hours_per_epoch"]) for r in rows]),
        ("Cost per epoch", "$", [float(r["cost"]) for r in rows]),
    ]
    spot_costs = [float(r["spot_cost"]) for r in rows]

    fig, row = plt.subplots(1, 2, figsize=(13, 5.6), facecolor="white")
    for ax, (label, unit, values) in zip(row, panels):
        is_cost = unit == "$"
        draw_env_columns(ax, envs)
        shown_values = values + spot_costs if is_cost else values
        set_log_axis(ax, "y", get_log_range(shown_values), unit)
        ax.set_title(label, loc="left", fontsize=11)
        for r, value, spot in zip(rows, values, spot_costs):
            model = r["model_name"]
            x = envs.index(r["env"]) + MODEL_OFFSETS[model]
            color = MODEL_COLORS[model]
            ax.scatter(
                x,
                value,
                s=55,
                color=color,
                marker=MODEL_MARKERS[model],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            text = f"{unit}{value:.2f}" if value < 10 else f"{unit}{value:,.0f}"
            # ResNet-18's column is leftmost, so its labels go left, clear of the others
            is_left = model == "resnet18"
            ax.text(
                x - 0.05 if is_left else x + 0.05,
                value,
                text,
                fontsize=7,
                va="center",
                ha="right" if is_left else "left",
            )
            if is_cost and not math.isclose(spot, value):
                ax.scatter(
                    x,
                    spot,
                    s=55,
                    facecolor="white",
                    edgecolor=color,
                    marker=MODEL_MARKERS[model],
                    linewidth=1.5,
                    zorder=3,
                )
    fig.legend(
        handles=get_model_handles() + get_spot_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, out)


def save_hourly_prices(out: str):
    # what an hour of each machine costs; the Mac's is its price spread over its life
    prices: dict[str, tuple[float, float]] = {}
    for r in get_epoch_costs():
        prices[r["machine"]] = (
            float(r["price_per_hour"]),
            float(r["spot_price_per_hour"]),
        )
    machines = sorted(prices, key=lambda m: prices[m][0])
    ys = list(range(len(machines)))

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(machines) + 1.8), facecolor="white")
    draw_arrow_axes(ax)
    x_max = max(p for p, _ in prices.values()) * 1.3
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.6, len(machines) - 0.4)
    ax.set_yticks(ys, machines, fontsize=9)
    ax.xaxis.set_major_formatter(lambda x, _: f"${x:.2f}")
    for y, machine in zip(ys, machines):
        on_demand, spot = prices[machine]
        ax.hlines(y, 0, on_demand, color=TEXT_COLOR, lw=1, alpha=0.3)
        ax.scatter(on_demand, y, s=55, color="#7F7F7F", zorder=3)
        ax.text(
            on_demand + x_max * 0.02, y, f"${on_demand:.4f}", fontsize=8, va="center"
        )
        if not math.isclose(spot, on_demand):
            ax.scatter(
                spot,
                y,
                s=55,
                facecolor="white",
                edgecolor="#7F7F7F",
                linewidth=1.5,
                zorder=3,
            )
            ax.text(
                spot,
                y + 0.22,
                f"${spot:.4f}",
                fontsize=8,
                ha="center",
                color=NOTE_GRAY,
            )
    ax.set_xlabel("Price per hour", fontsize=11)
    ax.legend(handles=get_spot_handles(), loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Hourly price of each machine", loc="left", fontsize=11)
    fig.tight_layout()
    save_figure(fig, out)


def save_price_per_hour_saved(local: str, cloud: str, out: str):
    # renting against hardware I own buys time; this is what each hour saved costs
    rows = [
        r
        for r in read_stage_csv("out/costs/local_vs_cloud.csv", "costs")
        if r["local"] == local and r["cloud"] == cloud and r["price_per_hour_saved"]
    ]
    if not rows:
        raise ValueError(f"no {local} vs {cloud} rows with time saved")
    models = [m for m in MODEL_NAMES if any(r["model_name"] == m for r in rows)]
    by_model = {r["model_name"]: r for r in rows}
    ys = list(range(len(models)))[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(models) + 2), facecolor="white")
    draw_arrow_axes(ax)
    x_max = max(float(r["price_per_hour_saved"]) for r in rows) * 1.25
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_yticks(ys, [MODEL_NAMES[m] for m in models], fontsize=10)
    ax.xaxis.set_major_formatter(lambda x, _: f"${x:.2f}")
    for y, model in zip(ys, models):
        on_demand = float(by_model[model]["price_per_hour_saved"])
        spot = float(by_model[model]["spot_price_per_hour_saved"])
        color = MODEL_COLORS[model]
        ax.hlines(y, 0, on_demand, color=color, lw=1, alpha=0.4)
        ax.scatter(
            on_demand, y, s=70, color=color, marker=MODEL_MARKERS[model], zorder=3
        )
        ax.scatter(
            spot,
            y,
            s=70,
            facecolor="white",
            edgecolor=color,
            marker=MODEL_MARKERS[model],
            linewidth=1.5,
            zorder=3,
        )
        ax.text(
            on_demand + x_max * 0.02, y, f"${on_demand:.2f}", fontsize=9, va="center"
        )
        ax.text(
            spot, y + 0.22, f"${spot:.2f}", fontsize=8, ha="center", color=NOTE_GRAY
        )
    ax.set_xlabel("Price of one hour saved per epoch", fontsize=11)
    ax.legend(handles=get_spot_handles(), loc="lower right", frameon=False, fontsize=9)
    ax.set_title(
        f"Renting {cloud} instead of training on {local}", loc="left", fontsize=11
    )
    fig.tight_layout()
    save_figure(fig, out)


def main():
    os.makedirs("out/charts", exist_ok=True)
    save_intensity("out/charts/arithmetic_intensity.png")
    save_gpu_buy_or_rent_cutoff("out/charts/gpu_buy_or_rent_cutoff.png")
    all_benchmarks = [
        BenchSummary.from_csv_row(row)
        for row in read_stage_csv("out/summarize/summary_bench.csv", "summarize")
    ]
    measured = {r.env for r in all_benchmarks}

    # every charted environment, repeats already averaged; other runs (thread
    # tests) are left out
    charted = {env for envs in CHARTS.values() for env in envs}
    roof_by_env = {
        roof.env: roof
        for roof in (
            RoofSummary.from_csv_row(row)
            for row in read_stage_csv("out/summarize/summary_peaks.csv", "summarize")
        )
        if roof.env in charted and roof.env in measured
    }
    if not roof_by_env:
        raise ValueError("no charted environment has both a roof and benchmarks")
    benchmarks = [r for r in all_benchmarks if r.env in roof_by_env]

    # the same axes on every chart so they can be compared
    ranges = get_ranges(list(roof_by_env.values()), benchmarks)
    x_range, y_range = ranges

    for env, roof in roof_by_env.items():
        env_benchmarks = [r for r in benchmarks if r.env == env]

        fig, ax = plt.subplots(figsize=(7, 5.6), facecolor="white")
        label = draw_roofline(ax, roof, env_benchmarks, x_range, y_range)
        ax.set_ylabel("Attainable GFLOP/s", fontsize=11)
        add_legend(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.83))
        align_to_slope(ax, label)
        fig.savefig(f"out/charts/roofline_{env}.png", dpi=200)
        plt.close(fig)
        print(f"wrote out/charts/roofline_{env}.png")

    for chart, envs in CHARTS.items():
        present = [env for env in envs if env in roof_by_env]
        if present:
            out = f"out/charts/roofline_{chart}.png"
            save_side_by_side(present, roof_by_env, benchmarks, ranges, out)

    # the report's tables, as charts
    all_roofs = [
        RoofSummary.from_csv_row(row)
        for row in read_stage_csv("out/summarize/summary_peaks.csv", "summarize")
    ]
    save_ceilings(
        all_roofs,
        STUDY_ENVS,
        "Measured ceilings of the study environments",
        "out/charts/ceilings.png",
    )
    save_ceilings(
        all_roofs,
        IRL_ENVS,
        "Measured ceilings of the real-world machines",
        "out/charts/ceilings_irl.png",
    )
    save_thread_sweep(all_roofs, "out/charts/thread_sweep.png")
    for name, envs in (("", STUDY_ENVS), ("_irl", IRL_ENVS)):
        save_model_dots(
            all_benchmarks,
            envs,
            lambda b: b.median_seconds * 1000,
            "Step time (ms)",
            "Median step time",
            f"out/charts/step_times{name}.png",
        )
    save_hourly_prices("out/charts/hourly_prices.png")
    save_epoch_costs(
        STUDY_ENVS,
        "One epoch of ImageNet at each environment's fastest batch size",
        "out/charts/epoch_costs.png",
    )
    save_epoch_costs(
        IRL_ENVS,
        "One epoch of ImageNet, real-world machines",
        "out/charts/epoch_costs_irl.png",
    )
    save_price_per_hour_saved(
        "apple-gpu", "gcp-gpu", "out/charts/price_per_hour_saved.png"
    )


if __name__ == "__main__":
    main()
