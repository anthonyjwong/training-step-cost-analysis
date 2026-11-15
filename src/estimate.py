import argparse
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torchvision
from torch.utils.flop_counter import FlopCounterMode

from utils import MODELS, getenv_list, write_csv

BYTES_PER_FLOAT = 4


@dataclass
class LayerCounts:
    forward_flops: int
    activation_bytes: int
    num_params: int


@dataclass
class StepComplexity:
    model_name: str
    batch_size: int
    num_params: int
    forward_flops: int
    train_flops: int
    param_bytes: int
    activation_bytes: int
    total_bytes: int
    arithmetic_intensity: float


def get_layer_counts(model_name: str) -> LayerCounts:
    IMAGE_SIZE = 224

    model = torchvision.models.get_model(model_name, weights=None).eval()
    counts = LayerCounts(forward_flops=0, activation_bytes=0, num_params=0)

    def measure_layer(
        layer: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ):
        input_size = inputs[0].numel()
        output_size = output.numel()

        if isinstance(layer, nn.Conv2d):
            kernel_height, kernel_width = layer.kernel_size
            # inputs_per_output = channels x kernel
            inputs_per_output = (
                (layer.in_channels // layer.groups) * kernel_height * kernel_width
            )

            # FLOPs = multiply + add (2) x outputs x inputs_per_output
            counts.forward_flops += 2 * output_size * inputs_per_output
        elif isinstance(layer, nn.Linear):
            # FLOPs = multiply + add (2) x outputs x inputs
            counts.forward_flops += 2 * output_size * layer.in_features

        # bytes = (forward (I + O) + backward (2I + O)) x 4
        forward_floats = input_size + output_size
        backward_floats = 2 * input_size + output_size
        counts.activation_bytes += (forward_floats + backward_floats) * BYTES_PER_FLOAT

    for layer in model.modules():
        is_leaf = len(list(layer.children())) == 0  # skip containers
        if is_leaf:
            layer.register_forward_hook(measure_layer)

    random_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        model(random_image)

    counts.num_params = sum(p.numel() for p in model.parameters())

    return counts


def get_step_complexity(model_name: str, batch_size: int) -> StepComplexity:
    counts = get_layer_counts(model_name)

    # step = per-image x batch_size
    forward_flops = counts.forward_flops * batch_size
    activation_bytes = counts.activation_bytes * batch_size

    # training step = forward + backward
    backward_flops = 2 * forward_flops
    train_flops = forward_flops + backward_flops

    # param traffic = forward + backward + SGD update
    forward_param_floats = counts.num_params  # read W
    backward_param_floats = 2 * counts.num_params  # read W, write dW
    update_param_floats = 5 * counts.num_params  # read W, dW, buf; write W, buf
    param_floats = forward_param_floats + backward_param_floats + update_param_floats
    param_bytes = param_floats * BYTES_PER_FLOAT
    total_bytes = param_bytes + activation_bytes

    return StepComplexity(
        model_name=model_name,
        batch_size=batch_size,
        num_params=counts.num_params,
        forward_flops=forward_flops,
        train_flops=train_flops,
        param_bytes=param_bytes,
        activation_bytes=activation_bytes,
        total_bytes=total_bytes,
        arithmetic_intensity=train_flops / total_bytes,
    )


def get_pytorch_flops(model_name: str) -> float:
    IMAGE_SIZE = 224
    BATCH = 2

    model: nn.Module = torchvision.models.get_model(model_name, weights=None).train()
    images = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)

    with FlopCounterMode(display=False) as counter:
        scores = cast(torch.Tensor, model(images))
        scores.sum().backward()

    return counter.get_total_flops() / BATCH


@dataclass
class FlopValidation:
    model_name: str
    estimated_gflop_per_image: float
    pytorch_gflop_per_image: float


@dataclass
class Args(argparse.Namespace):
    models: list[str]
    batch_sizes: list[int]
    out: str


def write_output(out: str, steps: list[StepComplexity], checks: list[FlopValidation]):
    ESTIMATION_HEADER = "{:14}{:>6}{:>11}{:>13}{:>9}{:>8}"
    ESTIMATION_ROW = "{:14}{:>6}{:>11.2f}{:>13.1f}{:>9.0f}{:>8.1f}"
    PYTORCH_COUNTER_HEADER = "{:14}{:>12}{:>12}"
    PYTORCH_COUNTER_ROW = "{:14}{:>12.2f}{:>12.2f}"
    VALIDATION_OUT = "out/estimate/flop_counter_validation.csv"

    write_csv(out, steps)
    write_csv(VALIDATION_OUT, checks)

    print(
        ESTIMATION_HEADER.format(
            "model", "batch", "params(M)", "train GFLOP", "MB", "AI"
        )
    )
    for step in steps:
        print(
            ESTIMATION_ROW.format(
                step.model_name,
                step.batch_size,
                step.num_params / 1e6,
                step.train_flops / 1e9,
                step.total_bytes / 1e6,
                step.arithmetic_intensity,
            )
        )

    print("\nTraining GFLOP per image")
    print(PYTORCH_COUNTER_HEADER.format("model", "hand count", "PyTorch"))
    for c in checks:
        print(
            PYTORCH_COUNTER_ROW.format(
                c.model_name, c.estimated_gflop_per_image, c.pytorch_gflop_per_image
            )
        )

    print(f"\nwrote {out}, {VALIDATION_OUT}")


def main():
    # default to env
    args = Args(
        models=getenv_list("MODELS", MODELS, str),
        batch_sizes=getenv_list("BATCH_SIZES", [8, 16, 32], int),
        out="out/estimate/arithmetic_intensity.csv",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--out")
    args = parser.parse_args(namespace=args)

    steps = [
        get_step_complexity(model_name, batch_size)
        for model_name in args.models
        for batch_size in args.batch_sizes
    ]
    checks = [
        FlopValidation(
            model_name,
            get_step_complexity(model_name, 1).train_flops / 1e9,
            get_pytorch_flops(model_name) / 1e9,
        )
        for model_name in args.models
    ]

    write_output(args.out, steps, checks)


if __name__ == "__main__":
    main()
