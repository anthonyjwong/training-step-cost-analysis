import csv
import os
import platform
import random
import socket
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Self, cast

import torch

if TYPE_CHECKING:
    # exists only for type checkers; importing it at runtime fails
    from _typeshed import DataclassInstance

MODELS = ["resnet18", "mobilenet_v2", "vgg16"]


@dataclass
class MachineInfo:
    host: str
    device: str
    device_name: str
    torch_version: str
    threads: int
    cpu_count: int

    @classmethod
    def from_device(cls, device: torch.device) -> Self:
        if device.type == "mps":
            device_name = "Apple GPU (MPS)"
        elif device.type == "cuda":
            device_name = torch.cuda.get_device_name(device)
        else:
            device_name = platform.processor() or platform.machine()

        return cls(
            host=socket.gethostname(),
            device=device.type,
            device_name=device_name,
            torch_version=torch.__version__,
            threads=torch.get_num_threads(),
            cpu_count=os.cpu_count() or 0,
        )


def getenv[T](name: str, default: T, cast: Callable[[str], T]) -> T:
    value = os.environ.get(name)
    return default if not value else cast(value)


def getenv_list[T](name: str, default: list[T], cast: Callable[[str], T]) -> list[T]:
    value = os.environ.get(name)
    return default if not value else [cast(v) for v in value.split()]


def get_seed() -> int:
    return getenv("SEED", random.randrange(2**31), int)


def wait_for_device(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def to_csv_row(record: "DataclassInstance") -> dict[str, object]:
    fields: dict[str, object] = asdict(record)
    row: dict[str, object] = {}
    for key, value in fields.items():
        if isinstance(value, dict):
            row.update(cast(dict[str, object], value))
        else:
            row[key] = value

    return row


def write_csv(path: str, records: Sequence["DataclassInstance"]):
    rows = [to_csv_row(r) for r in records]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
