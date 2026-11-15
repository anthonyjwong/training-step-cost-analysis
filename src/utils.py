import csv
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    # exists only for type checkers; importing it at runtime fails
    from _typeshed import DataclassInstance

MODELS = ["resnet18", "mobilenet_v2", "vgg16"]


def getenv_list[T](name: str, default: list[T], cast: Callable[[str], T]) -> list[T]:
    value = os.environ.get(name)
    return default if not value else [cast(v) for v in value.split()]


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
