import datetime
import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from enum import Enum
from http.client import HTTPResponse
from typing import cast

from prices_.gcp import COMPUTE_ENGINE, CatalogPage, CatalogSku
from utils import write_csv

PRICES_CSV = "out/prices/prices.csv"


class GcpSku(Enum):
    # Google SKU IDs for us-central1
    N2_CORE = "BB77-5FDA-69D9"
    N2_RAM = "5B01-D157-A097"
    N2_SPOT_CORE = "2856-91B1-406A"
    N2_SPOT_RAM = "6276-ABCE-E764"
    G2_CORE = "D18E-3563-415F"
    G2_RAM = "F0A4-E3D3-33BD"
    G2_SPOT_CORE = "59A6-9C49-BDDB"
    G2_SPOT_RAM = "BF80-650A-78E6"
    L4 = "A88A-5A60-E821"
    L4_SPOT = "AEC2-3D5C-61BF"


@dataclass
class SkuPrice:
    name: str
    sku: str
    description: str
    usd: float
    unit: str  # per hour, or per gibibyte hour
    effective: str  # since when this price started
    queried: str


def get_gcp_skus(token: str) -> list[CatalogSku]:
    """GCP catalog API does not support filtering, so search the entire catalog."""
    skus: list[CatalogSku] = []
    page = ""
    while True:
        url = (
            f"https://cloudbilling.googleapis.com/v1/services/{COMPUTE_ENGINE}/skus"
            f"?pageSize=5000&pageToken={page}"
        )
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        with cast(HTTPResponse, urllib.request.urlopen(request)) as response:
            body = cast(CatalogPage, json.load(response))
        skus += body.get("skus", [])
        page = body.get("nextPageToken", "")
        if not page:
            return skus


def get_gcp_prices(sku: CatalogSku) -> tuple[float, str]:
    # the last tier is the rate for ordinary use; each price is whole units + nanos
    expression = sku["pricingInfo"][0]["pricingExpression"]
    rate = expression["tieredRates"][-1]["unitPrice"]
    usd = int(rate.get("units", "0")) + rate.get("nanos", 0) / 1e9
    return usd, expression["usageUnitDescription"]


def get_cloud_rates(rows: list[dict[str, str]]) -> dict[GcpSku, float]:
    by_sku_name = {row["name"]: float(row["usd"]) for row in rows}
    missing = [
        sku.name.lower() for sku in GcpSku if sku.name.lower() not in by_sku_name
    ]
    if missing:
        raise ValueError(f"{PRICES_CSV} has no price for {', '.join(missing)}")
    return {sku: by_sku_name[sku.name.lower()] for sku in GcpSku}


def write_output(prices: list[SkuPrice]):
    write_csv(PRICES_CSV, prices)

    for p in prices:
        print(f"{p.name:13} {p.usd:12.9f} per {p.unit:15} {p.description}")
    print(f"wrote {PRICES_CSV}")


def main():
    if os.path.exists(PRICES_CSV):
        raise FileExistsError(
            f"{PRICES_CSV} already exists, delete it manually first if you would like to refresh it"
        )

    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    by_id = {s["skuId"]: s for s in get_gcp_skus(token)}
    missing = [f"{sku.name} ({sku.value})" for sku in GcpSku if sku.value not in by_id]
    if missing:
        raise ValueError(f"not in the catalog any more: {', '.join(missing)}")

    queried = datetime.date.today().isoformat()
    prices: list[SkuPrice] = []
    for gcp_sku in GcpSku:
        sku = by_id[gcp_sku.value]
        usd, unit = get_gcp_prices(sku)
        effective = sku["pricingInfo"][0]["effectiveTime"][:10]
        prices.append(
            SkuPrice(
                gcp_sku.name.lower(),
                gcp_sku.value,
                sku["description"],
                usd,
                unit,
                effective,
                queried,
            )
        )

    write_output(prices)


if __name__ == "__main__":
    main()
