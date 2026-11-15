"""Types for the JSON that Google's Cloud Billing Catalog API returns.

API: https://docs.cloud.google.com/billing/docs/reference/rest/v1/services.skus/list
"""

from typing import TypedDict

COMPUTE_ENGINE = "6F81-5844-456A"  # the catalog's Compute Engine service


# only the parts of each response that prices.py reads
class CatalogUnitPrice(TypedDict, total=False):
    units: str  # whole dollars, as a string
    nanos: int  # billionths of a dollar


class CatalogTieredRate(TypedDict):
    unitPrice: CatalogUnitPrice


class CatalogPricingExpression(TypedDict):
    usageUnitDescription: str
    tieredRates: list[CatalogTieredRate]


class CatalogPricingInfo(TypedDict):
    effectiveTime: str
    pricingExpression: CatalogPricingExpression


class CatalogSku(TypedDict):
    skuId: str
    description: str
    pricingInfo: list[CatalogPricingInfo]


class CatalogPage(TypedDict, total=False):
    skus: list[CatalogSku]
    nextPageToken: str
