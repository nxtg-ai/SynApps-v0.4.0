"""SynApps SDK — Python client for the SynApps workflow API."""

from synapps_sdk.client import SynApps
from synapps_sdk.async_client import AsyncSynApps
from synapps_sdk.exceptions import (
    SynAppsError,
    SynAppsAPIError,
    SynAppsConnectionError,
    SynAppsTimeoutError,
)

__all__ = [
    "SynApps",
    "AsyncSynApps",
    "SynAppsError",
    "SynAppsAPIError",
    "SynAppsConnectionError",
    "SynAppsTimeoutError",
]

__version__ = "0.1.0"
