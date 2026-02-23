"""SDK exception hierarchy."""


class SynAppsError(Exception):
    """Base exception for all SDK errors."""


class SynAppsAPIError(SynAppsError):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str, response_body: dict | None = None):
        self.status_code = status_code
        self.detail = detail
        self.response_body = response_body or {}
        super().__init__(f"HTTP {status_code}: {detail}")


class SynAppsConnectionError(SynAppsError):
    """Raised when the client cannot connect to the server."""


class SynAppsTimeoutError(SynAppsError):
    """Raised when a request or polling operation times out."""
