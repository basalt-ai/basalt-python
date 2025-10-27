"""Custom exceptions for the Basalt SDK."""


class BasaltError(Exception):
    """Base exception for all Basalt SDK errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class BasaltAPIError(BasaltError):
    """Base exception for API-related errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class BadRequestError(BasaltAPIError):
    """Raised when the API returns a 400 Bad Request error."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class UnauthorizedError(BasaltAPIError):
    """Raised when the API returns a 401 Unauthorized error."""

    def __init__(self, message: str):
        super().__init__(message, status_code=401)


class ForbiddenError(BasaltAPIError):
    """Raised when the API returns a 403 Forbidden error."""

    def __init__(self, message: str):
        super().__init__(message, status_code=403)


class NotFoundError(BasaltAPIError):
    """Raised when the API returns a 404 Not Found error."""

    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class NetworkError(BasaltError):
    """Raised when a network error occurs."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
