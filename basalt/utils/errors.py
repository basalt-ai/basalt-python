class FetchError(Exception):
    def __init__(self, message: str):
        self.message = message

class BadRequest(FetchError):
    pass

class Unauthorized(FetchError):
    pass

class Forbidden(FetchError):
    pass

class NotFound(FetchError):
    pass

class NetworkBaseError(FetchError):
    pass
