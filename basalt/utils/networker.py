"""Legacy synchronous network helper removed."""

raise ImportError(
    "basalt.utils.networker was removed. Use basalt._internal.http.HTTPClient"
    " for HTTP interactions."
)
