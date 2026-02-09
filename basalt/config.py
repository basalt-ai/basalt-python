import os

from ._version import __version__

build = os.getenv("BASALT_BUILD", "production")

api_url = os.getenv("BASALT_API_URL")
if not api_url:
    api_url = "http://localhost:3001" if build == "development" else "https://api.getbasalt.ai"

otel_endpoint = os.getenv("BASALT_OTEL_ENDPOINT")
if not otel_endpoint:
    otel_endpoint = "http://127.0.0.1:4317" if build == "development" else "https://grpc.otel.getbasalt.ai"


config: dict[str, str] = {
    "api_url": api_url,
    "otel_endpoint": otel_endpoint,
    "sdk_version": __version__,
    "sdk_type": "python",
}
