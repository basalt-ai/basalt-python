import os

from ._version import __version__

build = os.getenv("BASALT_BUILD", "production")


config: dict[str, str] = {
    'api_url': 'http://localhost:3001' if build == 'development' else 'https://api.getbasalt.ai',
    'otel_endpoint': 'http://127.0.0.1:4317' if build == 'development' else 'https://grpc.otel.getbasalt.ai',
    'sdk_version': __version__,
    'sdk_type': 'python',
}
