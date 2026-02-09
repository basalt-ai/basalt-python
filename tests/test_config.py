import subprocess
import sys


def test_basalt_api_url_takes_precedence():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
os.environ["BASALT_API_URL"] = "https://custom.api.com"
os.environ.pop("BASALT_BUILD", None)
from basalt.config import config
print(config["api_url"])
        """,
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "https://custom.api.com"


def test_basalt_api_url_overrides_build_development():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
os.environ["BASALT_API_URL"] = "https://custom.api.com"
os.environ["BASALT_BUILD"] = "development"
from basalt.config import config
print(config["api_url"])
        """,
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "https://custom.api.com"


def test_fallback_to_build_development():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
os.environ.pop("BASALT_API_URL", None)
os.environ["BASALT_BUILD"] = "development"
from basalt.config import config
print(config["api_url"])
        """,
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "http://localhost:3001"


def test_fallback_to_production_default():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
os.environ.pop("BASALT_API_URL", None)
os.environ.pop("BASALT_BUILD", None)
from basalt.config import config
print(config["api_url"])
        """,
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "https://api.getbasalt.ai"


def test_basalt_otel_endpoint_override():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
os.environ["BASALT_OTEL_ENDPOINT"] = "http://custom.otel.com:4317"
os.environ.pop("BASALT_BUILD", None)
from basalt.config import config
print(config["otel_endpoint"])
        """,
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "http://custom.otel.com:4317"


def test_basalt_otel_endpoint_overrides_build_development():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
os.environ["BASALT_OTEL_ENDPOINT"] = "http://custom.otel.com:4317"
os.environ["BASALT_BUILD"] = "development"
from basalt.config import config
print(config["otel_endpoint"])
        """,
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "http://custom.otel.com:4317"
