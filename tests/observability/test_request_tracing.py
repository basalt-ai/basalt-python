import importlib
from collections.abc import Iterator

from basalt.observability.spans import BasaltRequestSpan


class DummySpan:
    def __init__(self) -> None:
        self.variables = None
        self.attributes = {}
        self.exceptions = []
        self.status = None
        self._io_payload = {"input": None, "output": None, "variables": None}

    def set_io(self, *, input_payload=None, output_payload=None, variables=None):
        if variables is not None:
            self.variables = variables
            self._io_payload["variables"] = variables

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status


class DummyObserve:
    def __init__(self) -> None:
        self.inputs = []
        self.outputs = []
        self.entered = []

    def __call__(self, *, name, metadata) -> "_DummyContext":
        self.entered.append({"name": name, "metadata": metadata})
        return _DummyContext(self)

    def input(self, payload):
        self.inputs.append(payload)

    def output(self, payload):
        self.outputs.append(payload)


class _DummyContext:
    def __init__(self, observe) -> None:
        self.observe = observe
        self.span = DummySpan()

    def __enter__(self) -> DummySpan:
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


class RecordingSpan(BasaltRequestSpan):
    __slots__ = ("finalize_calls",)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.finalize_calls: list[dict] = []

    def finalize(self, span, *, duration_s, status_code, error=None):
        self.finalize_calls.append(
            {"span": span, "duration_s": duration_s, "status_code": status_code, "error": error}
        )
        return super().finalize(span, duration_s=duration_s, status_code=status_code, error=error)


def _patch_perf_counter(monkeypatch, values: list[float]) -> None:
    iterator: Iterator[float] = iter(values)
    module = importlib.import_module("basalt.observability.request_tracing")
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(iterator))
