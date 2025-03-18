from .protocols import ILogger

class Logger(ILogger):
    def __init__(self, log_level: str = 'all'):
        self._log_level = log_level

    def warn(self, *args):
        if self._can_warn():
            print(*args)

    def debug(self, *args):
        if self._can_debug():
            print(*args)

    def _can_warn(self):
        return self._log_level in ['all', 'warning', 'debug']

    def _can_debug(self):
        return self._log_level in ['debug']
