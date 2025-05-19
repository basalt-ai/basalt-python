from .protocols import ILogger, LogLevel

class Logger(ILogger):
    def __init__(self, log_level: LogLevel = 'all'):
        self._log_level = log_level

    def warn(self, *args):
        if self._can_warn():
            print(*args)

    def info(self, *args):
        if self._can_info():
            print(*args)

    def error(self, *args):
        if self._can_error():
            print(*args)

    def _can_warn(self):
        return self._log_level == 'all' or self._log_level == 'warning'

    def _can_info(self):
        return self._log_level == 'all'

    def _can_error(self):
        return True
