from .protocols import ILogger

class Logger(ILogger):
	def __init__(self, log_level: str = 'all'):
		self._log_level = log_level

	def warn(self, *args):
		if self._can_warn():
			print(*args)

	def _can_warn(self):
		return self._log_level == 'all' or self._log_level == 'warning'
