import unittest
from parameterized import parameterized
from utils import replace_variables, pick_typed

class TestUtils(unittest.TestCase):

	@parameterized.expand([
		# Replaces a variable
		('Hello {{name}}', { "name": 'Basalt' }, (set([]), 'Hello Basalt')),

		# Replaces a multiple occurences of a variable
		('Hello {{name}} {{name}}', { "name": 'Basalt' }, (set([]), 'Hello Basalt Basalt')),
		
		# Can replace multiple variables
		('{{a}} + {{b}} = {{c}}', { "a": 1, "b": 2, "c": 3 }, (set([]), '1 + 2 = 3')),
		
		# Can replace multiple variables
		('{{a}} + {{b}} = {{c {{c}} }}', { "a": 1, "b": 2, "c": 3 }, (set(["c {{c"]), '1 + 2 = {{c {{c}} }}')),
		
		# Doesn't replace or empty missing variables
		('Hello {{missing}}', {}, (set(["missing"]), 'Hello {{missing}}'))
	])
	def test_replace_variables(self, str_value, vars, expected):
		self.assertEqual(replace_variables(str_value, vars), expected)


	@parameterized.expand([
		({ "a": 1 }, int, True),
		({ "a": 1 }, float, False),
		({ "a": 2. }, float, True),
		({ "a": "1" }, str, True),
		({ "a": 1 }, bool, False),
		({ "a": 0 }, bool, False),
		({ "a": True, }, bool, True),
		({ "a": False }, bool, True),
	])
	def test_pick_typed(self, d, expected_type, succeeds):
		if succeeds:
			pick_typed(d, "a", expected_type)			
		else:
			with self.assertRaises(Exception):
				pick_typed(d, "a", expected_type)