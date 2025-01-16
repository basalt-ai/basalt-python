from setuptools import setup, find_packages

setup(
	name="basalt",
	version="0.1.0",
	description="Basalt SDK for python",
	author="Basalt",
	author_email="contact@getbasalt.io",
	url="http://github.com/...",
	packages=find_packages(),
	install_requires=[
		"requests>=2.32",
	],
	python_requires=">=3.6"
)
