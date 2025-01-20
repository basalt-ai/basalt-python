from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

def get_version():
    version_file = "basalt/_version.py"
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

setup(
	name="basalt_sdk",
	version=get_version(),
	description="Basalt SDK for python",
	long_description=long_description,
    long_description_content_type='text/markdown',
	license="MIT",
	keywords="basalt, ai, sdk, python",
	author="Basalt",
	author_email="support@getbasalt.ai",
	url="https://github.com/basalt-ai/basalt-python",
	packages=find_packages(),
	install_requires=[
		"requests>=2.32",
	],
	python_requires=">=3.6"
)
