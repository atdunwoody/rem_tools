# setup.py
# To install this package, activate your environmment and run:
# cd path/to/your/package
# where `path/to/your/package` is the directory containing this setup.py file.
# pip install -e .

# Then you can import any scripts from the package like this:
# from rem_tools.scripts import get_streams_and_thin


from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rem_tools",  # Use lowercase for package name
    version="0.1.0",
    author="Alex Thornton-Dunwoody",
    author_email="alex@lichenlandwater.com",
    description="Relative Elevation Model (REM) creation toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atdunwoody/rem_tools",
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        # List your project dependencies here, e.g.,
        # "numpy>=1.18.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
