#!/usr/bin/.env python
# -*- coding: utf-8 -*-

import io
import os

from pkg_resources import Requirement
from setuptools import find_packages, setup

# Package meta-data.
NAME = "pytorch-accelerated"
DESCRIPTION = ""
URL = "repo"
EMAIL = ""
AUTHOR = "Chris Hughes"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = 0.01

FILEPATH = os.path.abspath(os.path.dirname(__file__))
REQUIRED = []

with open("requirements.txt", "r") as f:
    for line in f.readlines():
        try:
            REQUIRED.append(str(Requirement.parse(line)))
        except ValueError:
            pass

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(FILEPATH, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*", "test"]
    ),
    scripts=[],
    install_requires=REQUIRED,
    include_package_data=True,
    classifiers=[
        "License :: Other/Proprietary License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)
