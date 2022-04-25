#!/usr/bin/.env python
# -*- coding: utf-8 -*-

import io
import os

from pkg_resources import Requirement
from setuptools import find_packages, setup

import versioneer

# Package meta-data.
NAME = "pytorch-accelerated"
DESCRIPTION = (
    "A lightweight library designed to accelerate the process of training PyTorch models by "
    "providing a minimal, but extensible training loop which is flexible enough to handle the majority "
    "of use cases, and capable of utilizing different hardware options with no code changes required."
)
URL = "https://github.com/Chris-hughes10/pytorch-accelerated"
EMAIL = "31883449+Chris-hughes10@users.noreply.github.com"
AUTHOR = "Chris Hughes"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = versioneer.get_version()

FILEPATH = os.path.abspath(os.path.dirname(__file__))
REQUIRED = []
EXAMPLES_REQUIRED = []

with open("requirements.txt", "r") as f:
    for line in f.readlines():
        try:
            REQUIRED.append(str(Requirement.parse(line)))
        except ValueError:
            pass

with open("requirements.examples.txt", "r") as f:
    for line in f.readlines():
        try:
            EXAMPLES_REQUIRED.append(str(Requirement.parse(line)))
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
    cmdclass=versioneer.get_cmdclass(),
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
    extras_require={"examples": EXAMPLES_REQUIRED},
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)
