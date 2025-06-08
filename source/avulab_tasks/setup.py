# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the Avular Origin One project.
#
# Based on code from the WheeledLab project (https://github.com/NVlabs/wheeledlab)
# Copyright (c) 2025–2027, The Wheeled Lab Project Developers.
#
# Modified by the Avulab Project Developers (5ARIP10), 2025–2027.
# Modifications include: New Path Names


"""Installation script for the 'avulab' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
]

# Installation operation
setup(
    name="avulab_tasks",
    packages=["avulab_tasks"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
