import os
import toml

# Conveniences to other module directories via relative paths
avulab_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
"""Path to the extension source directory."""

avulab_ASSETS_DATA_DIR = os.path.join(avulab_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

avulab_ASSETS_METADATA = toml.load(os.path.join(avulab_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = avulab_ASSETS_METADATA["package"]["version"]

from .mushr import *
from .origin_one_robot_cfg import OriginRobotCfg #