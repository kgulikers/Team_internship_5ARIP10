import os
import toml

# Conveniences to other module directories via relative paths
AVULAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
"""Path to the extension source directory."""

AVULAB_ASSETS_DATA_DIR = os.path.join(AVULAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

AVULAB_ASSETS_METADATA = toml.load(os.path.join(AVULAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = AVULAB_ASSETS_METADATA["package"]["version"]

#from .origin_one import *
from .origin_one_robot_cfg import OriginRobotCfg #