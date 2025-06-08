from .rss_cfgs import *

from wheeledlab_rl.utils.hydra import register_run_to_hydra

register_run_to_hydra("RSS_NAV_CONFIG", RSS_NAV_CONFIG)