# =============================================================================
# terrains/__init__.py — 自定义竞赛地形包
#
# 统一导出三个子地形配置类，供 env_cfgs.py 的 terrain_generator.sub_terrains 使用：
#   RCWallTerrainCfg           横向矮墙         → Robot-Rough-v0
#   RCLowBarTerrainCfg         低矮横杆（限高门）→ Robot-Crawl-v0
#   RCPyramidStairsTerrainCfg  金字塔台阶（改出生点）→ currently 未在 env_cfgs 中使用
# =============================================================================

from .competition_terrains import RCWallTerrainCfg, RCLowBarTerrainCfg, RCPyramidStairsTerrainCfg
from .stepping_stones import SteppingStonesTerrainCfg

__all__ = ["RCWallTerrainCfg", "RCLowBarTerrainCfg", "RCPyramidStairsTerrainCfg", "SteppingStonesTerrainCfg"]
