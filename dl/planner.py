"""
DL planner.

Must match the contract in DL_INTERFACE.md:
    decide(obstacle_info, game_speed, cfg) -> 'none' | 'jump' | 'duck'

If the DL contribution is on perception only, this file imports from
planner.py and re-exports `decide`. If the planner itself is learned,
implement it here.

Default: delegate to the classical planner. Swap when needed.
"""

from classical.planner import decide as _classical_decide


def decide(obstacle_info, game_speed, cfg):
    return _classical_decide(obstacle_info, game_speed, cfg)
