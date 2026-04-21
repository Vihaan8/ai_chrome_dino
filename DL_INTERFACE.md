# DL Interface

Frozen contract that both implementations (classical and DL) obey. Both implementations live in this repository. The `--impl` flag on `main.py` and `eval/run_eval.py` picks which module pair to import.


## Module pairs

| Implementation | Perception module | Planner module | Owner |
|---|---|---|---|
| Classical | `perception.py` | `planner.py` | Vihaan |
| DL | `perception_dl.py` | `planner_dl.py` | Anvita |


## Signatures

```
detect(frame, cfg) -> {
    'present': bool,
    'distance': int or None,     # pixels from dino right edge (x=90) to obstacle left edge; may be negative
    'type': 'ground' or 'flying' or None,
    'height': int or None,       # top-Y of obstacle in frame coordinates
}

decide(obstacle_info, game_speed, cfg) -> 'none' or 'jump' or 'duck'
```

- `frame` is a BGR numpy array, shape (200, 600, 3), dtype uint8. The Pygame clone renders it through `surfarray` and converts to BGR.
- `cfg` is the parsed `app/config.yaml` dict. Both implementations read the same keys; DL-specific parameters live under a `dl:` section that the classical modules ignore.
- `game_speed` is a float in pixels per frame. It starts at 6.0 and increases by 0.004 per frame.


## Distance semantics

Distance is measured from the dino's right edge at x=90 to the obstacle's left edge. A distance of zero means the obstacle is touching the dino's right side. A negative distance means the obstacle's left edge has passed the dino's right edge; this happens when a pterodactyl is horizontally overlapping the dino. Classical perception returns negative distances on purpose so the planner stays in `duck` until the threat clears. DL perception should do the same.


## What a DL implementation can replace

- Perception only. Replace `perception_dl.py` with a learned detector. Keep `planner_dl.py` as the default delegation to classical `planner.decide`.
- Planner only. Leave `perception_dl.py` as a delegation to classical `perception.detect` and put the learned policy in `planner_dl.py`.
- Both. End-to-end replacement.

`planner_dl.py` ships as a delegation to the classical planner by default. Change only what needs changing.


## Shared vs private

| Path | Who can edit |
|---|---|
| `perception.py`, `planner.py` | Vihaan |
| `perception_dl.py`, `planner_dl.py` | Anvita |
| `app/`, `eval/`, `main.py`, `DL_INTERFACE.md`, `TODO_DL.md`, `README.md` | Either, with coordination |

Any new key added under a `dl:` section in `app/config.yaml` is owned by Anvita and ignored by the classical modules. Any new key outside that section needs both owners to agree.


## Eval

```
python eval/run_eval.py --impl classical --episodes 100
python eval/run_eval.py --impl dl --episodes 100
python eval/failure_analysis.py
```

The same seeds, the same `max_frames`, the same game code. JSON logs for both implementations land in `eval/runs/` and are tagged `run_classical_*.json` or `run_dl_*.json`. `failure_analysis.py` reads every file in `runs/` regardless of impl.
