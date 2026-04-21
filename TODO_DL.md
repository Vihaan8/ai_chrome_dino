# DL Handoff Checklist (Anvita)

The DL code lives in this same repository, not a separate one. You own `perception_dl.py` and `planner_dl.py`. Everything under `app/` and `eval/` is shared; edit those only when both of us agree. Git-wise, work on a `dl` branch and rebase onto `main` when you need something we merged.

This file is a work queue. Check items off as you go.


## Big picture

The classical baseline is done and documented in `README.md`. Your job is to replace perception, or planning, or both, with a learned model and rerun the same evaluation. A DL version that only matches the classical numbers is still a legitimate finding and gets written up honestly. A DL version that beats classical on the hard failure case (late jumps at game speed 36 to 44) is the stronger story.


## Numbers to beat

Classical 100-episode results with decoys enabled (the current default game mode). Full breakdown in `eval/summary_100.txt`; reproducible by `python eval/run_eval.py --episodes 100 --impl classical`.

| Metric | Value |
|---|---|
| Mean score | 1,784 |
| Median score | 1,156 |
| Min score | 281 |
| Max score | 8,637 |
| Stdev | 1,882 |
| Percent reaching 1,000 frames | 54.0% |
| Percent reaching 5,000 frames | 8.0% |
| Percent reaching 10,000 frames (the cap) | 0.0% |
| Perception latency | 0.016 ms per frame |
| Planning latency | under 0.001 ms per frame |
| Ground (cactus) deaths | 67% |
| Flying (pterodactyl) deaths | 33% |
| Misclassification failures | 47% (clouds tagged as cacti) |
| Timing-error failures | 53% (acted on real obstacle but hit anyway) |

The cloud decoys drive classical's collapse: it jumps on every cloud, wastes 35 airborne frames, and gets hit by whatever real obstacle comes next. A DL perception module trained to distinguish clouds from cacti should recover most of the pre-decoy ceiling (mean score around 8,200) without reintroducing the cactus failures at speed. That is the headroom you are aiming for.


## Setup

- [ ] Clone this repo (if you haven't already), `pip install -r requirements.txt`.
- [ ] Create a `dl` branch: `git checkout -b dl`.
- [ ] Run `python main.py --seed 1 --impl classical` once to confirm the classical agent works on your machine.
- [ ] Run `python eval/run_eval.py --episodes 10 --impl classical` and confirm you reproduce highly variable scores (mean around 1,800, range 300 to 8,600). The variance is the point; decoys make classical erratic.
- [ ] Run `python main.py --impl dl`. You should see a `NotImplementedError` from `perception_dl.detect`. That is where your work starts.


## Interface

Both of your files must obey the frozen contract in `DL_INTERFACE.md`. Signatures cannot change, otherwise the shared harness breaks.

```
perception_dl.detect(frame, cfg) returns
  {'present': bool,
   'distance': int or None,     # pixels from dino right edge (x=90) to obstacle left edge; may be negative
   'type': 'ground' or 'flying' or None,
   'height': int or None}       # top-Y of obstacle in frame coordinates

planner_dl.decide(obstacle_info, game_speed, cfg) returns
  'none' or 'jump' or 'duck'
```

- [ ] Your `detect` accepts the same `cfg` dict parsed from `app/config.yaml` and uses `perception.dino_right_edge` for distance reference.
- [ ] Your `decide` can keep internal state (frame stack, recurrence) but cannot read from the `Game` object directly.
- [ ] Any DL-specific keys go under a new `dl:` section in `app/config.yaml`. Do not edit existing keys. Example:

```yaml
dl:
  model_path: weights/cnn.pt
  input_size: [84, 84]
  device: cpu
```


## Data Collection

The classical agent's JSON logs in `eval/runs/` include both what perception saw (`obstacle`) and what the game actually had (`obstacles_raw`, with real types including `'decoy'`). Use the `obstacles_raw` list as ground truth; the `obstacle` field is classical perception's output and is wrong for clouds. This distinction is the whole point of the DL contribution.

- [ ] Decide on your labeling approach: use `obstacles_raw` as ground truth, or hand-label a smaller sample.
- [ ] When labeling, map `'decoy'` to "not an obstacle" (or a separate class "decoy" depending on your output head design). The point is that your model must learn to ignore clouds even though they look like cacti to a contour detector.
- [ ] Export training frames. Suggested target is 2,000 to 5,000 labeled frames. Too few and you overfit; too many is just disk cost.
- [ ] Split by seed, not by frame. Train seeds, validation seeds, test seeds must be disjoint so the model cannot memorize a particular run.
- [ ] For VLM-style approaches (Claude, GPT-4o-mini with frames), skip labeling and prompt the model at inference time instead.


## Suggested Approaches

Pick one for the first iteration.

1. Small CNN classifier. Input is an 84 by 84 grayscale crop. Output is either `(jump, duck, none)` directly, or a structured `(type, distance_bucket, height)` that feeds into the classical planner. Target inference is under 2 ms per frame on CPU.
2. YOLOv8n fine-tuned on cacti and pterodactyl crops. Detector produces `obstacle_info`, classical planner reads it. Clean story for the write-up because the detector is a real pretrained model doing object detection.
3. VLM in the loop. Prompt Claude or GPT-4o-mini with the raw frame and a short system prompt; return one of `jump`, `duck`, `none`. Slow (hundreds of ms per frame) but different enough from classical to be a real contribution.
4. Pretrained DQN from a public Chrome Dino repository. Adapt the output head to the three-action space we use.


## Eval Parity

Comparison is only meaningful if both versions run under identical conditions.

- [ ] Same seed list in `eval.seeds` (config.yaml does not change).
- [ ] Same `eval.episodes` and `eval.max_frames`.
- [ ] Same `app/game.py` source (do not fork the game).
- [ ] Run `python eval/run_eval.py --episodes 100 --impl dl` from this repo. The `--impl dl` tag lands in the JSON filenames so both sets of runs can live in `eval/runs/` side by side.
- [ ] Report the same metrics: score stats, obstacles_cleared, survival thresholds (1k, 5k, cap), death cause breakdown, perception and planning ms per frame, and failure categorization from `eval/failure_analysis.py`.


## Deliverables

- [ ] Working `python main.py --impl dl` that plays one episode end-to-end.
- [ ] `python eval/run_eval.py --episodes 100 --impl dl` producing 100 JSON logs in `eval/runs/`.
- [ ] `eval/summary_100_dl.txt` filled in with your real numbers. The file already exists with the right template; replace every `TBD`.
- [ ] `README.md` DL Results section filled in. All tables exist with `TBD` placeholders under the `## DL Results` heading. Drop in your numbers.
- [ ] `README.md` Classical vs DL Comparison section filled in. The comparison table already has classical's numbers in the first column; fill the DL column and the delta column.
- [ ] Short analysis (two paragraphs at most) at the bottom of the Comparison section describing where DL beat or lost to classical and why. If DL only matches classical on mean score, write that honestly; it is a legitimate result.


## Where your numbers need to land

When you finish the 100-episode run, three files get touched:

| File | What to update |
|---|---|
| `eval/summary_100_dl.txt` | Replace every `TBD` with real numbers. This is the auditable source of truth, same format as classical's summary. |
| `README.md` under `## DL Results` | The score, cleared, percentile, threshold, death-cause, failure-analysis, and latency tables all have TBD rows. Fill them. Also note which pipeline modules are learned under "Which parts of the pipeline are learned". |
| `README.md` under `## Classical vs DL Comparison` | The comparison table has classical's numbers in the first column; fill the DL column and the delta column. Then write the short analysis. |

Both implementations' JSON logs (`run_classical_*.json` and `run_dl_*.json`) live in `eval/runs/` side by side. `failure_analysis.py` reads both impls and reports combined counts; split by impl yourself if you want per-impl breakdowns for the write-up.


## Common Pitfalls

- Rendering is not in the perception pipeline. `game.get_frame()` does a `surfarray` copy that takes around 10 microseconds. Do not optimize it, optimize your model.
- The dino's own body is inside the perception crop on purpose, so we can still see obstacles flying over it. A naive detector will label the dino as an obstacle every frame. Either mask the dino region in your training data or pass the dino's known x-range as an extra input channel to the model.
- Distance can be negative when a pterodactyl is directly above a ducking dino. Classical perception returns those cases so the planner stays in `duck` until the ptero clears. Do not clip distance to >= 0 in your inference code.
- Game speed is not visible in the frame. Pass it via `cfg` or recover it from consecutive frames in your planner state; do not assume it is constant.


## Coordination with Vihaan

- Edits to `app/` or `eval/` need a heads-up. Ping before pushing.
- If you hit a bug in the classical side (for example perception misbehaving on a new sprite), file it or push a patch to `main` rather than working around it in `perception_dl.py`.
- If we decide to add randomization to the game (day or night mode, obstacle variants), that is a joint change and we will re-run both baselines together.
