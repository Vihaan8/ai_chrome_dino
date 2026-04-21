# DL Handoff Checklist (Anvita)

The DL code lives in this same repository, not a separate one. You own `perception_dl.py` and `planner_dl.py`. Everything under `app/` and `eval/` is shared; edit those only when both of us agree. Git-wise, work on a `dl` branch and rebase onto `main` when you need something we merged.

This file is a work queue. Check items off as you go.


## Big picture

The classical baseline is done and documented in `README.md`. Your job is to replace perception, or planning, or both, with a learned model and rerun the same evaluation. A DL version that only matches the classical numbers is still a legitimate finding and gets written up honestly. A DL version that beats classical on the hard failure case (late jumps at game speed 36 to 44) is the stronger story.


## Numbers to beat

These are the classical 100-episode results. They live in `eval/summary_100.txt` and are reproducible by `python eval/run_eval.py --episodes 100 --impl classical`.

| Metric | Value |
|---|---|
| Mean score | 8,241 |
| Median score | 8,354 |
| Min score | 7,616 |
| Max score | 9,441 |
| Percent reaching 5,000 frames | 100.0% |
| Percent reaching 10,000 frames (the cap) | 0.0% |
| Perception latency | 0.015 ms per frame |
| Planning latency | under 0.001 ms per frame |
| Failure mode | 100 percent timing error on cacti at game speeds 36 to 44 |


## Setup

- [ ] Clone this repo (if you haven't already), `pip install -r requirements.txt`.
- [ ] Create a `dl` branch: `git checkout -b dl`.
- [ ] Run `python main.py --seed 1 --impl classical` once to confirm the classical agent works on your machine.
- [ ] Run `python eval/run_eval.py --episodes 10 --impl classical` and confirm you reproduce scores around 7,600 to 8,400.
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

The classical agent already produces a labeled dataset in `eval/runs/*.json`. Each JSON log pairs the frame state with what perception saw and what the planner decided, per frame. You can use those logs directly, or replay the same seeds and capture frames as PNGs with your own labels.

- [ ] Decide on your labeling approach: use the classical perception output as a weak label, or hand-label a smaller sample.
- [ ] Export training frames. Suggested target is 2,000 to 5,000 labeled frames. Too few and you overfit; too many and you are just paying ffor disk space.
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
- [ ] A short "DL results" section added to `README.md` following the same structure as the classical results section. Use real numbers, not placeholders.
- [ ] One combined comparison table (classical versus DL) in the README for the final write-up or video.
- [ ] A short analysis of where the DL version beats or loses to classical and why. If the DL version only matches the classical 8,241 mean score, write that honestly; it is a legitimate result.


## Common Pitfalls

- Rendering is not in the perception pipeline. `game.get_frame()` does a `surfarray` copy that takes around 10 microseconds. Do not optimize it, optimize your model.
- The dino's own body is inside the perception crop on purpose, so we can still see obstacles flying over it. A naive detector will label the dino as an obstacle every frame. Either mask the dino region in your training data or pass the dino's known x-range as an extra input channel to the model.
- Distance can be negative when a pterodactyl is directly above a ducking dino. Classical perception returns those cases so the planner stays in `duck` until the ptero clears. Do not clip distance to >= 0 in your inference code.
- Game speed is not visible in the frame. Pass it via `cfg` or recover it from consecutive frames in your planner state; do not assume it is constant.


## Coordination with Vihaan

- Edits to `app/` or `eval/` need a heads-up. Ping before pushing.
- If you hit a bug in the classical side (for example perception misbehaving on a new sprite), file it or push a patch to `main` rather than working around it in `perception_dl.py`.
- If we decide to add randomization to the game (day or night mode, obstacle variants), that is a joint change and we will re-run both baselines together.
