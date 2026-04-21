# DL Partner Handoff Checklist

This is the concrete work queue for the DL half of the project. The classical baseline is complete and documented in `README.md`; the contract both versions must obey is in `DL_INTERFACE.md`.

Numbers to beat (classical, 100 seeded episodes):

| Metric | Value |
|---|---|
| Mean score | 8,241 |
| Median score | 8,354 |
| Min / Max | 7,616 / 9,441 |
| % reaching 5,000 frames | 100.0% |
| % reaching 10,000 (cap) | 0.0% |
| Perception latency | 0.015 ms/frame |
| Planning latency | < 0.001 ms/frame |
| Failure mode | 100% `timing_error` at game speed 36–44 |


## Setup

- [ ] Clone this repo locally and run `python eval/run_eval.py --episodes 10` end-to-end once. Confirm you reproduce the numbers above before touching anything.
- [ ] Create a parallel repo for the DL version, matching this folder structure: `main.py`, `perception.py`, `planner.py`, `app/game.py`, `app/controller.py`, `app/config.yaml`, `eval/run_eval.py`, `eval/failure_analysis.py`.
- [ ] Copy `app/game.py` verbatim. The game is shared — don't fork it. If you fix a bug, patch this repo and pull.
- [ ] Keep `planner.py` unchanged to start (classical planner). Swap only `perception.py` for the first iteration.


## Interface

Both functions are imported by `main.py` and `eval/run_eval.py` exactly as in this repo. Signatures are frozen.

```
perception.detect(frame, cfg) -> {
    'present': bool,
    'distance': int or None,   # px from dino right edge (90) to obstacle left edge; can be negative
    'type': 'ground' or 'flying' or None,
    'height': int or None,     # top-Y of obstacle in frame coords
}

planner.decide(obstacle_info, game_speed, cfg) -> 'none' | 'jump' | 'duck'
```

- [ ] Your `detect` must accept the same `cfg` dict (parsed from `app/config.yaml`) and respect the `perception.dino_right_edge` key for distance reference.
- [ ] Your `decide` may be stateful internally, but must not touch `Game` or its attributes.
- [ ] Any DL-specific parameters (model path, input size, device, batch size) go under a `dl:` section in `config.yaml`. Do not change existing keys.


## Data Collection

The classical agent already produces labeled frames — every JSON log in `eval/runs/` pairs a frame's raw state with a perception output. You can use this as a weak labeler, or sample frames directly.

- [ ] Export labeled frames: run `python eval/run_eval.py --episodes 20`, then write a small script that reads each run's log, replays the game with the same seed, saves frames as PNGs, and writes a label CSV with `(frame_path, obstacle_type, distance, height, action)`. Target: 2,000–5,000 labeled frames.
- [ ] Split by seed, not by frame: train seeds, val seeds, test seeds must be disjoint. Use a 70/15/15 split on the seed list.
- [ ] For VLM approaches, skip labeling entirely and prompt the model with the raw frame.


## Suggested Approaches

Pick one for the first pass.

1. **Small CNN classifier.** Input: 84×84 grayscale crop. Output head: `(type, distance_bucket, height)` or directly `(jump, duck, none)`. Train on the exported frames. Target inference: ≤ 2 ms/frame on CPU.
2. **YOLOv8n fine-tuned** on cacti + pterodactyl crops. Use the detector to produce `obstacle_info`, keep the classical planner. Nice for the writeup because it's a pretrained model doing something real.
3. **VLM-in-the-loop.** Prompt Claude or GPT-4o-mini with the frame and a short system prompt; have it return one of `jump`, `duck`, `none` directly. Slow per frame (~200 ms+) but distinct from the classical approach and topical for 2026.
4. **Pretrained DQN** from an existing Chrome Dino project on GitHub. Adapt the output to our action space.


## Eval Parity

The comparison is only meaningful if both versions run under identical conditions.

- [ ] Same seed list in `eval.seeds` in both `config.yaml` files.
- [ ] Same `eval.episodes`, `eval.max_frames`.
- [ ] Same `app/game.py` source (same physics, same obstacle distribution).
- [ ] Run `python eval/run_eval.py --impl dl` in your repo — the `--impl` tag flows into the JSON filenames so we can merge log sets for the writeup.
- [ ] Report the same metrics: score stats, obstacles_cleared, survival thresholds (1k/5k/cap), death cause, perception and planning ms/frame, failure categorization.


## Deliverables

- [ ] DL repo with a working `python main.py` that plays one episode end-to-end.
- [ ] `python eval/run_eval.py --episodes 100 --impl dl` producing 100 JSON logs.
- [ ] A `README.md` in your repo following the same structure as this one (motivation → tree → run → architecture → perception/planner → eval → results → limitations). Don't invent numbers — run the eval and use the real ones.
- [ ] One combined comparison table (yours vs classical) for the shared writeup / video.
- [ ] A short section on where your DL version outperforms or underperforms classical and why. If it only matches classical's ~8,241 mean score, that's a legitimate finding — don't inflate it.


## Common Pitfalls

- Rendering is not in the perception pipeline, but `game.get_frame()` does copy the `surfarray`. If inference is already fast, don't waste time optimizing the frame extraction.
- The dino's own body is inside the perception crop. If a naive detector classifies it as an obstacle, perception will report `present=True` on every frame. The classical version handles this with a fixed-x-band filter; you may need an equivalent in your model (mask the dino region in training data, or provide the dino's x-range as an input feature).
- Distance can be negative when a pterodactyl is directly above a ducking dino. The classical planner uses this to stay in `duck`. Don't clip distance to ≥ 0 at inference.
- Game speed is not in the frame. Pass it separately via `cfg` or maintain it in your planner state.
