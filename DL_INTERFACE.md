# DL Interface

Contract for the DL version of this project. The partner's repo should mirror
these module signatures so both implementations share the same eval harness.

## Module signatures

### `perception.py`

```
detect(frame, cfg) -> obstacle_info
```

- `frame`: BGR numpy array, shape `(H, W, 3)`, dtype uint8.
- `cfg`: parsed `app/config.yaml`.
- Returns:
  ```
  {
    'present':  bool,
    'distance': int or None,   # px from dino's right edge to obstacle's left edge
    'type':     'ground' or 'flying' or None,
    'height':   int or None,   # top-Y of obstacle in frame coordinates
  }
  ```

Distance may be negative when the obstacle is horizontally overlapping the dino
(e.g. a pterodactyl passing over a ducking dino). Classical perception detects
this case; the DL version should too.

### `planner.py`

```
decide(obstacle_info, game_speed, cfg) -> action
```

Returns `'none'`, `'jump'`, or `'duck'`. Classical planner is stateless; a DL
planner may keep state internally (frame stack, recurrent) but must not touch
game internals.

## What to replace

- Perception only: swap `perception.py`, keep the classical planner. Easiest
  path — a small CNN mapping a crop to `(type, distance, height)`.
- Planner only: keep classical perception, learn the action policy.
- Both: full end-to-end.

## Config

Add DL-specific keys under a `dl:` section in `config.yaml`:

```yaml
dl:
  model_path: weights/cnn.pt
  input_size: [84, 84]
```

Do not change existing keys or the eval harness breaks.

## Eval

Both implementations run the same script with the same seeds:

```
python eval/run_eval.py --impl classical
python eval/run_eval.py --impl dl         # in partner's repo
python eval/failure_analysis.py
```

Seeds and episode count live in `eval.seeds` and `eval.episodes`. Keep them
identical between repos so the comparison is apples-to-apples.

Output: `eval/runs/run_<impl>_<seed>_<i>.json`, one file per episode. Frame-by-
frame log of obstacle info + action taken. Failure analysis reads these and
categorizes: survived, missed_detection, misclassification, late_reaction,
timing_error.

## Suggested approaches

- Small CNN classifier on ~500 labeled frames, output head: `{jump, duck, none}`.
- YOLOv8n fine-tuned on cacti + pterodactyl crops, pair with classical planner.
- VLM (Claude / GPT-4V) with a prompt template describing the scene; slower but
  interesting for the 2026 write-up.
- Pretrained DQN for Chrome Dino if you find one.

Target: inference under ~5 ms per frame. A model that beats classical on score
but runs at 50 ms/frame is too slow — the agent won't react in time.
