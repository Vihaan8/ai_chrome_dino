# DL Implementation Notes

Reference for the DL pipeline in this repo: what it does, how it was trained, how to retrain, and what is still open. The frozen interface both implementations obey is in `DL_INTERFACE.md`; the headline numbers live in `README.md` and `eval/summary_100_dl.txt`.


## Current state

- `dl/perception.py` is a two-stage cascade: classical contour detection proposes candidate bounding boxes, then `dl.model.ObstacleClassifier` classifies each 32x32 patch as `decoy`, `ground`, or `flying`. Decoys are suppressed. Distance and height are read from the bbox, same references as classical (`dino_right_edge = 90`, ground tolerance 6).
- `dl/planner.py` delegates to the classical rule-based planner. No learned planner yet.
- `dl/model.py` defines a 3-conv-block CNN with BatchNorm and Dropout 0.3.
- Trained weights live in `dl/weights/cnn.pt` (about 600 KB) and are loaded once per process.


## Results on the current game

100 seeded episodes, light-blue random-Y cloud decoys, perception threshold 220.

| Metric | Classical | DL |
|---|---|---|
| Mean score | 2,511 | 4,869 |
| Median score | 1,302 | 5,253 |
| Percent reaching 5,000 | 16% | 50% |
| Percent reaching 10,000 (cap) | 0% | 0% |
| Perception latency | 0.016 ms | 0.181 ms |
| Misclassification failures | 57% | 4% |
| Missed-detection failures | 0% | 39% |
| Timing-error failures | 43% | 57% |

DL nearly doubles mean and quadruples median. The CNN solves the decoy problem almost completely (misclassification 4 percent vs 57 percent for classical). What remains: missed detections inherited from the classical contour stage (the CNN never sees what contour drops), and timing errors at high speed (a planner issue, not perception).


## Retraining

```bash
python dl/train.py
```

The script collects its own training data by running the classical agent on 80 seeded episodes (seeds 200 to 279, disjoint from eval seeds). It labels every contour bbox by IoU-matching against the game's ground-truth `obstacles_raw` list. Training runs 30 epochs of Adam with cosine learning rate decay, weight decay, class oversampling, horizontal flip and brightness augmentation, and best-checkpoint saving by validation accuracy.

Final weights save to `dl/weights/cnn.pt`, replacing whatever was there.

Last retrain produced 75,275 labeled patches and reached 100 percent validation accuracy because the color gap between light-blue clouds and dark cacti/pteros makes 3-class classification nearly trivial.


## Interface

```
dl.perception.detect(frame, cfg) -> {
    'present': bool,
    'distance': int or None,     # pixels from x=90 to obstacle left edge, may be negative
    'type': 'ground' or 'flying' or None,
    'height': int or None,       # top-Y of obstacle in frame coordinates
}

dl.planner.decide(obstacle_info, game_speed, cfg) -> 'none' or 'jump' or 'duck'
```

DL-specific config lives under a `dl:` section in `app/config.yaml`:

```yaml
dl:
  model_path: dl/weights/cnn.pt
  device: cpu
```


## Open questions and future work

Things that could push DL further if anyone wants to extend:

- **Replace the classical contour stage.** The 39 percent missed-detection rate is the ceiling on this cascade. A learned proposer (RPN-style, or an end-to-end detector) would catch obstacles the threshold filter drops.
- **Learn the planner.** The planner is still the hand-tuned reaction-distance rule. At high game speed (above 35) it runs out of margin regardless of perception quality. Imitation learning from classical trajectories, or an RL policy, could close the remaining gap toward the 8,200 pre-decoy ceiling.
- **Retrain on new game variants.** Any change to cloud appearance, sprite sizes, or obstacle types needs a retrain. Keep seeds 200 to 279 disjoint from the eval seeds.
- **Model compression.** The model is already small (~600 KB, 0.181 ms per frame) but if the eval set grows or the model does, quantization or pruning is trivial wins.


## Common pitfalls

- Rendering (`game.get_frame()`) does a `surfarray` copy at about 10 microseconds. Not worth optimizing vs model inference.
- The dino's body is inside the perception crop on purpose (so obstacles flying over it are still visible). The contour stage filters out ground-touching blobs whose x falls inside the dino band; the DL patches therefore never contain the dino alone.
- Distance can be negative (pterodactyl horizontally overlapping the dino). The planner relies on negative distances to keep ducking until the ptero clears. Do not clip.
- Game speed is not in the frame. The planner reads it from `game.game_speed` via the main loop; perception is frame-only.
