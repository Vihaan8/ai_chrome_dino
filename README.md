# Dinosaur Agent: Classical vs Deep Learning

A class project for IDS 705 (Duke) on agentic AI. We built two agents that play Chrome Dino, one rule-based and one learned, and compared how well they perform on the same game with the same seeds.

![Gameplay overview](docs/screenshot_overview.png)
<!-- screenshot placeholder: general gameplay showing the dinosaur running past cacti, a pterodactyl, and a cloud -->


## Table of Contents

1. [What is Chrome Dino](#what-is-chrome-dino)
2. [The Cloud Challenge](#the-cloud-challenge)
3. [Classical Approach](#1-classical-approach)
   - [Method](#method)
   - [What we built](#what-we-built)
   - [Why it matters](#why-it-matters)
   - [Decisions we made](#decisions-we-made)
4. [Deep Learning Approach](#2-deep-learning-approach)
   - [Method](#method-1)
   - [What we built](#what-we-built-1)
   - [Why it matters](#why-it-matters-1)
   - [Decisions we made](#decisions-we-made-1)
5. [Results](#results)
6. [Repository Layout](#repository-layout)
7. [How to Run](#how-to-run)
8. [Team](#team)


## What is Chrome Dino

Chrome Dino is the offline mini-game built into Google Chrome. A pixel-art dinosaur runs to the right. The player jumps over cacti that come from the right edge and ducks under pterodactyls that fly across. The game speed continuously increases, so reactions have to get faster as the run goes on. The score is how long the dinosaur survives.

We built a Pygame clone of Chrome Dino and put two agents in front of it. Both agents read the screen pixel by pixel, decide whether to jump, duck, or do nothing, and send the action back to the game. No browser, no keystrokes, just a direct Python API.


## The Cloud Challenge

To make the comparison between the two agents meaningful, we added light-blue cloud decoys to the game. Clouds drift across the screen at random heights but do not collide with anything. To a human, clouds look obviously different from a cactus.

![Classical jumps on a cloud](docs/screenshot_classical_cloud.png)
<!-- screenshot placeholder: classical agent airborne over a cloud while a real cactus is approaching -->

A rule-based agent cannot tell the difference. Its perception finds dark pixels and groups them into shapes. A cloud is a dark shape. A cactus is a dark shape. They look identical to a brightness threshold. So the classical agent jumps on every cloud, wastes 35 airborne frames (the full jump duration), and very often lands on the real cactus that arrives during those frames.

![DL agent ignores the cloud](docs/screenshot_dl_cloud.png)
<!-- screenshot placeholder: DL agent running normally while a cloud passes overhead -->

A learned agent sees the same shapes but classifies them by what they look like, not just whether they are dark. It learned from training examples that clouds are a separate class and do not require action. So it runs through them.

You could patch the classical agent. For example, add a rule "if the dark blob is more than 40 pixels wide, ignore it". That fixes this specific cloud. Change the cloud's size, color, or shape and the rule breaks and you write another rule. Add a new kind of obstacle and you write another. The learned model generalizes from examples without new rules. That difference between rules that break on new inputs and learned models that generalize is the whole point of this project.


## 1. Classical Approach

### Method

Classical computer vision. Find dark pixels, group them into shapes, classify each shape by its position on screen, and act based on a hand-tuned rule. No training data, no model weights, pure pixel math.

### What we built

Two small Python files doing very specific jobs.

- `perception.py`. Crops the area in front of the dinosaur. Converts to grayscale. Thresholds any pixel darker than grayscale 220 as "on". Runs OpenCV `findContours` to detect connected dark blobs. For each blob, reads the bottom Y position: bottom near the ground line is classified as cactus, higher is classified as pterodactyl. Returns the nearest blob in front of the dinosaur as `{present, distance, type, height}`.
- `planner.py`. Takes the classified obstacle, the current game speed, and the config. Returns `jump`, `duck`, or `none`. The decision is a single if-else: if the obstacle is within `70 + 2.0 * game_speed` pixels and is a cactus, jump; if it is a high pterodactyl, duck; otherwise do nothing.

Together they run in about 16 microseconds per frame on CPU.

### Why it matters

Classical CV is the right baseline. It is fast, interpretable, and has no training pipeline, so it sets a clear bar any learned model has to beat. If handcrafted rules are enough to play the game, you do not need a neural network. It also makes the failure modes explicit: whatever classical cannot do becomes the exact gap the DL version is trying to fill.

### Decisions we made

- **Fixed threshold instead of adaptive.** Grayscale cutoff of 220 picks up both real obstacles and the light-blue clouds. Adaptive thresholding (Otsu, adaptive mean) would handle varying brightness but adds complexity that this game's uniform background does not need.
- **Classify by position, not shape.** Contour bottom near the ground line means cactus; higher means pterodactyl. This avoids any shape classifier and keeps the pipeline purely geometric. It also means classical cannot distinguish clouds from cacti, which is the failure mode we want to measure.
- **Dino self-filter.** The dinosaur lives inside the perception crop on purpose, so we can still see obstacles flying over it. We filter the dinosaur's own contour by checking that a ground-touching blob falls inside a known x-band (45 to 95).
- **Linear reaction distance.** `reaction_distance = base + speed_factor * game_speed`. We deliberately picked a tight `speed_factor` of 2.0 so the agent runs out of reaction margin at high game speeds. This keeps a real, categorizable failure mode visible in the eval. Raising the factor removes it but hides the planner's limit.


## 2. Deep Learning Approach

### Method

A two-stage cascade. First stage is the same classical contour detection, which finds candidate bounding boxes. Second stage is a small convolutional neural network that classifies each 32x32 patch as cactus, pterodactyl, or cloud. Clouds are discarded. The nearest non-cloud obstacle is returned.

### What we built

Four files.

- `perception_dl.py`. Runs classical contour detection to find candidate bboxes, then passes each patch through the CNN. Suppresses clouds, returns the nearest real obstacle as the same `{present, distance, type, height}` dict classical uses.
- `model_dl.py`. The CNN architecture. Three convolution blocks (each Conv + BatchNorm + ReLU + MaxPool), then a classifier head (flatten, linear 128, Dropout 0.3, linear 3). Inputs a 32x32 grayscale patch, outputs three class logits.
- `train_perception_dl.py`. Collects training data by running the classical agent on 80 seeded episodes (seeds 200 to 279, disjoint from eval seeds). Labels each contour bbox by IoU-matching it against the game's ground-truth obstacle list. Trains for 30 epochs with Adam, cosine learning rate decay, class oversampling, horizontal flip and brightness augmentation, and best-checkpoint saving. Last run collected 75,275 labeled patches and reached 100 percent validation accuracy.
- `planner_dl.py`. Unchanged from classical. Delegates every decision to the classical rule-based planner.

Trained weights live in `weights/cnn.pt`, about 600 KB on disk, about 0.18 ms per frame at inference on CPU.

### Why it matters

This is the whole point of the comparison. Classical cannot distinguish shapes that look similar but carry different meanings (cloud vs cactus). A learned classifier can. Even a tiny CNN trained on a few tens of thousands of patches nearly eliminates the cloud confusion that dominates classical's failures (misclassification drops from 57 percent to 4 percent). It shows that once the scene contains semantic ambiguity, rule-based perception hits a ceiling that learning clears without a redesign.

### Decisions we made

- **Classifier cascade instead of end-to-end detector.** The CNN only classifies bboxes that classical already proposed. This keeps the model tiny, easy to train, and interpretable (you can look at each classified crop). The tradeoff is that if classical contour detection misses an obstacle, the CNN never sees it. In practice this shows up as 39 percent missed-detection failures, the structural ceiling of this approach.
- **Three positive classes, not two plus "nothing".** Cactus, pterodactyl, cloud. Treating cloud as a named class makes the decision boundary sharper and the training objective cleaner than a "real vs nothing" binary.
- **Supervised labels from the game's ground truth.** The game knows what it spawned. Every obstacle in `obstacles_raw` carries a type, including `decoy` for clouds. The training script IoU-matches each contour bbox against these and labels accordingly. No hand-labeling, but also no weak supervision.
- **Same planner as classical.** We only replaced perception. This keeps the comparison fair: any score difference between the two agents is attributable to perception, not to a better policy on top.
- **Small CNN.** About 600 KB, a handful of million multiply-adds per inference. Trains in five minutes on CPU, runs in under a millisecond per frame. Larger models would be overkill for a three-class problem where the classes are visually this distinct.


## Results

100 seeded episodes. Same game, same seeds, same planner. Only perception differs.

| Metric | Classical | DL |
|---|---|---|
| Mean score | 2,511 | 4,869 |
| Median score | 1,302 | 5,253 |
| Runs reaching 5,000 frames | 16% | 50% |
| Cloud misclassification (fraction of failures) | 57% | 4% |
| Perception time per frame | 0.016 ms | 0.181 ms |

DL roughly doubles mean score and quadruples median. The cloud confusion that dominated classical almost vanishes with the learned classifier. Perception is ten times slower in DL, but still well under a millisecond per frame and nowhere near the 16 ms per frame budget that 60 fps gameplay allows.

Full per-run numbers are in `eval/summary_100.txt` (classical) and `eval/summary_100_dl.txt` (DL). Failure breakdowns are produced by `eval/failure_analysis.py`.

![Score distribution classical vs DL](docs/screenshot_score_histogram.png)
<!-- screenshot placeholder: bar chart or histogram comparing classical and DL score distributions -->


## Repository Layout

```
.
├── main.py                        # entry point, plays one or more episodes; --impl classical or --impl dl
├── perception.py                  # classical detector: threshold, contours, geometric classification
├── planner.py                     # shared rule-based planner (jump / duck / none)
├── perception_dl.py               # DL detector: classical contour proposals, CNN classifies each
├── planner_dl.py                  # DL planner, delegates to classical (perception-only replacement)
├── model_dl.py                    # CNN architecture (3-class classifier)
├── train_perception_dl.py         # data collection + training, produces weights/cnn.pt
├── weights/cnn.pt                 # trained CNN weights (~600 KB)
├── requirements.txt
├── app/
│   ├── game.py                    # Pygame clone of Chrome Dino with cloud decoys
│   ├── controller.py              # action dispatcher
│   └── config.yaml                # all thresholds, crop region, eval settings
├── eval/
│   ├── run_eval.py                # batch 100-episode runner
│   ├── failure_analysis.py        # classify deaths into 5 buckets
│   ├── summary_100.txt            # classical 100-episode results
│   └── summary_100_dl.txt         # DL 100-episode results
├── DL_INTERFACE.md                # frozen contract both implementations obey
├── TODO_DL.md                     # DL implementation notes
└── README.md
```

The `app/` folder holds everything the agents do not decide: the game engine, the sprite rendering, the collision physics, the action dispatcher, and the shared configuration file. Agents import `Game` from here but do not edit it. The `eval/` folder runs both agents on the same seeds and produces comparable numbers. `DL_INTERFACE.md` documents the function signatures both `perception.py` / `planner.py` and `perception_dl.py` / `planner_dl.py` must obey; that contract is what lets `--impl` swap between them.


## How to Run

Install dependencies once:

```bash
pip install -r requirements.txt
```

Watch one episode in a window:

```bash
python main.py --impl classical --seed 1     # rule-based agent, jumps on clouds
python main.py --impl dl --seed 1            # learned agent, runs through clouds
```

Run the full 100-episode evaluation:

```bash
python eval/run_eval.py --impl classical --episodes 100
python eval/run_eval.py --impl dl --episodes 100
python eval/failure_analysis.py
```

Retrain the DL model from scratch (about 5 to 10 minutes):

```bash
python train_perception_dl.py
```

The script collects fresh training data by running the classical agent on 80 seeded episodes, trains the CNN, and saves the new weights to `weights/cnn.pt`. It does not touch anything else.


## Team

Vihaan Manchanda, Anvita Suresh

IDS 705, Duke University
