# Dinosaur Agent: Classical vs Deep Learning

A class project for IDS 705 (Duke) on agentic AI. We built two agents that play Chrome Dino, one rule-based and one learned, and compared how well they perform on the same game.

![Gameplay overview](docs/screenshot_overview.png)
<!-- screenshot placeholder: general gameplay with dinosaur running, cacti, pterodactyls, and a cloud visible -->


## What is Chrome Dino

Chrome Dino is the offline mini-game built into Google Chrome. A pixel-art dinosaur runs to the right. The player jumps over cacti that come from the right side of the screen and ducks under pterodactyls that fly across. The game speed continuously increases, so reactions have to get faster as the run goes on. The score is how long the dinosaur survives.

We built a Pygame clone of Chrome Dino and put two agents in front of it.


## What we built

Two agents that read the screen pixel by pixel, decide whether to jump, duck, or do nothing, and send the action back to the game. No browser, no keystrokes, just a direct Python API.

- **Classical agent**. Hand-coded rules. It finds dark shapes using a brightness threshold and decides what to do based on where those shapes are. Fast but brittle; it knows nothing about what a shape actually is.
- **DL agent**. A small convolutional neural network. It uses the same shape detection, but each shape is then classified by the network as cactus, pterodactyl, or cloud. It was trained on 75,000 labeled examples.

Both agents share the same game, the same planner (the rule that decides jump vs duck vs nothing given a classified obstacle), and the same 100 deterministic seeded episodes. The only difference between them is how perception classifies a shape.


## The core idea

We added light-blue cloud decoys to the game. Clouds drift through but don't collide with anything. To a human, clouds look completely different from cacti.

![Classical jumps on a cloud](docs/screenshot_classical_cloud.png)
<!-- screenshot placeholder: classical agent airborne over a cloud with a real cactus about to arrive -->

The classical agent cannot tell a cloud from a cactus. A cloud is a dark blob to its threshold filter, a cactus is a dark blob, they look the same. When it sees a cloud, it jumps, committing the dinosaur to 35 airborne frames with no ability to react. If a real cactus arrives during that jump, the dinosaur lands on it.

![DL agent ignores the cloud](docs/screenshot_dl_cloud.png)
<!-- screenshot placeholder: DL agent running normally with a cloud passing overhead -->

The DL agent learned during training that clouds are a third class and don't require action. It runs right through them.

You could patch the classical agent by adding a rule like "if the dark blob is more than 40 pixels wide, ignore it". That fixes this specific cloud. Change the cloud's size, color, or shape and you write another rule. Add a new obstacle and you write another. A learned model generalizes from examples without new rules. That tradeoff between rules and learning is what this project is about.


## Results

100 seeded episodes. Same game, same seeds, same planner. Only perception differs.

| Metric | Classical | DL |
|---|---|---|
| Mean score | 2,511 | 4,869 |
| Median score | 1,302 | 5,253 |
| Runs reaching 5,000 frames | 16% | 50% |
| Cloud misclassification (clouds tagged as obstacles) | 57% of failures | 4% of failures |
| Perception time per frame | 0.016 ms | 0.181 ms |

DL roughly doubles mean score and quadruples median. The cloud confusion that dominates classical almost disappears with the learned classifier.

![Score distribution classical vs DL](docs/screenshot_score_histogram.png)
<!-- screenshot placeholder: histogram or bar chart comparing classical vs DL score distributions -->

Full per-run numbers are in `eval/summary_100.txt` (classical) and `eval/summary_100_dl.txt` (DL).


## How to run

Install dependencies once:

```bash
pip install -r requirements.txt
```

Watch an episode in a window:

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

Retrain the DL model (about 5 to 10 minutes, runs from scratch):

```bash
python train_perception_dl.py
```


## How the two agents actually work

**Classical perception** crops the area in front of the dinosaur, converts it to grayscale, thresholds so pixels darker than 220 become "on", and uses OpenCV contour detection to find connected dark blobs. For each blob it reads the bottom Y position: near the ground means cactus, higher means pterodactyl. The nearest blob ahead of the dinosaur is returned.

**DL perception** runs the same contour detection to find candidate bounding boxes, then passes each 32x32 patch through a small CNN (three convolutional blocks, BatchNorm, Dropout) trained on 75,000 labeled patches. The CNN outputs one of three classes: cactus, pterodactyl, or cloud. Clouds are discarded and the nearest real obstacle is returned.

**The planner** is the same in both cases. Given the classified obstacle, game speed, and a reaction distance that scales with speed, it decides jump, duck, or nothing based on an if-else rule.


## Repository

```
.
├── main.py                        # entry point, plays the game
├── perception.py                  # classical detector
├── planner.py                     # shared rule-based planner
├── perception_dl.py               # DL detector
├── planner_dl.py                  # DL planner, delegates to classical
├── model_dl.py                    # CNN architecture
├── train_perception_dl.py         # data collection and training script
├── weights/cnn.pt                 # trained CNN weights
├── app/
│   ├── game.py                    # Pygame clone of Chrome Dino
│   ├── controller.py              # action dispatcher
│   └── config.yaml                # all thresholds and settings
├── eval/
│   ├── run_eval.py                # batch seeded runs
│   ├── failure_analysis.py        # classify deaths into 5 buckets
│   ├── summary_100.txt            # classical 100-episode results
│   └── summary_100_dl.txt         # DL 100-episode results
├── DL_INTERFACE.md                # frozen contract both implementations obey
├── TODO_DL.md                     # DL implementation notes
└── README.md
```


## Team

Vihaan Manchanda, Anvita Suresh

IDS 705, Duke University
