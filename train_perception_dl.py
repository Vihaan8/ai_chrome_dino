"""
Collect training data from live game episodes and train the CNN.

Run once:  python train_perception_dl.py
Output:    weights/cnn.pt
"""

import copy
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from app import controller
from app.game import Game
from model_dl import ObstacleClassifier
import perception as classical_perception
import planner as classical_planner

# size each crop gets resized to before saving / training
CROP_SIZE = 32
# pixels of padding added around each detected bbox
PAD = 8
# numeric labels for each obstacle type
LABEL_MAP = {"decoy": 0, "ground": 1, "flying": 2}


# ---------------------------------------------------------------------------
# Contour extraction (mirrors perception.py; returns all candidates)
# ---------------------------------------------------------------------------


def _candidates(frame, cfg):
    # same contour pipeline as classical perception — find all dark blobs in the crop region
    p = cfg["perception"]
    x0, x1 = p["crop_x_start"], p["crop_x_end"]
    y0, y1 = p["crop_y_start"], p["crop_y_end"]
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, p["threshold"], 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < p["min_contour_area"]:
            continue
        ax, ay = x + x0, y + y0
        # skip the dino — ground-touching blob inside the dino x-band
        if (
            ay + h >= p["ground_line_y"] - p["ground_tolerance"]
            and ax >= p["dino_mask_x_start"]
            and ax + w <= p["dino_mask_x_end"]
        ):
            continue
        boxes.append((ax, ay, w, h))
    return boxes


def _match_label(box, raw_obstacles):
    """iou-match a contour bbox to the closest raw obstacle; return its type or none."""
    bx, by, bw, bh = box
    best_iou, best_type = 0.0, None
    for o in raw_obstacles:
        ox, oy, ow, oh = o[0], o[1], o[2], o[3]
        # compute intersection area
        ix = max(0.0, min(bx + bw, ox + ow) - max(bx, ox))
        iy = max(0.0, min(by + bh, oy + oh) - max(by, oy))
        inter = ix * iy
        if inter == 0:
            continue
        iou = inter / (bw * bh + ow * oh - inter)
        if iou > best_iou:
            best_iou, best_type = iou, o[4]
    # require at least 20% overlap to count as a match, otherwise skip this blob
    return best_type if best_iou > 0.2 else None


def _patch(frame, ax, ay, w, h):
    # cut out a padded region around the bbox, convert to grayscale, resize to 32x32
    H, W = frame.shape[:2]
    x0 = max(0, ax - PAD)
    y0 = max(0, ay - PAD)
    x1 = min(W, ax + w + PAD)
    y1 = min(H, ay + h + PAD)
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect(n_episodes=80, max_frames_per_ep=3000, seeds=None):
    with open(os.path.join(ROOT, "app", "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    # use seeds 200-279 so they don't overlap with the eval seeds (1-5 x cohorts)
    seeds = seeds or list(range(200, 200 + n_episodes))
    game = Game(headless=True, fast=True)
    images, labels = [], []

    for i, seed in enumerate(seeds):
        game.reset(seed=seed)
        while not game.done and game.frame < max_frames_per_ep:
            frame = game.get_frame()
            # grab ground truth obstacle list before stepping (includes decoys with real types)
            raw = copy.deepcopy(game.obstacles)
            for box in _candidates(frame, cfg):
                # figure out what this blob actually is using game ground truth
                ltype = _match_label(box, raw)
                if ltype is None:
                    continue
                images.append(_patch(frame, *box))
                labels.append(LABEL_MAP[ltype])
            # use the classical agent to keep the dino alive longer = more frames with obstacles
            obs = classical_perception.detect(frame, cfg)
            action = classical_planner.decide(obs, game.game_speed, cfg)
            controller.apply(action, game)

        counts = {v: labels.count(v) for v in range(3)}
        print(
            f"[{i+1:3d}/{n_episodes}] seed={seed:4d}  "
            f"frames={game.frame:4d}  "
            f"total={len(labels)}  "
            f"decoy={counts[0]} ground={counts[1]} flying={counts[2]}"
        )

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _oversample(X, y):
    # repeat samples from minority classes so all three classes have equal representation
    _, counts = np.unique(y, return_counts=True)
    max_n = counts.max()
    Xs, ys = [X], [y]
    for cls in range(3):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        deficit = max_n - len(idx)
        if deficit > 0:
            # randomly duplicate existing samples to make up the difference
            extra = np.random.choice(idx, deficit, replace=True)
            Xs.append(X[extra])
            ys.append(y[extra])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def _augment(xb):
    # randomly flip horizontally and jitter brightness to help the model generalize
    if np.random.rand() < 0.5:
        xb = xb[:, :, :, ::-1].copy()
    noise = np.random.uniform(0.85, 1.15)
    return np.clip(xb * noise, 0.0, 1.0).astype(np.float32)


def train(images, labels, n_epochs=30, batch_size=64):
    # normalize pixel values to [0,1] and add channel dimension for pytorch
    X = images.astype(np.float32) / 255.0
    X = X[:, np.newaxis, :, :]  # (N, 1, 32, 32)
    X, y = _oversample(X, labels)

    # hold out 15% for validation
    n_val = max(1, int(0.15 * len(X)))
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    device = torch.device("cpu")
    model = ObstacleClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # cosine schedule gradually lowers the learning rate over training
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None

    for epoch in range(n_epochs):
        model.train()
        # shuffle training data each epoch
        perm = np.random.permutation(len(X_tr))
        X_tr, y_tr = X_tr[perm], y_tr[perm]
        total_loss, n_batches = 0.0, 0
        for b in range(0, len(X_tr), batch_size):
            xb = _augment(X_tr[b : b + batch_size])
            xb = torch.tensor(xb, device=device)
            yb = torch.tensor(y_tr[b : b + batch_size], device=device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        # check validation accuracy after each epoch and save the best weights
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, device=device)
            yv = torch.tensor(y_val, device=device)
            preds = model(xv).argmax(1)
            acc = (preds == yv).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {epoch+1:2d}/{n_epochs}  "
            f"loss={total_loss/n_batches:.4f}  val_acc={acc:.3f}"
        )

    # restore the best checkpoint before returning
    model.load_state_dict(best_state)
    print(f"\nBest val accuracy: {best_acc:.3f}")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)

    print("=== Collecting training data ===")
    images, labels = collect()
    unique, counts = np.unique(labels, return_counts=True)
    names = {0: "decoy", 1: "ground", 2: "flying"}
    print(f"\nTotal samples: {len(labels)}")
    for u, c in zip(unique, counts):
        print(f"  {names[u]:7s}: {c}")

    print("\n=== Training CNN ===")
    model = train(images, labels)

    out = os.path.join(ROOT, "weights", "cnn.pt")
    torch.save(model.state_dict(), out)
    print(f"\nSaved weights -> {out}")
    print("Done. Run: python main.py --impl dl")


if __name__ == "__main__":
    main()
