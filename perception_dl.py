"""
DL perception. Owner: Anvita Suresh.

Pipeline: classical contour detection finds candidate bounding boxes,
then a small CNN classifies each crop as ground / flying / decoy.
Decoys are suppressed; geometry (distance, height) is read from the bbox.

Contract (DL_INTERFACE.md):
    detect(frame, cfg) -> {'present': bool,
                           'distance': int or None,
                           'type': 'ground' | 'flying' | None,
                           'height': int or None}

Train the model first:  python train_perception_dl.py
"""

import os

import cv2
import numpy as np
import torch

from model_dl import ObstacleClassifier

# each crop is resized to this before going into the cnn
CROP_SIZE = 32
# padding added around each contour bbox so the cnn sees a bit of context
PAD = 8

# loaded once on first call to detect(), then reused every frame
_model = None
_device = None


def _load_model(cfg):
    global _model, _device
    # only load once — skip if already in memory
    if _model is not None:
        return
    root = os.path.dirname(os.path.abspath(__file__))
    dl = cfg.get("dl", {})
    path = dl.get("model_path", os.path.join(root, "weights", "cnn.pt"))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model weights not found at {path}. " "Run: python train_perception_dl.py"
        )
    _device = torch.device(dl.get("device", "cpu"))
    _model = ObstacleClassifier().to(_device)
    _model.load_state_dict(torch.load(path, map_location=_device, weights_only=True))
    _model.eval()


def _candidates(frame, cfg):
    """return contour bounding boxes (abs coords), dino excluded."""
    p = cfg["perception"]
    x0, x1 = p["crop_x_start"], p["crop_x_end"]
    y0, y1 = p["crop_y_start"], p["crop_y_end"]
    # crop to the region where obstacles actually appear
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # dark pixels become white so contours outline the obstacles
    _, binary = cv2.threshold(gray, p["threshold"], 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # skip tiny noise blobs
        if w * h < p["min_contour_area"]:
            continue
        ax, ay = x + x0, y + y0
        # skip the dino's own contour — it's ground-touching and in the dino x-band
        if (
            ay + h >= p["ground_line_y"] - p["ground_tolerance"]
            and ax >= p["dino_mask_x_start"]
            and ax + w <= p["dino_mask_x_end"]
        ):
            continue
        boxes.append((ax, ay, w, h))
    return boxes


def _make_patches(frame, boxes):
    # for each contour bbox, cut out a padded grayscale crop and resize to 32x32
    H, W = frame.shape[:2]
    patches = []
    for ax, ay, w, h in boxes:
        x0 = max(0, ax - PAD)
        y0 = max(0, ay - PAD)
        x1 = min(W, ax + w + PAD)
        y1 = min(H, ay + h + PAD)
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        patches.append(
            cv2.resize(gray, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        )
    return patches


def _classify(patches):
    # normalize to [0,1], add channel dim, run through cnn, return predicted class per patch
    X = np.array(patches, dtype=np.float32) / 255.0
    X = X[:, np.newaxis, :, :]
    with torch.no_grad():
        logits = _model(torch.tensor(X, device=_device))
        return logits.argmax(1).cpu().numpy().tolist()


def detect(frame, cfg):
    _load_model(cfg)
    p = cfg["perception"]
    dino_right = p["dino_right_edge"]
    ground_y = p["ground_line_y"]
    tol = p["ground_tolerance"]

    boxes = _candidates(frame, cfg)
    if not boxes:
        return {"present": False, "distance": None, "type": None, "height": None}

    preds = _classify(_make_patches(frame, boxes))

    # walk through every candidate and keep only real obstacles (not decoys)
    nearest = None
    for (ax, ay, w, h), pred in zip(boxes, preds):
        if pred == 0:  # decoy — ignore it, this is the whole point of the dl version
            continue
        # use y position to distinguish ground obstacles from flying ones, same as classical
        kind = "ground" if (ay + h) >= ground_y - tol else "flying"
        dist = ax - dino_right
        # keep whichever real obstacle is closest to the dino
        if nearest is None or dist < nearest["distance"]:
            nearest = {
                "present": True,
                "distance": int(dist),
                "type": kind,
                "height": int(ay),
            }

    if nearest is None:
        return {"present": False, "distance": None, "type": None, "height": None}
    return nearest
