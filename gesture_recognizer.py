import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class HandGesture:
    name: str
    description: str


def _vector(a: Tuple[float, float], b: Tuple[float, float]) -> np.ndarray:
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=float)


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return angle in degrees between vectors."""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = max(min(cosang, 1.0), -1.0)
    return math.degrees(math.acos(cosang))


def fingers_up(landmarks_norm: List[Tuple[float, float]]) -> Dict[str, bool]:
    """
    Very simple heuristic: treat y decreasing as "up".

    landmarks_norm: list of (x, y) image-normalized coords for 21 MediaPipe points.
    Returns dict with keys: thumb, index, middle, ring, pinky.
    """
    # Indices for fingertips and lower joints in MediaPipe Hands
    tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    pips = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

    up: Dict[str, bool] = {}
    for name in tips:
        tip_y = landmarks_norm[tips[name]][1]
        pip_y = landmarks_norm[pips[name]][1]
        # finger is "up" if tip is above (smaller y) than pip
        up[name] = tip_y < pip_y
    return up


def pinch_distance(landmarks_norm: List[Tuple[float, float]]) -> float:
    """Distance between index fingertip and thumb tip."""
    thumb_tip = np.array(landmarks_norm[4])
    index_tip = np.array(landmarks_norm[8])
    return float(np.linalg.norm(index_tip - thumb_tip))


def pinch_distances(landmarks_norm: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Distances (normalized) between thumb tip and other fingertips.
    Keys: index, middle, ring, pinky.
    """
    thumb_tip = np.array(landmarks_norm[4])
    tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    out: Dict[str, float] = {}
    for k, idx in tips.items():
        out[k] = float(np.linalg.norm(np.array(landmarks_norm[idx]) - thumb_tip))
    return out


def classify_gesture(
    landmarks_norm: List[Tuple[float, float]],
    pinch_threshold: float = 0.05,
) -> HandGesture:
    """
    Classify a few basic gestures:
    - MOVE: index up, others down
    - PINCH_LEFT_CLICK: thumb-index pinch with index up
    - PINCH_RIGHT_CLICK: thumb-middle pinch (approx) with index and middle up
    - OPEN_PALM: all fingers up
    - FIST: all fingers down
    """
    up = fingers_up(landmarks_norm)
    d_pinch = pinch_distance(landmarks_norm)
    d = pinch_distances(landmarks_norm)

    all_up = all(up.values())
    all_down = not any(up.values())

    if all_up:
        return HandGesture("OPEN_PALM", "Open palm - neutral / scroll mode")
    if all_down:
        return HandGesture("FIST", "Fist - stop / reset")

    # Index only up → move cursor
    if up["index"] and not (up["middle"] or up["ring"] or up["pinky"]):
        return HandGesture("MOVE", "Move cursor with index finger")

    # Key-press pinches (one-shot press handled in controller)
    if up["middle"] and d["middle"] < pinch_threshold:
        return HandGesture("PINCH_SPACE", "Thumb-middle pinch (space)")
    if up["ring"] and d["ring"] < pinch_threshold:
        return HandGesture("PINCH_ENTER", "Thumb-ring pinch (enter)")
    if up["pinky"] and d["pinky"] < pinch_threshold:
        return HandGesture("PINCH_ESC", "Thumb-pinky pinch (esc)")

    # Thumb-index pinch for primary click
    if up["index"] and d_pinch < pinch_threshold:
        return HandGesture("PINCH_LEFT_CLICK", "Pinch for left click")

    # Index+middle up and close together → right click
    index_tip = np.array(landmarks_norm[8])
    middle_tip = np.array(landmarks_norm[12])
    d_index_middle = float(np.linalg.norm(index_tip - middle_tip))
    if up["index"] and up["middle"] and d_index_middle < pinch_threshold:
        return HandGesture("PINCH_RIGHT_CLICK", "Two-finger pinch for right click")

    # Default neutral
    return HandGesture("UNKNOWN", "Unclassified gesture")

