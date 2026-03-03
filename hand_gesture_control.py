import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from gesture_recognizer import HandGesture, classify_gesture


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")


@dataclass
class GestureState:
    last_gesture: Optional[str] = None
    last_gesture_time: float = 0.0
    click_cooldown: float = 0.25  # seconds
    key_cooldown: float = 0.45  # seconds
    scroll_cooldown: float = 0.03  # seconds
    # Pinch/drag state (thumb-index)
    pinch_index_active: bool = False
    pinch_index_start_time: float = 0.0
    dragging: bool = False
    drag_hold_time: float = 0.45  # seconds to start drag
    # Edge-trigger for other pinches (keys)
    last_key_gesture: Optional[str] = None
    last_key_time: float = 0.0
    # Scroll tracking
    last_palm_y: Optional[float] = None
    last_scroll_time: float = 0.0


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def ensure_hand_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return
    print(f"Downloading hand model to: {MODEL_PATH}", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model download complete.", flush=True)


def draw_hand_landmarks(frame_bgr: np.ndarray, landmarks_norm, color=(0, 255, 0)):
    h, w, _ = frame_bgr.shape
    pts = []
    for lm in landmarks_norm:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
        cv2.circle(frame_bgr, (x, y), 3, color, -1)

    for a, b in HAND_CONNECTIONS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], color, 2)


class HandGestureController:
    def __init__(self, camera_index: int = 0, smooth_factor: float = 0.25):
        self.cap = self._open_camera(camera_index)
        # Try to set a reasonable resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        self.screen_w, self.screen_h = pyautogui.size()

        ensure_hand_model()

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._vision = vision
        self._python = python
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.state = GestureState()
        self.prev_cursor: Optional[Tuple[int, int]] = None
        self.smooth_factor = smooth_factor

    def _open_camera(self, camera_index: int) -> cv2.VideoCapture:
        """
        Windows webcams often work better with CAP_DSHOW.
        We try a couple of backends and camera indices.
        """
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        indices = [camera_index, 0, 1, 2, 3]

        for idx in indices:
            for backend in backends:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                # Warm-up reads
                ok = False
                for _ in range(10):
                    ok, _frame = cap.read()
                    if ok:
                        break
                    time.sleep(0.05)
                if ok:
                    return cap
                cap.release()

        # Fallback: default backend
        return cv2.VideoCapture(camera_index)

    def _smooth_cursor(self, target_x: int, target_y: int) -> Tuple[int, int]:
        if self.prev_cursor is None:
            self.prev_cursor = (target_x, target_y)
            return target_x, target_y
        px, py = self.prev_cursor
        nx = int(px + self.smooth_factor * (target_x - px))
        ny = int(py + self.smooth_factor * (target_y - py))
        self.prev_cursor = (nx, ny)
        return nx, ny

    def _handle_click(self, gesture: HandGesture):
        now = time.time()
        if now - self.state.last_gesture_time < self.state.click_cooldown:
            return
        if gesture.name == "PINCH_RIGHT_CLICK":
            pyautogui.click(button="right")
            self.state.last_gesture_time = now
            self.state.last_gesture = gesture.name

    def _handle_key_press(self, gesture_name: str):
        now = time.time()
        if now - self.state.last_key_time < self.state.key_cooldown:
            return
        if self.state.last_key_gesture == gesture_name:
            return

        key_map = {
            "PINCH_SPACE": "space",
            "PINCH_ENTER": "enter",
            "PINCH_ESC": "esc",
        }
        key = key_map.get(gesture_name)
        if not key:
            return
        pyautogui.press(key)
        self.state.last_key_time = now
        self.state.last_key_gesture = gesture_name

    def _update_pinch_drag_and_click(self, hand_lms):
        """
        Thumb-index pinch:
        - quick pinch (release before drag_hold_time): left click
        - hold pinch past drag_hold_time: drag (mouseDown) until release
        """
        now = time.time()
        thumb = hand_lms[4]
        index = hand_lms[8]
        d = float(np.hypot(thumb.x - index.x, thumb.y - index.y))
        pinch_threshold = 0.045
        is_pinching = d < pinch_threshold

        # rising edge
        if is_pinching and not self.state.pinch_index_active:
            self.state.pinch_index_active = True
            self.state.pinch_index_start_time = now

        # hold -> start drag
        if is_pinching and self.state.pinch_index_active and not self.state.dragging:
            if now - self.state.pinch_index_start_time >= self.state.drag_hold_time:
                pyautogui.mouseDown()
                self.state.dragging = True

        # falling edge
        if not is_pinching and self.state.pinch_index_active:
            duration = now - self.state.pinch_index_start_time
            self.state.pinch_index_active = False
            self.state.pinch_index_start_time = 0.0

            if self.state.dragging:
                pyautogui.mouseUp()
                self.state.dragging = False
            else:
                # Treat as a click if it was a quick pinch
                if duration < self.state.drag_hold_time and now - self.state.last_gesture_time >= self.state.click_cooldown:
                    pyautogui.click()
                    self.state.last_gesture_time = now
                    self.state.last_gesture = "PINCH_LEFT_CLICK"

    def _update_scroll(self, gesture_name: str, hand_lms):
        if gesture_name != "OPEN_PALM":
            self.state.last_palm_y = None
            return

        now = time.time()
        if now - self.state.last_scroll_time < self.state.scroll_cooldown:
            return

        wrist = hand_lms[0]
        palm_y = float(wrist.y)
        if self.state.last_palm_y is None:
            self.state.last_palm_y = palm_y
            return

        dy = palm_y - self.state.last_palm_y
        self.state.last_palm_y = palm_y

        # deadzone to avoid micro-scroll
        if abs(dy) < 0.01:
            return

        # dy > 0 means hand moved down (image coords); scroll down should be negative in pyautogui
        scroll_amount = int(np.clip(-dy * 1800, -600, 600))
        if scroll_amount != 0:
            pyautogui.scroll(scroll_amount)
            self.state.last_scroll_time = now

    def run(self):
        if not self.cap.isOpened():
            print("Could not open camera. Make sure a webcam is connected.", flush=True)
            return

        print("Starting hand gesture control.", flush=True)
        print("Controls:", flush=True)
        print("- Index finger up alone: move mouse cursor", flush=True)
        print("- Thumb-index quick pinch: left click", flush=True)
        print("- Thumb-index pinch hold: click-and-drag", flush=True)
        print("- Index+middle close together: right click", flush=True)
        print("- Thumb-middle pinch: press SPACE", flush=True)
        print("- Thumb-ring pinch: press ENTER", flush=True)
        print("- Thumb-pinky pinch: press ESC", flush=True)
        print("- Open palm + move up/down: scroll", flush=True)
        print("- Open palm: neutral", flush=True)
        print("- Fist: stop/reset", flush=True)
        print("Press 'q' to quit.", flush=True)

        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read frame from camera.", flush=True)
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w, _ = frame.shape
            gesture: Optional[HandGesture] = None

            if results.hand_landmarks:
                hand_lms = results.hand_landmarks[0]
                draw_hand_landmarks(frame, hand_lms)

                # Collect normalized coordinates (x, y) for gesture recognizer
                landmarks_norm = [(lm.x, lm.y) for lm in hand_lms]
                gesture = classify_gesture(landmarks_norm)

                # Cursor control using index fingertip (id 8)
                index_tip = hand_lms[8]
                cursor_x = int(index_tip.x * self.screen_w)
                cursor_y = int(index_tip.y * self.screen_h)
                cursor_x, cursor_y = self._smooth_cursor(cursor_x, cursor_y)

                # Always update pinch/drag state if we have a hand
                self._update_pinch_drag_and_click(hand_lms)

                if gesture.name == "MOVE" or self.state.dragging:
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0)

                if gesture.name in {"PINCH_RIGHT_CLICK"}:
                    self._handle_click(gesture)

                if gesture.name in {"PINCH_SPACE", "PINCH_ENTER", "PINCH_ESC"}:
                    self._handle_key_press(gesture.name)
                else:
                    self.state.last_key_gesture = None

                self._update_scroll(gesture.name, hand_lms)

                cv2.putText(
                    frame,
                    f"Gesture: {gesture.name}" + ("  [DRAG]" if self.state.dragging else ""),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Hand Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    controller = HandGestureController()
    controller.run()


if __name__ == "__main__":
    main()

