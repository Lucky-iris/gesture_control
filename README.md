# Hand Gesture Control System

This project lets you control your laptop or desktop using simple hand gestures captured from your webcam.

Current features:

- Move mouse cursor using your index finger
- Left-click with a thumb–index quick pinch
- Click-and-drag with a thumb–index pinch hold
- Right-click with a close index–middle two‑finger gesture
- Press keyboard keys using different pinches (SPACE / ENTER / ESC)
- Scroll using open palm (move hand up/down)
- Visual feedback window with detected hand landmarks and gesture name

## 1. Install dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

On Windows, it is recommended to use a Python virtual environment (for example, with `python -m venv .venv` and `.\.venv\Scripts\activate`).

## 2. Run the controller

In the same terminal:

```bash
python hand_gesture_control.py
```

Make sure:

- Your webcam is connected and not used by other applications.
- The OpenCV preview window is focused when you want to quit (press `q`).
- First run may download the MediaPipe model to `models/hand_landmarker.task`.

If you see `Failed to read frame from camera`, close other camera apps and check Windows camera privacy permissions for Python / Desktop apps.

## 3. Gesture guide

- **Cursor move**: Keep only the index finger up; move your hand to move the mouse cursor.
- **Left click**: Quick pinch thumb + index (release quickly).
- **Drag**: Hold thumb + index pinch for ~0.5s to start drag; keep holding while moving; release to drop.
- **Right click**: Hold index and middle fingers up and bring their tips close together.
- **Press SPACE**: Pinch thumb + middle fingertip.
- **Press ENTER**: Pinch thumb + ring fingertip.
- **Press ESC**: Pinch thumb + pinky fingertip.
- **Scroll**: Show open palm and move hand up/down.
- **Open palm**: All fingers up; neutral state.
- **Fist**: All fingers down; used as a stop/reset pose.

You can tweak behavior (sensitivity, click cooldown, etc.) in `hand_gesture_control.py`.

