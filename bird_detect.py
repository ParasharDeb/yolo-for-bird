"""Real-time bird detection from a laptop webcam using YOLOv8.

Runs the COCO-pretrained YOLOv8 model and keeps only the "bird" class (id 14),
so everything else in frame is ignored.

CPU inference is slower than the camera, so three stages run concurrently: a
capture thread keeps only the newest frame, a detector thread chews on whichever
frame is newest whenever it is free, and the main thread draws the most recent
boxes over the live feed. The preview therefore stays smooth at camera framerate
while the boxes refresh at whatever rate the CPU manages.

Keys while the window is focused:
    q / Esc   quit
    s         save the current frame to captures/
    c         cycle the confidence threshold (0.25 / 0.40 / 0.55)
"""

import argparse
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BIRD_CLASS_ID = 14  # "bird" in the COCO dataset
CONF_STEPS = (0.25, 0.40, 0.55)
BOX_COLOR = (0, 215, 255)  # BGR: amber
TEXT_COLOR = (20, 20, 20)
STALE_AFTER = 1.0  # seconds before boxes are considered too old to draw


def parse_args():
    p = argparse.ArgumentParser(description="Real-time bird detection via webcam.")
    p.add_argument("--camera", type=int, default=0,
                   help="Webcam index (0 is usually the built-in camera).")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLOv8 weights. n/s/m/l/x trade speed for accuracy.")
    p.add_argument("--imgsz", type=int, default=480,
                   help="Inference size. Lower is faster; 320-640 is a sane range.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Starting confidence threshold.")
    p.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    p.add_argument("--height", type=int, default=720, help="Requested capture height.")
    p.add_argument("--no-mirror", action="store_true",
                   help="Do not flip the preview horizontally.")
    return p.parse_args()


def open_camera(index, width, height):
    """Open the webcam, preferring DirectShow on Windows for a fast startup."""
    backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                if sys.platform == "win32" else [cv2.CAP_ANY])
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low
            return cap
        cap.release()
    return None


class LatestFrame:
    """A one-slot mailbox: writers overwrite, readers take the newest frame.

    Dropping stale frames is what keeps latency flat - a queue would let the
    slow consumer fall further behind the camera with every frame.
    """

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._new = threading.Event()

    def put(self, frame):
        with self._lock:
            self._frame = frame
        self._new.set()

    def get(self, timeout=None):
        """Block until a fresh frame arrives, then return it (None on timeout)."""
        if not self._new.wait(timeout):
            return None
        with self._lock:
            self._new.clear()
            return self._frame

    def peek(self):
        """Return the current frame without waiting, fresh or not."""
        with self._lock:
            return self._frame


def capture_loop(cap, mailbox, mirror, stop):
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            stop.set()
            break
        if mirror:
            frame = cv2.flip(frame, 1)  # mirror so the preview feels natural
        mailbox.put(frame)


class Detector(threading.Thread):
    """Detects birds on the newest available frame, in a loop, until stopped."""

    def __init__(self, model, mailbox, imgsz, conf, stop):
        super().__init__(daemon=True)
        self.model = model
        self.mailbox = mailbox
        self.imgsz = imgsz
        self.conf = conf
        self.stop_event = stop
        self.boxes = []  # (x1, y1, x2, y2, confidence) tuples
        self.updated_at = 0.0
        self.infer_times = deque(maxlen=15)
        self._lock = threading.Lock()

    def snapshot(self):
        """Return (boxes, timestamp, detections-per-second) for the drawing thread."""
        with self._lock:
            fps = (len(self.infer_times) / sum(self.infer_times)
                   if self.infer_times else 0.0)
            return list(self.boxes), self.updated_at, fps

    def run(self):
        while not self.stop_event.is_set():
            frame = self.mailbox.get(timeout=0.5)
            if frame is None:
                continue
            start = time.perf_counter()
            try:
                result = self.model.predict(
                    frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    classes=[BIRD_CLASS_ID],  # YOLO discards every other class
                    verbose=False,
                )[0]
            except Exception as exc:  # keep the preview alive if one frame upsets the model
                print(f"Detection failed: {exc}")
                continue
            elapsed = time.perf_counter() - start
            found = [(*(int(v) for v in b.xyxy[0]), float(b.conf[0]))
                     for b in result.boxes]
            with self._lock:
                self.boxes = found
                self.updated_at = time.time()
                self.infer_times.append(elapsed)


def draw_label(frame, text, x, y):
    """Draw text on a filled background so it stays readable over any scene."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(y - th - base - 4, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + base + 4), BOX_COLOR, -1)
    cv2.putText(frame, text, (x + 4, top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)


def draw_hud(frame, view_fps, det_fps, count, conf):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 38), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    status = (f"view {view_fps:4.1f} fps | detect {det_fps:4.1f} fps | "
              f"birds: {count} | conf: {conf:.2f} | [q]uit [s]ave [c]onf")
    cv2.putText(frame, status, (10, h - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    print(f"Loading {args.model} ...")
    model = YOLO(args.model)  # downloads the weights once, then caches them
    # The first inference allocates buffers and runs ~40x slower than the rest;
    # pay that cost here rather than on the first live frame.
    model.predict(np.zeros((args.height, args.width, 3), np.uint8),
                  imgsz=args.imgsz, classes=[BIRD_CLASS_ID], verbose=False)

    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        sys.exit(f"Could not open camera {args.camera}. "
                 "Close other apps using it, or try --camera 1.")

    captures_dir = Path(__file__).parent / "captures"
    conf_idx = min(range(len(CONF_STEPS)), key=lambda i: abs(CONF_STEPS[i] - args.conf))
    conf = args.conf

    stop = threading.Event()
    mailbox = LatestFrame()
    grabber = threading.Thread(target=capture_loop,
                               args=(cap, mailbox, not args.no_mirror, stop),
                               daemon=True)
    detector = Detector(model, mailbox, args.imgsz, conf, stop)
    grabber.start()
    detector.start()

    window = "Bird Detector - YOLOv8"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    view_times = deque(maxlen=30)
    last_shown = time.perf_counter()

    print("Running. Press 'q' in the window to quit.")
    try:
        while not stop.is_set():
            frame = mailbox.peek()
            if frame is None:  # camera has not delivered anything yet
                if cv2.waitKey(10) & 0xFF == 27:
                    break
                continue
            frame = frame.copy()  # the capture thread may swap the buffer under us

            boxes, updated_at, det_fps = detector.snapshot()
            fresh = time.time() - updated_at < STALE_AFTER
            if fresh:
                for x1, y1, x2, y2, score in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                    draw_label(frame, f"bird {score:.2f}", x1, y1)

            now = time.perf_counter()
            view_times.append(now - last_shown)
            last_shown = now
            elapsed = sum(view_times)
            view_fps = len(view_times) / elapsed if elapsed else 0.0
            draw_hud(frame, view_fps, det_fps, len(boxes) if fresh else 0, conf)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                captures_dir.mkdir(exist_ok=True)
                name = datetime.now().strftime("bird_%Y%m%d_%H%M%S_%f.jpg")
                cv2.imwrite(str(captures_dir / name), frame)
                print(f"Saved {captures_dir / name}")
            if key == ord("c"):
                conf_idx = (conf_idx + 1) % len(CONF_STEPS)
                conf = CONF_STEPS[conf_idx]
                detector.conf = conf
                print(f"Confidence threshold -> {conf:.2f}")

            # The window manager reports 0 once the user clicks the X button.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        grabber.join(timeout=2)
        detector.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
