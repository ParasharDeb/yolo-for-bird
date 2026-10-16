"""Real-time bird detection with YOLOv8 and ESP32 USB Serial control.

Sends BLINK once when a bird appears, and DONT_BLINK once after no bird
has been detected for BIRD_LOST_SECONDS. It does not send a command every frame.

Keys:
    q / Esc   quit
    s         save the current frame to captures/
    c         cycle confidence threshold (0.25 / 0.40 / 0.55)
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
import serial
from ultralytics import YOLO

BIRD_CLASS_ID = 14
CONF_STEPS = (0.25, 0.40, 0.55)
BOX_COLOR = (0, 215, 255)
TEXT_COLOR = (20, 20, 20)
STALE_AFTER = 1.0

ESP32_PORT = "COM7"
ESP32_BAUD = 115200
BIRD_LOST_SECONDS = 1.5  # Avoid stopping on a single missed inference


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 bird detection with ESP32 LED control.")
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
    p.add_argument("--no-esp32", action="store_true",
                   help="Run YOLO without connecting to the ESP32.")
    p.add_argument("--esp32-port", default=ESP32_PORT,
                   help="ESP32 serial port (default: COM7).")
    return p.parse_args()


def open_camera(index, width, height):
    """Open the webcam, preferring DirectShow on Windows."""
    backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                if sys.platform == "win32" else [cv2.CAP_ANY])
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    return None


class LatestFrame:
    """One-slot mailbox that keeps the newest camera frame."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._new = threading.Event()

    def put(self, frame):
        with self._lock:
            self._frame = frame
        self._new.set()

    def get(self, timeout=None):
        if not self._new.wait(timeout):
            return None
        with self._lock:
            self._new.clear()
            return self._frame

    def peek(self):
        with self._lock:
            return self._frame


def capture_loop(cap, mailbox, mirror, stop):
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            stop.set()
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        mailbox.put(frame)


class Detector(threading.Thread):
    """Detect birds on the newest available frame until stopped."""

    def __init__(self, model, mailbox, imgsz, conf, stop):
        super().__init__(daemon=True)
        self.model = model
        self.mailbox = mailbox
        self.imgsz = imgsz
        self.conf = conf
        self.stop_event = stop
        self.boxes = []
        self.updated_at = 0.0
        self.infer_times = deque(maxlen=15)
        self._lock = threading.Lock()

    def snapshot(self):
        with self._lock:
            total = sum(self.infer_times)
            fps = len(self.infer_times) / total if total > 0 else 0.0
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
                    classes=[BIRD_CLASS_ID],
                    verbose=False,
                )[0]
            except Exception as exc:
                print(f"Detection failed: {exc}")
                continue

            elapsed = time.perf_counter() - start
            found = [
                (*(int(v) for v in box.xyxy[0]), float(box.conf[0]))
                for box in result.boxes
            ]
            with self._lock:
                self.boxes = found
                self.updated_at = time.time()
                self.infer_times.append(elapsed)


def draw_label(frame, text, x, y):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(y - th - base - 4, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + base + 4), BOX_COLOR, -1)
    cv2.putText(frame, text, (x + 4, top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)


def draw_hud(frame, view_fps, det_fps, count, conf, esp_status, bird_status):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 38), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    status = (
        f"view {view_fps:4.1f} fps | detect {det_fps:4.1f} fps | "
        f"birds: {count} | conf: {conf:.2f} | bird: {bird_status} | "
        f"ESP32: {esp_status} | [q]uit [s]ave [c]onf"
    )
    cv2.putText(frame, status, (10, h - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)


def send_command(esp32, command):
    """Send a single state-change command; return True on success."""
    if esp32 is None or not esp32.is_open:
        return False
    try:
        esp32.write((command + "\n").encode("ascii"))
        esp32.flush()
        print(f"Sent {command} to ESP32.")
        return True
    except serial.SerialException as exc:
        print(f"ESP32 communication failed while sending {command}: {exc}")
        return False


def main():
    args = parse_args()

    esp32 = None
    if not args.no_esp32:
        try:
            esp32 = serial.Serial(args.esp32_port, ESP32_BAUD, timeout=1)
            time.sleep(2)  # Opening the serial port can reset the ESP32.
            print(f"Connected to ESP32 on {args.esp32_port}")
        except serial.SerialException as exc:
            print(f"Could not connect to ESP32: {exc}")
            print("YOLO will continue without ESP32 triggering.")
    else:
        print("ESP32 connection disabled by --no-esp32.")

    cap = None
    stop = threading.Event()
    grabber = None
    detector = None
    bird_command_active = False
    try:
        print("Loading YOLO model...")
        model = YOLO(args.model)
        model.predict(
            np.zeros((args.height, args.width, 3), dtype=np.uint8),
            imgsz=args.imgsz,
            classes=[BIRD_CLASS_ID],
            verbose=False,
        )

        cap = open_camera(args.camera, args.width, args.height)
        if cap is None:
            sys.exit(
                f"Could not open camera {args.camera}. Close other apps using it, "
                "or try --camera 1."
            )

        captures_dir = Path(__file__).parent / "captures"
        conf_idx = min(range(len(CONF_STEPS)), key=lambda i: abs(CONF_STEPS[i] - args.conf))
        conf = args.conf

        mailbox = LatestFrame()
        grabber = threading.Thread(
            target=capture_loop,
            args=(cap, mailbox, not args.no_mirror, stop),
            daemon=True,
        )
        detector = Detector(model, mailbox, args.imgsz, conf, stop)
        grabber.start()
        detector.start()

        window = "Bird Detector - YOLOv8"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        view_times = deque(maxlen=30)
        last_shown = time.perf_counter()

        last_bird_seen = 0.0
        bird_status = "not detected"
        esp_status = "connected" if esp32 and esp32.is_open else "offline"

        print("Running. Press 'q' or Esc in the window to quit.")
        while not stop.is_set():
            frame = mailbox.peek()
            if frame is None:
                if cv2.waitKey(10) & 0xFF == 27:
                    break
                continue
            frame = frame.copy()

            boxes, updated_at, det_fps = detector.snapshot()
            now = time.time()
            fresh = updated_at > 0 and (now - updated_at) < STALE_AFTER
            bird_detected = fresh and len(boxes) > 0

            if bird_detected:
                last_bird_seen = now
                if not bird_command_active:
                    if send_command(esp32, "BLINK"):
                        esp_status = "connected"
                    bird_command_active = True
                    bird_status = "detected"
            elif bird_command_active and (now - last_bird_seen >= BIRD_LOST_SECONDS):
                # Send DONT_BLINK only once when the bird-presence state changes.
                if send_command(esp32, "DONT_BLINK"):
                    esp_status = "connected"
                bird_command_active = False
                bird_status = "not detected"

            if fresh:
                for x1, y1, x2, y2, score in boxes:
                    x1, y1 = max(0, x1), max(0, y1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                    draw_label(frame, f"bird {score:.2f}", x1, y1)

            now_perf = time.perf_counter()
            view_times.append(now_perf - last_shown)
            last_shown = now_perf
            elapsed = sum(view_times)
            view_fps = len(view_times) / elapsed if elapsed else 0.0
            draw_hud(
                frame, view_fps, det_fps, len(boxes) if fresh else 0,
                conf, esp_status, bird_status
            )
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                captures_dir.mkdir(exist_ok=True)
                name = datetime.now().strftime("bird_%Y%m%d_%H%M%S_%f.jpg")
                path = captures_dir / name
                cv2.imwrite(str(path), frame)
                print(f"Saved {path}")
            if key == ord("c"):
                conf_idx = (conf_idx + 1) % len(CONF_STEPS)
                conf = CONF_STEPS[conf_idx]
                detector.conf = conf
                print(f"Confidence threshold -> {conf:.2f}")

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Ensure LED is switched off when the Python app exits.
        if bird_command_active:
            send_command(esp32, "DONT_BLINK")
        stop.set()
        if grabber is not None:
            grabber.join(timeout=2)
        if detector is not None:
            detector.join(timeout=2)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if esp32 is not None and esp32.is_open:
            esp32.close()


if __name__ == "__main__":
    main()
