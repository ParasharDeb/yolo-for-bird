# Real-time bird detection

YOLOv8 running on the laptop webcam, filtered down to the COCO `bird` class.

## Run

```
.venv\Scripts\python bird_detect.py
```

The first run downloads `yolov8n.pt` (~6 MB) into this folder and caches it.
Startup takes a few seconds: the first inference is ~40x slower than the rest,
so the app warms the model up before opening the camera.

## Keys

| Key | Action |
| --- | --- |
| `q` / `Esc` | quit |
| `s` | save the current frame (boxes included) to `captures/` |
| `c` | cycle the confidence threshold (0.25 / 0.40 / 0.55) |

## Options

```
--camera 1          pick a different webcam
--imgsz 320         smaller = faster, less accurate (default 480)
--model yolov8s.pt  more accurate, slower (n < s < m < l < x)
--conf 0.4          starting confidence threshold
--no-mirror         don't flip the preview horizontally
```

## How it stays real-time on a CPU

This machine has no NVIDIA GPU, so inference runs on the Ryzen 5 5500U at
roughly **9-14 detections/sec** at `--imgsz 480` — slower than the camera's
30 fps. Rather than let that stutter the video, three stages run concurrently:

- a **capture thread** reads the camera into a one-slot mailbox, overwriting
  any frame the detector was too slow to pick up (dropping stale frames is what
  keeps latency flat — a queue would fall further behind with every frame);
- a **detector thread** runs YOLO on whatever frame is newest whenever it is free;
- the **main thread** draws the most recent boxes over the live feed.

The preview therefore stays smooth at full camera framerate while boxes refresh
at ~10 fps. The HUD reports both rates separately. The trade-off is that boxes
are drawn on a slightly newer frame than they were computed from, so a
fast-moving bird's box lags it by ~100 ms. Boxes older than 1 second are
dropped rather than left floating.

Measured (`--imgsz`, sustained): 320 → ~10-19 fps, 480 → ~9-14 fps, 640 → ~5 fps.
The spread is thermal throttling; a laptop on battery sits at the low end.

## Notes

- Detection is limited to birds by passing `classes=[14]` to the model, so YOLO
  discards the other 79 COCO classes before they reach the drawing code.
- Verified against reference photos: house sparrow 0.94, rock dove 0.84,
  mallard pair 0.95 / 0.87, and zero false positives on a bird-free image.
- COCO's `bird` class is generic — it will not tell species apart. That needs a
  model fine-tuned on a bird dataset (e.g. CUB-200 or NABirds).
- Torch is the CPU-only build. If you later add an NVIDIA GPU, reinstall torch
  from the CUDA index and the same code will use it automatically.
