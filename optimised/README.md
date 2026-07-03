# Optimised Dart Sense pipeline

Same pipeline as `score_images_gui.py` in the repo root (crop -> detect ->
calibrate -> warp -> score), but re-targeted at low-end hardware. The
scoring math in `get_scores.py` is copied **unmodified** - only the model
runtime changed.

## What changed and why

The original app loads a YOLOv8n `weights.pt` checkpoint through
`ultralytics` + PyTorch. That stack is the dominant cost on constrained
hardware, independent of the neural net itself:

| | ultralytics + torch | onnxruntime (this folder) |
|---|---|---|
| cold `import` time | 1.37 s | 0.10 s (13x faster) |
| RSS after loading the model | 295 MB | 40 MB (7x less) |
| install size | ~700 MB+ (torch alone) | ~15-90 MB depending on platform wheel |

Measured on this machine with `python -X importtime` and
`resource.getrusage` - see "Reproducing these numbers" below.

So the biggest, safest win for "lower-end hardware" is swapping the
inference runtime, not touching the model weights. `onnx_yolo.py` implements
a small `OnnxYolo` class that runs the exported ONNX graph through
`onnxruntime` and decodes its raw output (box regression + per-class scores,
manual NMS via `cv2.dnn.NMSBoxes`) into the exact same shape
(`.boxes.cls` / `.boxes.xywhn` / `.boxes.conf`) that
`GetScores.process_yolo_output()` already expects. That's why
`get_scores.py` needed zero changes.

## Accuracy validation

`benchmark.py` runs both the original PyTorch model and each ONNX model over
the 10 sample images in `data/darts/images/small_sample` and checks whether
the predicted visit (darts + total score) matches exactly:

| model | input size | exact match | file size |
|---|---|---|---|
| `dart_yolo_800.onnx` (fp32) | 800x800 | **10/10** | 12.4 MB |
| `dart_yolo_800_fp16.onnx` | 800x800 | **10/10** | 6.2 MB |
| `dart_yolo_640.onnx` (not shipped, regenerate if needed) | 640x640 | 4/10 | 12.3 MB |
| `dart_yolo_320.onnx` (not shipped) | 320x320 | 1/10 | 12.1 MB |
| dynamic int8 quantized (not shipped) | 800x800 | 7/10 | 3.5 MB |

**Only the two shipped models are recommended.** This checkpoint was trained
at `imgsz=800` and is noticeably less accurate at lower input resolutions -
the small board calibration markers (corner of the 20/6/3/11 segments) stop
being detected reliably, which fails the whole homography step. If you need
a genuinely fast+accurate low-resolution model, the real fix is retraining
at that resolution (see `training/` in the repo root), not just re-exporting.

fp16 halves the file size for identical accuracy, so it's the default model
in `score_images_gui.py`/`score_images_cli.py` here - worthwhile purely for
disk/RAM-constrained boards even though it isn't faster on a CPU without
native fp16 arithmetic (see next section).

## Quantization: what actually helped

Dynamic-range int8 quantization (`onnxruntime.quantization.quantize_dynamic`)
shrinks the file 3.5x, but on this x86 CPU it was measured to be **slower**
than fp32 (207ms vs 132ms/frame at 800px, single thread) and less accurate
(7/10 vs 10/10). Dynamic quantization only rewrites MatMul-like ops with
quant/dequant wrappers; it doesn't get real int8 conv kernels, so on a CPU
without dedicated int8 SIMD it just adds overhead. It's included in
`export_model.py --int8` for completeness (e.g. if you want a smaller
download and don't care about raw speed), but it is **not** a speed
optimization here. Static/QAT quantization with a proper int8 execution
provider (ARM NEON dot-product, XNNPACK, or a microcontroller NPU) is a
different story - see the ESP32 section below.

## Speed (single CPU thread, x86 dev box)

| model | ms/frame | FPS |
|---|---|---|
| `dart_yolo_800.onnx` | 126 | 8.0 |
| `dart_yolo_800_fp16.onnx` | 136 | 7.3 |
| ultralytics/torch, `weights.pt`, 800px | ~131 | 7.6 |

Raw per-frame compute is similar to the original (same architecture, same
FLOPs) - the win here is everything *around* inference (import time, RAM,
install footprint), which is what actually gates whether the app runs at
all on something like a Raspberry Pi Zero or an old netbook, as opposed to
how many FPS it hits once it's running.

## Files

- `onnx_yolo.py` - `OnnxYolo`: onnxruntime-based drop-in replacement for
  `ultralytics.YOLO`, duck-typing just enough of the `Results` object for
  `get_scores.py` to work unchanged.
- `get_scores.py` - unmodified copy of the repo root's scoring logic (pure
  numpy/cv2, no torch dependency either way).
- `score_images_gui.py` - the same step-by-step Tkinter viewer as the
  original, pointed at the ONNX backend.
- `score_images_cli.py` - headless equivalent, for boards with no display.
- `export_model.py` - regenerates the shipped ONNX models from `weights.pt`,
  or produces other resolutions/precisions (needs torch/ultralytics - see
  `requirements-export.txt`).
- `benchmark.py` - reproduces the accuracy/speed tables above.
- `models/` - `dart_yolo_800.onnx` (fp32) and `dart_yolo_800_fp16.onnx`,
  regeneratable with `export_model.py`.

## Usage

```bash
# on the low-end device: only the lightweight runtime deps
pip install -r requirements.txt

# GUI (needs a Tk display - `apt install python3-tk` if tkinter is missing)
python score_images_gui.py

# headless
python score_images_cli.py --model models/dart_yolo_800_fp16.onnx --imgsz 800

# on a dev machine with torch/ultralytics installed:
pip install -r requirements-export.txt
python export_model.py                       # regenerate the shipped pair
python export_model.py --imgsz 640 --half     # experiment with other variants
python benchmark.py                            # reproduce the tables above
```

`score_images_gui.py` has a "Model input size" field alongside the model
file path - it must match the resolution the .onnx file was exported with,
since (unlike ultralytics) a plain ONNX graph has a fixed input shape.

## Can this run on an ESP32?

Short answer: **not this model, not as-is** - and I'd rather say that
clearly than ship something that quietly can't work.

- Neither PyTorch nor onnxruntime run on ESP32 at all (no such build target);
  on-device inference would require Espressif's **ESP-DL** or
  **TensorFlow Lite for Microcontrollers**, both C/C++ toolchains with their
  own (re-)quantization pipeline - `weights.onnx` would need converting via
  `onnx2tf` -> TFLite -> int8, not just what's in this folder.
- The model itself is ~3M parameters / 8 GFLOPs at 800x800 (still ~1.3
  GFLOPs even at a 320x320 export, which we measured above as already too
  low-resolution for the calibration step to work reliably). An ESP32-S3
  with ESP-DL's vector instructions realistically handles CNNs on the order
  of a few hundred MFLOPs at low resolution (think MobileNet-scale at
  96-128px) at a handful of FPS. This model is roughly one to two orders of
  magnitude past that, at the *one* resolution where it's actually accurate.
- RAM is the other wall: a plain ESP32 has 320-520 KB of SRAM; even an
  ESP32-S3 with 8 MB of PSRAM would need every intermediate activation
  tensor kept tiny. An 800x800x3 input alone is ~1.9 MB in fp32 - workable
  only with aggressive int8 quantization and a much smaller input
  resolution than this checkpoint was trained for.

**What would actually get dart-scoring onto ESP32-class hardware:**

1. **Don't run detection on the ESP32 at all.** Use it as a camera node
   (e.g. `esp32-cam`) that streams frames over Wi-Fi to whatever runs this
   `optimised/` pipeline - a Pi, an old laptop, a phone. This is exactly how
   the existing app already works (IP Webcam streaming to the PC running
   `GUI.py`), just with a cheaper camera. No model changes needed, and it's
   the option validated in this repo.
2. **If on-device inference is a hard requirement**, it means training a
   deliberately small detector from scratch for a small fixed input (e.g.
   96-128px, far fewer channels than YOLOv8n) using ESP-DL's/TFLite Micro's
   int8 quantization-aware training flow, and accepting materially worse
   accuracy and a low single-digit FPS. That's a separate model-design and
   data-collection project, not something an export script can produce from
   `weights.pt` - the geometry (tiny calibration-corner markers on a full
   board) is inherently hard at the resolutions an ESP32 can afford.

## Reproducing the numbers in this README

```bash
python -X importtime -c "import onnxruntime" 2>&1 | tail -1
python -X importtime -c "import ultralytics" 2>&1 | tail -1
python benchmark.py
```
