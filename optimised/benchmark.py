"""
benchmark.py

Reproduces the accuracy + speed numbers quoted in README.md:
  1. Accuracy: runs the original PyTorch model (ground truth) and each ONNX
     variant over the bundled sample images, and reports how often the final
     visit score matches exactly.
  2. Speed: single-threaded (`intra_op_num_threads=1`, to approximate a
     constrained single-core device) latency per model.

Needs both the export requirements (torch/ultralytics, for ground truth) and
the runtime requirements (onnxruntime). See ../requirements-export.txt and
requirements.txt.

Usage:
    python benchmark.py
    python benchmark.py --models models/dart_yolo_800.onnx:800 models/dart_yolo_640.onnx:640
"""

import argparse
import os
import time

import cv2
import numpy as np

from get_scores import GetScores
from onnx_yolo import OnnxYolo

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')


def centre_square_crop(img):
    h, w = img.shape[:2]
    size = min(h, w)
    x0, y0 = (w - size) // 2, (h - size) // 2
    return img[y0:y0 + size, x0:x0 + size], size


def score_result(result, predict, crop_size):
    calibration_coords, dart_coords = predict.process_yolo_output(result)
    detected = np.count_nonzero(np.all(calibration_coords >= 0, axis=1))
    if detected < 4:
        return None, None
    H = predict.find_homography(calibration_coords, crop_size)
    board = predict.transform_to_boardplane(H[0], dart_coords, crop_size)
    darts, score = predict.score(np.array(board))
    return ' '.join(darts), score


def run_accuracy(images_dir, weights_pt, model_specs, conf):
    from ultralytics import YOLO  # ground truth only

    torch_model = YOLO(weights_pt)
    predict = GetScores(weights_pt)
    detectors = {name: OnnxYolo(path, imgsz=imgsz) for name, path, imgsz in model_specs}

    files = sorted(f for f in os.listdir(images_dir) if f.endswith(IMG_EXTS))
    match = {name: 0 for name in detectors}
    total = 0

    print(f"\n{'image':30s} {'ground truth':20s}", end='')
    for name in detectors:
        print(f" {name:>16s}", end='')
    print()

    for f in files:
        img = cv2.imread(os.path.join(images_dir, f))
        if img is None:
            continue
        crop, size = centre_square_crop(img)
        total += 1

        gt_result = torch_model(crop, conf=conf, verbose=False)[0]
        gt_visit, gt_score = score_result(gt_result, predict, size)
        gt_str = f"{gt_visit}={gt_score}" if gt_visit is not None else "-"
        print(f"{f:30.30s} {gt_str:20.20s}", end='')

        for name, det in detectors.items():
            r = det(crop, conf=conf, iou=0.7)
            visit, score = score_result(r, predict, size)
            ok = (visit, score) == (gt_visit, gt_score)
            match[name] += int(ok)
            tag = 'OK' if ok else 'DIFF'
            print(f" {tag:>16s}", end='')
        print()

    print("\naccuracy (exact visit+score match vs PyTorch ground truth):")
    for name in detectors:
        print(f"  {name:20s} {match[name]}/{total}")


def run_speed(model_specs, sample_image, threads=1):
    import onnxruntime as ort

    img = cv2.imread(sample_image)
    crop, _ = centre_square_crop(img)

    print(f"\nspeed (single image, {threads} thread(s), CPUExecutionProvider):")
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads

    for name, path, imgsz in model_specs:
        sess = ort.InferenceSession(path, sess_options=so, providers=['CPUExecutionProvider'])
        inp_name = sess.get_inputs()[0].name
        resized = cv2.resize(crop, (imgsz, imgsz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])

        for _ in range(3):
            sess.run(None, {inp_name: chw})
        n = 15
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(None, {inp_name: chw})
        dt = (time.perf_counter() - t0) / n
        print(f"  {name:20s} {dt*1000:7.1f} ms/frame  ({1/dt:5.2f} FPS)  {os.path.getsize(path)/1e6:5.1f} MB")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default=os.path.join(here, '..', 'data', 'darts', 'images', 'small_sample'))
    ap.add_argument('--weights', default=os.path.join(here, '..', 'weights.pt'))
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--models', nargs='+',
                    default=[
                        f"{os.path.join(here, 'models', 'dart_yolo_800.onnx')}:800",
                        f"{os.path.join(here, 'models', 'dart_yolo_800_fp16.onnx')}:800",
                    ],
                    help='list of path:imgsz entries')
    ap.add_argument('--skip-accuracy', action='store_true',
                    help='skip the torch ground-truth comparison (no torch/ultralytics needed)')
    args = ap.parse_args()

    model_specs = []
    for spec in args.models:
        path, imgsz = spec.rsplit(':', 1)
        model_specs.append((os.path.basename(path), path, int(imgsz)))

    if not args.skip_accuracy:
        run_accuracy(args.images, args.weights, model_specs, args.conf)

    sample = sorted(os.path.join(args.images, f) for f in os.listdir(args.images) if f.endswith(IMG_EXTS))[0]
    run_speed(model_specs, sample)


if __name__ == '__main__':
    main()
