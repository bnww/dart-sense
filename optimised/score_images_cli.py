"""
score_images_cli.py (optimised)

Headless equivalent of score_images.py, but running on onnxruntime instead of
PyTorch/ultralytics. No display/tkinter required, so this is what you'd run
on a headless low-end box (Raspberry Pi, old laptop used as a server, etc.)
to sanity check the optimised pipeline before wiring it into a live app.

Usage:
    python score_images_cli.py
    python score_images_cli.py --images ../data/darts/images/small_sample --annotate
    python score_images_cli.py --model models/dart_yolo_800_fp16.onnx --imgsz 800 --conf 0.5
"""

import os
import argparse
import numpy as np
import cv2

from get_scores import GetScores
from onnx_yolo import OnnxYolo

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')


def centre_square_crop(img):
    h, w = img.shape[:2]
    size = min(h, w)
    x0, y0 = (w - size) // 2, (h - size) // 2
    return img[y0:y0 + size, x0:x0 + size], size


def annotate(crop, predict, H_matrix, dart_coords_board, darts, display_size):
    warped = cv2.warpPerspective(crop, H_matrix, (crop.shape[1], crop.shape[0]))
    warped = cv2.resize(warped, (display_size, display_size))

    radii = predict.scoring_radii * display_size
    angles = np.append(predict.segment_angles, 81)
    outer, inner = radii[-1], radii[2]
    c = display_size / 2
    for ang in angles:
        oa = outer * np.cos(np.deg2rad(ang)); oo = (outer**2 - oa**2) ** 0.5
        ia = inner * np.cos(np.deg2rad(ang)); io = (inner**2 - ia**2) ** 0.5
        if ang > 0:
            pts = [((c + oa, c + oo), (c + ia, c + io)),
                   ((c - oa, c - oo), (c - ia, c - io))]
        else:
            pts = [((c - oa, c + oo), (c - ia, c + io)),
                   ((c + oa, c - oo), (c + ia, c - io))]
        for p1, p2 in pts:
            cv2.line(warped, tuple(np.round(p1).astype(int)),
                     tuple(np.round(p2).astype(int)), (255, 0, 0), 1)
    for r in np.round(radii).astype(int):
        cv2.circle(warped, (int(c), int(c)), int(r), (255, 0, 0), 1)

    for (dx, dy), label in zip(dart_coords_board, darts):
        px, py = int(round(dx * display_size)), int(round(dy * display_size))
        cv2.circle(warped, (px, py), 5, (0, 255, 255), 2)
        cv2.putText(warped, label, (px - 10, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return warped


def process_image(path, model, predict, conf, annotate_dir=None, display_size=720):
    img = cv2.imread(path)
    if img is None:
        print(f"  could not read {path}")
        return None

    crop, crop_size = centre_square_crop(img)
    result = model(crop, conf=conf)
    calibration_coords, dart_coords = predict.process_yolo_output(result)

    detected = np.count_nonzero(np.all(calibration_coords >= 0, axis=1))
    if detected < 4:
        print(f"  only {detected}/4 calibration points found - skipping")
        return {'image': os.path.basename(path), 'darts': [], 'score': None,
                'note': 'insufficient calibration points'}

    H_matrix = predict.find_homography(calibration_coords, crop_size)
    board_coords = predict.transform_to_boardplane(H_matrix[0], dart_coords, crop_size)
    darts, score = predict.score(np.array(board_coords))

    if annotate_dir is not None and len(board_coords) > 0:
        os.makedirs(annotate_dir, exist_ok=True)
        out = annotate(crop, predict, H_matrix[0], board_coords, darts, display_size)
        cv2.imwrite(os.path.join(annotate_dir, os.path.basename(path)), out)

    return {'image': os.path.basename(path), 'darts': darts, 'score': score}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default=os.path.join(here, '..', 'data', 'darts', 'images', 'small_sample'))
    ap.add_argument('--model', default=os.path.join(here, 'models', 'dart_yolo_800_fp16.onnx'))
    ap.add_argument('--imgsz', type=int, default=800,
                    help='must match the input size the ONNX model was exported with')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--annotate', action='store_true')
    ap.add_argument('--annotate-dir', default='predictions')
    ap.add_argument('--display-size', type=int, default=720)
    args = ap.parse_args()

    model = OnnxYolo(args.model, imgsz=args.imgsz)
    predict = GetScores(args.model)

    files = sorted(f for f in os.listdir(args.images) if f.endswith(IMG_EXTS))
    if not files:
        print(f"No images found in {args.images}")
        return

    print(f"Scoring {len(files)} image(s) from {args.images} with {args.model} (imgsz={args.imgsz})\n")
    results = []
    for f in files:
        print(f)
        r = process_image(os.path.join(args.images, f), model, predict, args.conf,
                          args.annotate_dir if args.annotate else None,
                          args.display_size)
        if r is None:
            continue
        results.append(r)
        if r['score'] is not None:
            visit = ' '.join(r['darts']) if r['darts'] else '(no darts)'
            print(f"  -> {visit}  = {r['score']}\n")
        else:
            print(f"  -> {r.get('note', 'no score')}\n")

    scored = [r for r in results if r['score'] is not None]
    print("=" * 40)
    print(f"Processed {len(results)} image(s), scored {len(scored)}")
    if args.annotate:
        print(f"Annotated images written to ./{args.annotate_dir}/")


if __name__ == '__main__':
    main()
