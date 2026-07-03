"""
score_images.py

Runs the Dart Sense pipeline over a folder of still images instead of a live
video stream. For each image it does exactly what the real-time app does per
frame:

    load image -> YOLO detect -> separate darts/calibration points
                -> find homography -> transform darts to board plane
                -> score each dart -> sum the visit

Optionally writes annotated copies (board outline + dart markers + labels) so
you can eyeball the results, mirroring the GUI's 'boardplane' display.

Usage:
    python score_images.py
    python score_images.py --images data/darts/images/small_sample --annotate
    python score_images.py --weights weights.pt --conf 0.5 --display-size 720
"""

import os
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

from get_scores import GetScores  # reuse the app's scoring logic unchanged


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')


def centre_square_crop(img):
    """Centre-crop to the largest square. Returns (cropped, crop_start, size).

    The live app crops a fixed 1200x1600 stream; here we crop each image to its
    own centred square so calibration points near the edges aren't lost on
    samples of differing aspect ratios. For square inputs this is a no-op.
    """
    h, w = img.shape[:2]
    size = min(h, w)
    x0 = (w - size) // 2
    y0 = (h - size) // 2
    crop = img[y0:y0 + size, x0:x0 + size]
    return crop, np.array([x0, y0]), size


def annotate(crop, predict, H_matrix, dart_coords_board, darts, display_size):
    """Produce a board-plane image with outline, dart markers and labels."""
    warped = cv2.warpPerspective(crop, H_matrix, (crop.shape[1], crop.shape[0]))
    warped = cv2.resize(warped, (display_size, display_size))

    # board outline (segment lines + scoring rings)
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

    # dart markers + labels
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

    crop, crop_start, crop_size = centre_square_crop(img)

    # YOLO inference on the cropped square
    result = model(crop, conf=conf, verbose=False)[0]

    # separate darts vs calibration points (app's own logic)
    calibration_coords, dart_coords = predict.process_yolo_output(result)

    # need at least 4 calibration points for a homography
    detected = np.count_nonzero(np.all(calibration_coords >= 0, axis=1))
    if detected < 4:
        print(f"  only {detected}/4 calibration points found - skipping")
        return {'image': os.path.basename(path), 'darts': [], 'score': None,
                'note': 'insufficient calibration points'}

    # homography: image plane -> board plane, then transform dart coords
    H_matrix = predict.find_homography(calibration_coords, crop_size)
    board_coords = predict.transform_to_boardplane(H_matrix[0], dart_coords, crop_size)

    darts, score = predict.score(np.array(board_coords))

    if annotate_dir is not None and len(board_coords) > 0:
        os.makedirs(annotate_dir, exist_ok=True)
        out = annotate(crop, predict, H_matrix[0], board_coords, darts, display_size)
        out_path = os.path.join(annotate_dir, os.path.basename(path))
        cv2.imwrite(out_path, out)

    return {'image': os.path.basename(path), 'darts': darts, 'score': score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='data/darts/images/small_sample',
                    help='folder of images to score')
    ap.add_argument('--weights', default='weights.pt')
    ap.add_argument('--conf', type=float, default=0.5,
                    help='YOLO confidence threshold')
    ap.add_argument('--annotate', action='store_true',
                    help='write annotated board-plane images')
    ap.add_argument('--annotate-dir', default='predictions')
    ap.add_argument('--display-size', type=int, default=720)
    args = ap.parse_args()

    model = YOLO(args.weights)
    predict = GetScores(args.weights)

    files = sorted(f for f in os.listdir(args.images) if f.endswith(IMG_EXTS))
    if not files:
        print(f"No images found in {args.images}")
        return

    print(f"Scoring {len(files)} image(s) from {args.images}\n")
    results = []
    for f in files:
        print(f"{f}")
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
