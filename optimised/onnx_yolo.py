"""
onnx_yolo.py

A minimal, dependency-light YOLOv8 detector that runs on onnxruntime instead
of the full ultralytics/PyTorch stack. It replicates just enough of the
ultralytics `Results` interface (`.boxes.cls`, `.boxes.xywhn`, `.boxes.conf`)
that `get_scores.GetScores.process_yolo_output()` works completely unchanged.

Why this exists: PyTorch + ultralytics need ~1-2 GB of installed packages and
hundreds of MB of RAM at runtime, which rules out low-end hardware (small
SBCs, old laptops, and anything approaching microcontroller territory).
onnxruntime's CPU package is a few tens of MB, starts faster, and uses far
less memory and CPU for the same forward pass.

No NMS/decoding library is used either (no torchvision) - both are re-implemented
here with plain numpy + cv2.dnn.NMSBoxes, which every OpenCV build ships with.
"""

import numpy as np
import cv2
import onnxruntime as ort


class SimpleBoxes:
    """Duck-types ultralytics' `Boxes` just enough for GetScores.process_yolo_output."""

    def __init__(self, cls, xywhn, conf):
        self.cls = cls        # (N,)  float class id per detection
        self.xywhn = xywhn    # (N,4) normalized [cx, cy, w, h] in [0, 1]
        self.conf = conf      # (N,)  confidence per detection


class SimpleResult:
    def __init__(self, boxes):
        self.boxes = boxes


EMPTY_BOXES = SimpleBoxes(np.zeros(0), np.zeros((0, 4)), np.zeros(0))


class OnnxYolo:
    """Runs a YOLOv8-detect ONNX export and returns an ultralytics-shaped result."""

    def __init__(self, model_path, imgsz=320, num_classes=7, providers=None):
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.session = ort.InferenceSession(
            model_path,
            providers=providers or ['CPUExecutionProvider'],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, bgr):
        # crop fed in is already square (see centre_square_crop), so a plain
        # resize is equivalent to ultralytics' letterbox (no padding needed).
        resized = cv2.resize(bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(chw)

    def __call__(self, bgr, conf=0.5, iou=0.7):
        inp = self._preprocess(bgr)
        raw = self.session.run(None, {self.input_name: inp})[0]  # (1, 4+nc, N)
        preds = raw[0].T  # (N, 4+nc)

        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:4 + self.num_classes]
        cls_id = np.argmax(scores, axis=1)
        cls_conf = scores[np.arange(len(scores)), cls_id]

        keep = cls_conf >= conf
        boxes_xywh, cls_id, cls_conf = boxes_xywh[keep], cls_id[keep], cls_conf[keep]
        if len(boxes_xywh) == 0:
            return SimpleResult(EMPTY_BOXES)

        x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        rects = np.stack([x - w / 2, y - h / 2, w, h], axis=1)  # x,y,w,h for cv2.dnn.NMSBoxes

        keep_idx = []
        for c in np.unique(cls_id):
            mask = cls_id == c
            idxs = np.nonzero(mask)[0]
            nms_idx = cv2.dnn.NMSBoxes(
                rects[idxs].tolist(), cls_conf[idxs].tolist(), conf, iou
            )
            if len(nms_idx) > 0:
                keep_idx.extend(idxs[np.array(nms_idx).flatten()])

        if not keep_idx:
            return SimpleResult(EMPTY_BOXES)

        keep_idx = np.array(keep_idx)
        xywhn = boxes_xywh[keep_idx] / self.imgsz
        out = SimpleBoxes(
            cls=cls_id[keep_idx].astype(np.float32),
            xywhn=xywhn.astype(np.float32),
            conf=cls_conf[keep_idx].astype(np.float32),
        )
        return SimpleResult(out)
