"""
export_model.py

Regenerates the ONNX models shipped in optimised/models/, and lets you
produce additional variants (different input resolution, fp16, int8) from
the original weights.pt checkpoint.

This step still needs the full ultralytics + PyTorch stack (see
../requirements-export.txt) - that's fine, it only runs once, offline, on
your dev machine. Nothing in optimised/ needs torch at *inference* time.

Usage:
    # regenerate the two models shipped by default (fp32 + fp16 @ 800)
    python export_model.py

    # export a custom resolution, e.g. for a faster/lower-accuracy variant
    python export_model.py --imgsz 640 --out models/dart_yolo_640.onnx

    # dynamic-range int8 quantization (smaller file, NOT necessarily faster
    # on x86 - see README "Quantization" section for measured numbers)
    python export_model.py --imgsz 800 --int8 --out models/dart_yolo_800_int8.onnx
"""

import argparse
import os


def export_onnx(weights, imgsz, half, out_path):
    from ultralytics import YOLO

    model = YOLO(weights)
    exported = model.export(format='onnx', imgsz=imgsz, simplify=True,
                            opset=12, dynamic=False, half=half)
    if out_path and os.path.abspath(exported) != os.path.abspath(out_path):
        os.replace(exported, out_path)
        exported = out_path
    print(f"wrote {exported}")
    return exported


def quantize_int8(src_path, dst_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(src_path, dst_path, weight_type=QuantType.QUInt8)
    print(f"wrote {dst_path}  "
         f"({os.path.getsize(src_path)/1e6:.1f} MB -> {os.path.getsize(dst_path)/1e6:.1f} MB)")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join(here, '..', 'weights.pt'))
    ap.add_argument('--imgsz', type=int, default=None,
                    help='export input resolution; omit to regenerate the default 800 fp32+fp16 pair')
    ap.add_argument('--half', action='store_true', help='export weights as fp16')
    ap.add_argument('--int8', action='store_true',
                    help='additionally produce a dynamic-range int8 quantized copy')
    ap.add_argument('--out', default=None, help='output .onnx path')
    args = ap.parse_args()

    models_dir = os.path.join(here, 'models')
    os.makedirs(models_dir, exist_ok=True)

    if args.imgsz is None:
        # default: regenerate exactly what's shipped
        fp32 = export_onnx(args.weights, 800, False,
                           os.path.join(models_dir, 'dart_yolo_800.onnx'))
        export_onnx(args.weights, 800, True,
                   os.path.join(models_dir, 'dart_yolo_800_fp16.onnx'))
        return

    out = args.out or os.path.join(models_dir, f'dart_yolo_{args.imgsz}.onnx')
    fp32_or_fp16 = export_onnx(args.weights, args.imgsz, args.half, out)

    if args.int8:
        int8_out = os.path.splitext(out)[0] + '_int8.onnx'
        quantize_int8(fp32_or_fp16, int8_out)


if __name__ == '__main__':
    main()
