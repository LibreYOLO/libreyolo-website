"""Check src/data/docs/onnx-io/tinyformer.json against LibreYOLO predictions.

Usage: python scripts/onnx-io/check_tinyformer.py <work_dir>

The input is built by _common.preprocess from the entry's input block
(stretch, pillow-bilinear). The script also proves the resize notes: a
fixed-point port matches Pillow bit for bit, the float weights of _common are
a level or two off at most, and OpenCV's resizes change the detections.
"""

import contextlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

from libreyolo import SAMPLE_IMAGE, LibreYOLO  # noqa: E402  (export + reference only)
from PIL import Image  # noqa: E402  (only to prove the resize notes, never to build an input)

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
WORK.mkdir(parents=True, exist_ok=True)
CONF, MAX_DET = 0.25, 300
WEIGHTS = "LibreTinyFormers.pt"

variant = c.load_variant("tinyformer", "tinyformer")
inp = variant["input"]
exports = {e["when"]: e for e in variant["tasks"]["detect"]}
assert set(exports) == {"default"}, "nms=True is documented as rejected"
results = []


def decode(logits: np.ndarray, boxes: np.ndarray, orig_w: int, orig_h: int) -> dict:
    """The entry's postprocess. logits [queries, nc] raw, boxes [queries, 4]
    normalized cx, cy, w, h. No NMS, no clipping."""
    nc = logits.shape[1]
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    flat = prob.reshape(-1)
    top = np.argsort(-flat, kind="stable")[: min(MAX_DET, flat.size)]
    scores, query, cls = flat[top], top // nc, top % nc
    cx, cy, w, h = boxes[query].astype(np.float64).T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    xyxy[:, [0, 2]] *= orig_w
    xyxy[:, [1, 3]] *= orig_h
    keep = scores > CONF
    return {"boxes": xyxy[keep], "scores": scores[keep], "classes": cls[keep]}


def run(sess, x: np.ndarray) -> dict:
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, {inp["name"]: x})))


def detections(sess, x: np.ndarray, orig_w: int, orig_h: int, i: int = 0) -> dict:
    outs = run(sess, x)
    return decode(outs["pred_logits"][i], outs["pred_boxes"][i], orig_w, orig_h)


def expect_failure(label: str, fn, error_type, text: str) -> None:
    try:
        fn()
    except error_type as exc:
        ok = text in str(exc)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: raised {type(exc).__name__}: {str(exc)[:110]}")
        results.append(ok)
        return
    print(f"[FAIL] {label}: did not raise")
    results.append(False)


def pillow_fixed_point(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Pillow's own 8-bit path: 22-bit integer weights, width pass then height."""
    def axis_pass(src: np.ndarray, out_size: int) -> np.ndarray:
        q = np.floor(c.pillow_bilinear_weights(src.shape[0], out_size) * (1 << 22) + 0.5).astype(np.int64)
        acc = (1 << 21) + np.tensordot(q, src.astype(np.int64), axes=(1, 0))
        return np.clip(acc >> 22, 0, 255).astype(np.uint8)

    if img.shape[1] != out_w:
        img = np.moveaxis(axis_pass(np.moveaxis(img, 1, 0), out_w), 0, 1)
    if img.shape[0] != out_h:
        img = axis_pass(img, out_h)
    return img


image = cv2.imread(str(SAMPLE_IMAGE))
oh, ow = image.shape[:2]
flipped = cv2.flip(image, 1)

for imgsz in (640, 512):
    ref = c.reference_detections(LibreYOLO(WEIGHTS).predict(SAMPLE_IMAGE, imgsz=imgsz, conf=CONF))
    export = exports["default"]
    path = WORK / f"tinyformer_s_{imgsz}.onnx"
    LibreYOLO(WEIGHTS).export(format="onnx", imgsz=imgsz, output_path=str(path))
    meta = c.onnx_metadata(path)
    sess = c.session(path)
    assert sess.get_inputs()[0].name == inp["name"]
    assert [o.name for o in sess.get_outputs()] == [o["name"] for o in export["outputs"]]
    H, W = c.input_hw(sess, meta)
    assert (H, W) == (imgsz, imgsz), (H, W)
    x, _ = c.preprocess(image, inp, H, W, meta)
    outs = run(sess, x)
    assert outs["pred_logits"].shape == (1, 300, int(meta["nc"])), outs["pred_logits"].shape
    assert outs["pred_boxes"].shape == (1, 300, 4), outs["pred_boxes"].shape
    ours = decode(outs["pred_logits"][0], outs["pred_boxes"][0], ow, oh)
    results.append(c.compare_detections(ours, ref, label=f"tinyformer detect default imgsz={imgsz} input={H}x{W}"))

    if imgsz == 640:
        # "Pillow computes in fixed point": a fixed-point port is bit-exact,
        # and _common's float weights are a level or two off at most.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = np.asarray(Image.fromarray(rgb).resize((W, H), Image.Resampling.BILINEAR))
        exact = np.array_equal(pillow_fixed_point(rgb, W, H), pil)
        print(f"[{'PASS' if exact else 'FAIL'}] fixed-point resize equals Pillow Image.resize(BILINEAR) bit for bit")
        results.append(exact)
        diff = np.abs(c.pillow_bilinear_resize(rgb, W, H).astype(int) - pil.astype(int))
        ok = diff.max() <= 2
        print(f"[{'PASS' if ok else 'FAIL'}] float-weight resize vs Pillow: max level diff={diff.max()}, "
              f"values differing={int((diff > 0).sum())} of {diff.size}")
        results.append(ok)

        # OpenCV resizes are documented as NOT matching: they must miss the bar.
        for name in ("bilinear", "area", "bicubic"):
            xo, _ = c.preprocess(image, {**inp, "interpolation": name}, H, W, meta)
            line = io.StringIO()
            with contextlib.redirect_stdout(line):
                matched = c.compare_detections(detections(sess, xo, ow, oh), ref)
            numbers = line.getvalue().split(": ", 1)[1].strip()
            print(f"[{'FAIL' if matched else 'PASS'}] cv2.resize {name} misses the bar, as documented: {numbers}")
            results.append(not matched)

        # "Batch size 1 only": two stacked images fail at run time.
        xb, _ = c.preprocess(flipped, inp, H, W, meta)
        expect_failure("default export with 2 images stacked (batch 2) fails",
                       lambda: run(sess, np.concatenate([x, xb])), Exception, "GatherElements")

        # LibreYOLO's own ONNX backend against the .pt.
        onnx_ref = c.reference_detections(LibreYOLO(str(path)).predict(SAMPLE_IMAGE, conf=CONF))
        results.append(c.compare_detections(onnx_ref, ref, label="LibreYOLO(onnx).predict vs .pt"))
    del sess
    path.unlink()

expect_failure("non-square imgsz=(384, 640) rejected (Square input only)",
               lambda: LibreYOLO(WEIGHTS).export(format="onnx", imgsz=(384, 640), output_path=str(WORK / "x.onnx")),
               NotImplementedError, "Rectangular imgsz export is not supported for tinyformer")
expect_failure("imgsz=500 rejected (multiple of 32)",
               lambda: LibreYOLO(WEIGHTS).export(format="onnx", imgsz=500, output_path=str(WORK / "x.onnx")),
               ValueError, "divisible by 32")
expect_failure("nms=True rejected",
               lambda: LibreYOLO(WEIGHTS).export(format="onnx", nms=True, output_path=str(WORK / "x.onnx")),
               NotImplementedError, "Embedded NMS ONNX export currently supports YOLO9")
(WORK / "x.onnx").unlink(missing_ok=True)

c.finish(results)
