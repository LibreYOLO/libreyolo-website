"""Check src/data/docs/onnx-io/yolo9.json against LibreYOLO predictions.

Covers the yolo9, yolo9_e2e and yolo9_p2 variants.

Usage: python scripts/onnx-io/check_yolo9.py <work_dir>
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

from libreyolo import SAMPLE_IMAGE, LibreYOLO  # noqa: E402  (export + reference only)

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
WORK.mkdir(parents=True, exist_ok=True)
CONF, IOU, MAX_DET = 0.25, 0.45, 300

variant = c.load_variant("yolo9", "yolo9")
image = cv2.imread(str(SAMPLE_IMAGE))
results = []

for imgsz in (640, (384, 640)):
    model = LibreYOLO("LibreYOLO9t.pt")
    ref = c.reference_detections(model.predict(SAMPLE_IMAGE, imgsz=imgsz, conf=CONF, iou=IOU))
    for export in variant["tasks"]["detect"]:
        nms = export["when"] == "nms=True"
        path = WORK / f"yolo9t_{'nms' if nms else 'raw'}.onnx"
        LibreYOLO("LibreYOLO9t.pt").export(format="onnx", imgsz=imgsz, nms=nms, output_path=str(path))
        meta = c.onnx_metadata(path)
        sess = c.session(path)
        assert sess.get_inputs()[0].name == variant["input"]["name"]
        H, W = c.input_hw(sess, meta)
        x, geo = c.preprocess(image, variant["input"], H, W, meta)
        outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {variant["input"]["name"]: x})))

        if nms:
            rows = outs["output"][0]
            rows = rows[rows[:, 4] > CONF]
            boxes, scores, classes = rows[:, :4], rows[:, 4], rows[:, 5].astype(np.int64)
        else:
            pred = outs["output"][0].T  # [anchors, 4 + nc]
            anchor, cls = np.nonzero(pred[:, 4:] > CONF)
            boxes, scores, classes = pred[anchor, :4], pred[anchor, 4 + cls], cls
            keep = c.class_aware_nms(boxes, scores, classes, IOU, MAX_DET)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        boxes = c.to_original_xyxy(boxes, geo)
        if not nms:  # default export postprocess: drop zero-area boxes after clipping
            nonempty = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores, classes = boxes[nonempty], scores[nonempty], classes[nonempty]
        ours = {"boxes": boxes, "scores": scores, "classes": classes}
        results.append(c.compare_detections(ours, ref, label=f"yolo9 detect {export['when']} imgsz={imgsz} input={H}x{W}"))
        path.unlink()


# ---------------------------------------------------------------------------
# yolo9_e2e and yolo9_p2: default export only (nms=True is rejected).
# Each size runs at the default conf and at a low conf. At conf 0.25 the
# sample image gives few boxes, which NMS or no NMS would decode the same.
# The low conf is where the claims bite: E2E keeps overlapping boxes that NMS
# would remove and hits the 300 pair cap (420 pairs above 1e-4 at 640), P2
# needs class-aware NMS (class-agnostic NMS drops 27 of 51 boxes at 0.001).
# ---------------------------------------------------------------------------


def check_output_shape(export, out, nc, H, W, label):
    """The output tensor has 4 + nc rows and the anchor count the JSON states."""
    (spec,) = export["outputs"]
    m = re.search(r"strides ([0-9, and]+?), (\d+) at (\d+) x (\d+)", spec["meaning"])
    strides = [int(v) for v in re.findall(r"\d+", m.group(1))]
    count, ch, cw = int(m.group(2)), int(m.group(3)), int(m.group(4))
    ok = (
        spec["shape"] == "[batch, 4 + nc, anchors]"
        and sum((ch // st) * (cw // st) for st in strides) == count
        and out.shape == (1, 4 + nc, sum((H // st) * (W // st) for st in strides))
    )
    print(f"[{'PASS' if ok else 'FAIL'}] {label} output shape: {tuple(out.shape)}, strides {strides}")
    return ok


def drop_empty(ours):
    b = ours["boxes"]
    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return {k: v[keep] for k, v in ours.items()}


def decode_e2e(pred, conf):
    """One-to-one head: top max_det (anchor, class) pairs, then conf. No NMS."""
    pred = pred.T  # [anchors, 4 + nc]
    nc = pred.shape[1] - 4
    flat = pred[:, 4:].reshape(-1)  # pair i = anchor i // nc, class i % nc
    top = np.argsort(-flat, kind="stable")[: min(MAX_DET, flat.size)]
    top = top[flat[top] > conf]
    return pred[top // nc, :4], flat[top], (top % nc).astype(np.int64)


def decode_nms(pred, conf):
    """Same steps as the yolo9 default export: multi-label conf, class-aware NMS."""
    pred = pred.T  # [anchors, 4 + nc]
    anchor, cls = np.nonzero(pred[:, 4:] > conf)
    boxes, scores, classes = pred[anchor, :4], pred[anchor, 4 + cls], cls
    keep = c.class_aware_nms(boxes, scores, classes, IOU, MAX_DET)
    return boxes[keep], scores[keep], classes[keep]


for key, ckpt, decode, low_conf in (
    ("yolo9_e2e", "LibreYOLO9E2Et.pt", decode_e2e, 1e-4),
    ("yolo9_p2", "LibreYOLO9P2s-visdrone.pt", decode_nms, 1e-3),
):
    variant = c.load_variant("yolo9", key)
    whens = [e["when"] for e in variant["tasks"]["detect"]]
    try:
        LibreYOLO(ckpt).export(format="onnx", nms=True, output_path=str(WORK / f"{key}_nms.onnx"))
        rejected = False
    except NotImplementedError:
        rejected = True
    ok = rejected and whens == ["default"] and any("no nms=True export" in n for n in variant["input"]["notes"])
    print(f"[{'PASS' if ok else 'FAIL'}] {key} nms=True rejected by the exporter and absent from the JSON")
    results.append(ok)

    (export,) = variant["tasks"]["detect"]
    for imgsz in (640, (384, 640)):
        model = LibreYOLO(ckpt)
        path = WORK / f"{key}_raw.onnx"
        LibreYOLO(ckpt).export(format="onnx", imgsz=imgsz, output_path=str(path))
        meta = c.onnx_metadata(path)
        sess = c.session(path)
        assert sess.get_inputs()[0].name == variant["input"]["name"]
        assert [o.name for o in sess.get_outputs()] == [o["name"] for o in export["outputs"]]
        H, W = c.input_hw(sess, meta)
        x, geo = c.preprocess(image, variant["input"], H, W, meta)
        (out,) = sess.run(None, {variant["input"]["name"]: x})
        label = f"{key} detect default imgsz={imgsz} input={H}x{W}"
        results.append(check_output_shape(export, out, int(meta["nc"]), H, W, label))

        for conf in (CONF, low_conf):
            ref = c.reference_detections(model.predict(SAMPLE_IMAGE, imgsz=imgsz, conf=conf, iou=IOU))
            boxes, scores, classes = decode(out[0], conf)
            ours = drop_empty({"boxes": c.to_original_xyxy(boxes, geo), "scores": scores, "classes": classes})
            results.append(c.compare_detections(ours, ref, label=f"{label} conf={conf}"))
        path.unlink()

c.finish(results)
