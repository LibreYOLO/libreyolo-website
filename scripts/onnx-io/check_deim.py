"""Check src/data/docs/onnx-io/deim.json against LibreYOLO predictions.

Usage: python scripts/onnx-io/check_deim.py <work_dir>

Covers both registry keys and both DEIMv2 preprocessing branches:
LibreDEIMn (deim), LibreDEIMv2atto (deimv2, plain /255) and LibreDEIMv2s
(deimv2, DINOv3 sizes with ImageNet mean and std). LibreDEIMn is also checked
at a second square size (480) so the "read height and width from imgsz" claim
is not only proven at the native size.
"""

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
# (registry key, checkpoint, export imgsz or None for the default, documented query count)
CASES = [
    ("deim", "LibreDEIMn.pt", None, 300),
    ("deim", "LibreDEIMn.pt", 480, 300),
    ("deimv2", "LibreDEIMv2atto.pt", None, 100),
    ("deimv2", "LibreDEIMv2s.pt", None, 300),
]


def variant_for(key: str, size: str) -> dict:
    """Variants are split by model size, listed in each variant's sizes field."""
    for variant in c.load_entry("deim")["variants"]:
        if key in variant["keys"] and size in variant["sizes"].get(key, []):
            return variant
    raise KeyError(f"deim.json has no variant for {key} size {size!r}")


def decode(logits: np.ndarray, boxes: np.ndarray, orig_w: int, orig_h: int) -> dict:
    """The entry's postprocess steps, for one image."""
    nc = logits.shape[-1]
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64).reshape(-1)))  # sigmoid
    top = np.argsort(-scores, kind="stable")[:MAX_DET]  # top 300 over queries x nc
    query, cls, scores = top // nc, top % nc, scores[top]
    keep = scores > CONF
    query, cls, scores = query[keep], cls[keep], scores[keep]
    cx, cy, w, h = boxes[query].astype(np.float64).T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    xyxy *= [orig_w, orig_h, orig_w, orig_h]  # no clipping, no NMS
    return {"boxes": xyxy, "scores": scores, "classes": cls.astype(np.int64)}


def report(ok: bool, label: str, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{': ' + detail if detail else ''}")
    return ok


def run_batch(sess, x: np.ndarray):
    try:
        return sess.run(None, {sess.get_inputs()[0].name: x}), None
    except Exception as exc:  # noqa: BLE001  (onnxruntime raises its own error types)
        return None, str(exc).splitlines()[0][:150]


def raises_not_implemented(fn) -> tuple[bool, str]:
    try:
        fn()
    except NotImplementedError as exc:
        return True, f"NotImplementedError: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return False, "export succeeded"


image = cv2.imread(str(SAMPLE_IMAGE))
second = cv2.flip(image, 1)  # a different image for the batch checks
oh, ow = image.shape[:2]
results = []

for key, ckpt, imgsz, queries in CASES:
    model = LibreYOLO(ckpt)
    size = model.size
    variant = variant_for(key, size)
    export = variant["tasks"]["detect"][0]
    assert export["when"] == "default"
    tag = f"{key} {size} detect default imgsz={imgsz or model.input_size}"
    ref = c.reference_detections(
        model.predict(SAMPLE_IMAGE, imgsz=imgsz, conf=CONF, iou=IOU, max_det=MAX_DET)
    )

    path = WORK / f"{Path(ckpt).stem}_{imgsz or 'default'}.onnx"
    kwargs = {"imgsz": imgsz} if imgsz else {}
    LibreYOLO(ckpt).export(format="onnx", output_path=str(path), **kwargs)
    meta = c.onnx_metadata(path)
    sess = c.session(path)
    inp = sess.get_inputs()[0]
    H, W = c.input_hw(sess, meta)
    nc = int(meta["nc"])
    io_ok = (
        inp.name == variant["input"]["name"]
        and inp.type == "tensor(float)"
        and H == W == int(meta["imgsz"])
        and [o.name for o in sess.get_outputs()] == [o["name"] for o in export["outputs"]]
        and [o.shape[1:] for o in sess.get_outputs()] == [[queries, nc], [queries, 4]]
        and f"{queries} queries" in export["outputs"][0]["meaning"]
    )
    results.append(report(io_ok, f"{tag} io", f"input {inp.name} {inp.shape}, outputs "
                          + ", ".join(f"{o.name} {o.shape}" for o in sess.get_outputs())))

    x, geo = c.preprocess(image, variant["input"], H, W, meta)
    names = [o["name"] for o in export["outputs"]]
    logits, boxes = sess.run(names, {inp.name: x})
    ours = decode(logits[0], boxes[0], geo["orig_w"], geo["orig_h"])
    results.append(c.compare_detections(ours, ref, label=f"{tag} input={H}x{W} (numpy + onnxruntime)"))

    backend = c.reference_detections(
        LibreYOLO(str(path)).predict(SAMPLE_IMAGE, conf=CONF, iou=IOU, max_det=MAX_DET)
    )
    results.append(c.compare_detections(backend, ref, label=f"{tag} LibreYOLO(.onnx).predict"))

    # "Batch size 1 only": the default graph fails on 2 stacked images.
    x2, _ = c.preprocess(second, variant["input"], H, W, meta)
    out, err = run_batch(sess, np.concatenate([x, x2]))
    results.append(report(out is None, f"{tag} batch 2 fails on the default export",
                          err or "batch 2 ran, so drop the batch size 1 note"))
    path.unlink()

    if imgsz is None:
        # "export with batch=2": that graph takes batch 1, 2 and 3 with the same per-image outputs.
        path_b = WORK / f"{Path(ckpt).stem}_batch2.onnx"
        LibreYOLO(ckpt).export(format="onnx", output_path=str(path_b), batch=2)
        sess_b = c.session(path_b)
        one, e1 = run_batch(sess_b, x)
        two, e2 = run_batch(sess_b, np.concatenate([x, x2]))
        three, e3 = run_batch(sess_b, np.concatenate([x2, x, x2]))
        ok = not (e1 or e2 or e3)
        detail = e1 or e2 or e3 or ""
        if ok:
            per_image = []
            for batch_out, index, single in ((two, 0, x), (two, 1, x2), (three, 1, x)):
                solo = sess_b.run(None, {inp.name: single})
                per_image.append(c.compare_detections(
                    decode(batch_out[0][index], batch_out[1][index], ow, oh),
                    decode(solo[0][0], solo[1][0], ow, oh),
                    label=f"{tag} batch=2 export, image {index} of batch {len(batch_out[0])} vs alone",
                ))
            ok = all(per_image)
            batch1 = decode(one[0][0], one[1][0], ow, oh)
            ok = c.compare_detections(batch1, ref, label=f"{tag} batch=2 export run on 1 image vs .pt") and ok
        results.append(report(ok, f"{tag} batch=2 export takes batch 1, 2 and 3", detail))
        path_b.unlink()

        ok, detail = raises_not_implemented(
            lambda: LibreYOLO(ckpt).export(format="onnx", output_path=str(path), nms=True)
        )
        results.append(report(ok, f"{key} {size} nms=True rejected", detail))
        ok, detail = raises_not_implemented(
            lambda: LibreYOLO(ckpt).export(format="onnx", output_path=str(path), imgsz=(384, 640))
        )
        results.append(report(ok, f"{key} {size} imgsz=(384, 640) rejected", detail))
        path.unlink(missing_ok=True)

c.finish(results)
