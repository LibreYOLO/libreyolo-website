"""Check src/data/docs/onnx-io/dfine.json against LibreYOLO predictions.

Usage: python scripts/onnx-io/check_dfine.py <work_dir>

D-FINE stretches the image with Pillow's BILINEAR filter. The input block says
``"resize": "stretch"`` with ``"interpolation": "pillow-bilinear"``, which
_common.preprocess builds with numpy only. The script also proves that OpenCV
resizes do not reproduce it, and runs the default graph at batch 2.
"""

import contextlib
import io
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
CONF, MAX_DET, QUERIES, MASK_IOU = 0.25, 300, 300, 0.99
CHECKPOINTS = {"detect": "LibreDFINEn.pt", "segment": "LibreDFINEn-seg.pt"}
BATCH1_NOTE = "Batch size 1 only."

variant = c.load_variant("dfine", "dfine")
inp = variant["input"]
image = cv2.imread(str(SAMPLE_IMAGE))
results = []


# ---------------------------------------------------------------- outputs


def check_shapes(sess, export, H, W, nc, label) -> bool:
    """Every output named in the entry exists with the documented shape."""
    sizes = {"queries": QUERIES, "nc": nc, "mask_h": H // 4, "mask_w": W // 4}
    graph = {o.name: o.shape for o in sess.get_outputs()}
    ok = [o["name"] for o in export["outputs"]] == list(graph)
    for o in export["outputs"]:
        doc = re.findall(r"\w+", o["shape"])
        got = graph.get(o["name"], [])
        ok &= len(doc) == len(got)
        for sym, dim in zip(doc, got):
            if sym == "batch":
                ok &= isinstance(dim, str) or dim == 1
            else:
                ok &= dim == (int(sym) if sym.isdigit() else sizes[sym])
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: output names and shapes {graph}")
    return bool(ok)


def decode(outs: dict, geo: dict, task: str, b: int = 0) -> dict:
    """The postprocess steps of the entry, in order, for batch element b."""
    oh, ow, H, W = geo["orig_h"], geo["orig_w"], geo["input_h"], geo["input_w"]
    logits, boxes = outs["pred_logits"][b], outs["pred_boxes"][b]
    nc = logits.shape[1]
    scores = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).reshape(-1)
    top = np.argsort(-scores, kind="stable")[:MAX_DET]
    top = top[scores[top] > CONF]
    query, cls = top // nc, top % nc
    cx, cy, w, h = boxes[query].astype(np.float64).T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    xyxy[:, [0, 2]] *= ow
    xyxy[:, [1, 3]] *= oh
    det = {"boxes": xyxy, "scores": scores[top], "classes": cls.astype(np.int64)}
    if task == "segment":
        ys, xs = np.mgrid[0:oh, 0:ow]
        masks = []
        for q, (x1, y1, x2, y2) in zip(query, xyxy):
            m = cv2.resize(outs["pred_masks"][b, q], (W, H), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_LINEAR)
            inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
            masks.append((m >= 0.5) & inside)
        det["masks"] = np.asarray(masks, dtype=bool).reshape(-1, oh, ow)
    return det


def reference(result, task: str) -> dict:
    ref = c.reference_detections(result)
    if task == "segment":
        ref["masks"] = c._np(result.masks.data).astype(bool)
    return ref


def match(ours: dict, ref: dict) -> list:
    """The same greedy same-class IoU pairing as _common.compare_detections."""
    pairs, used = [], set()
    if not len(ours["boxes"]) or not len(ref["boxes"]):
        return pairs
    ious = c.iou_matrix(ref["boxes"], ours["boxes"])
    for i in np.argsort(-ref["scores"]):
        cands = [j for j in np.argsort(-ious[i]) if j not in used and ours["classes"][j] == ref["classes"][i]]
        if cands and ious[i, cands[0]] >= 0.5:
            used.add(cands[0])
            pairs.append((i, cands[0]))
    return pairs


def compare(ours: dict, ref: dict, label: str) -> bool:
    ok = c.compare_detections(ours, ref, label=label)
    if "masks" in ref:
        ious = []
        for i, j in match(ours, ref):
            a, b = ref["masks"][i], ours["masks"][j]
            union = np.logical_or(a, b).sum()
            ious.append(1.0 if union == 0 else np.logical_and(a, b).sum() / union)
        worst = min(ious) if ious else 0.0
        mask_ok = len(ious) == len(ref["masks"]) == len(ours["masks"]) and worst >= MASK_IOU
        print(f"[{'PASS' if mask_ok else 'FAIL'}] {label} masks: pairs={len(ious)} min_mask_iou={worst:.4f}")
        ok = ok and mask_ok
    return ok


def expect_raises(fn, label: str) -> bool:
    try:
        fn()
    except NotImplementedError as e:
        print(f"[PASS] {label}: NotImplementedError: {str(e).splitlines()[0][:110]}")
        return True
    print(f"[FAIL] {label}: export did not raise")
    return False


def check_batch2(sess, names, meta, H, W, task, ckpt, ref) -> bool:
    """Run the default graph on [image, flipped image] and compare each element."""
    flipped_path = WORK / "flipped.png"
    flipped = cv2.flip(image, 1)
    cv2.imwrite(str(flipped_path), flipped)
    refs = [ref, reference(LibreYOLO(ckpt).predict(str(flipped_path), conf=CONF, max_det=MAX_DET), task)]
    flipped_path.unlink()
    xs, geos = zip(*(c.preprocess(im, inp, H, W, meta) for im in (image, flipped)))
    note = BATCH1_NOTE in " ".join(inp["notes"])
    try:
        outs = dict(zip(names, sess.run(None, {inp["name"]: np.concatenate(xs)})))
    except Exception as e:  # onnxruntime raises its own exception types
        print(f"[{'PASS' if note else 'FAIL'}] dfine {task} batch 2 fails{' as the notes say' if note else ''}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        return note
    ok = not note
    for b in range(2):
        ok &= compare(decode(outs, geos[b], task, b), refs[b], f"dfine {task} default batch=2 element {b}")
    if note:
        print(f"[FAIL] dfine {task} batch 2 works but the notes say batch 1 only")
    return bool(ok)


# ---------------------------------------------------------------- run

for task, ckpt in CHECKPOINTS.items():
    exports = variant["tasks"][task]
    assert [e["when"] for e in exports] == ["default"]
    export = exports[0]
    for imgsz in (640, 320):
        label = f"dfine {task} default imgsz={imgsz}"
        ref = reference(LibreYOLO(ckpt).predict(SAMPLE_IMAGE, imgsz=imgsz, conf=CONF, max_det=MAX_DET), task)
        path = WORK / f"dfinen_{task}_{imgsz}.onnx"
        LibreYOLO(ckpt).export(format="onnx", imgsz=imgsz, output_path=str(path))
        meta = c.onnx_metadata(path)
        sess = c.session(path)
        assert sess.get_inputs()[0].name == inp["name"]
        H, W = c.input_hw(sess, meta)
        results.append(check_shapes(sess, export, H, W, int(meta["nc"]), f"{label} input={H}x{W}"))

        assert inp["resize"] == "stretch" and inp["interpolation"] == "pillow-bilinear"
        x, geo = c.preprocess(image, inp, H, W, meta)
        names = [o.name for o in sess.get_outputs()]
        outs = dict(zip(names, sess.run(None, {inp["name"]: x})))
        results.append(compare(decode(outs, geo, task), ref, f"{label} input={H}x{W}"))

        if imgsz == 640:
            # The notes say OpenCV resizes do not reproduce the input. Prove it.
            if task == "detect":
                for interp in ("bilinear", "area"):
                    alt = dict(inp, interpolation=interp)
                    xa, _ = c.preprocess(image, alt, H, W, meta)
                    oa = dict(zip(names, sess.run(None, {inp["name"]: xa})))
                    with contextlib.redirect_stdout(io.StringIO()) as buf:
                        same = c.compare_detections(decode(oa, geo, task), ref)
                    print(f"[{'FAIL' if same else 'PASS'}] cv2 {interp} stretch differs as the notes say:{buf.getvalue().split(':', 1)[1].rstrip()}")
                    results.append(not same)
            # LibreYOLO's own ONNX backend on the same file, against the .pt.
            backend = reference(LibreYOLO(str(path)).predict(SAMPLE_IMAGE, conf=CONF, max_det=MAX_DET), task)
            results.append(compare(backend, ref, f"LibreYOLO(onnx).predict {task} vs .pt"))
            results.append(check_batch2(sess, names, meta, H, W, task, ckpt, ref))
        del sess
        path.unlink()

    results.append(expect_raises(
        lambda: LibreYOLO(ckpt).export(format="onnx", nms=True, output_path=str(WORK / "nms.onnx")),
        f"dfine {task} nms=True export is rejected"))
    assert "Square input only." in " ".join(inp["notes"])
    results.append(expect_raises(
        lambda: LibreYOLO(ckpt).export(format="onnx", imgsz=(384, 640), output_path=str(WORK / "rect.onnx")),
        f"dfine {task} imgsz=(384, 640) export is rejected"))

for leftover in WORK.glob("*.onnx"):
    leftover.unlink()
c.finish(results)
