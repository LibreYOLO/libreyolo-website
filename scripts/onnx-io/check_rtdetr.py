"""Check src/data/docs/onnx-io/rtdetr.json against LibreYOLO predictions.

Usage: python scripts/onnx-io/check_rtdetr.py <work_dir>

Covers the smallest checkpoint of every registry key and task: LibreRTDETR and
LibreRTDETRv2 detect (cv2 stretch), LibreRTDETRv2 OBB (Pillow resize to a rounded
size, top-left pad) and LibreRTDETRv4 detect (Pillow stretch). The OBB variant
uses ``"resize": "custom"`` because its resized size is rounded, not truncated
as in ``_common``'s letterbox. Its input is built here, from the entry's notes.
Every default export also runs at batch 2 with two different images.
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

from libreyolo import SAMPLE_IMAGE, LibreYOLO  # noqa: E402  (export + reference only)

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
WORK.mkdir(parents=True, exist_ok=True)
CONF, MAX_DET, QUERIES = 0.25, 300, 300
FAMILY = "rtdetr"
DEVICE = "cpu"  # the reference and the export both run on CPU

# (registry key, task, smallest checkpoint, square sizes that must export and match)
CASES = [
    ("rtdetr", "detect", "LibreRTDETRr18.pt", [None, 512]),
    ("rtdetrv2", "detect", "LibreRTDETRv2r18.pt", [None]),
    ("rtdetrv2", "obb", "LibreRTDETRv2n-obb.pt", [None]),
    ("rtdetrv4", "detect", "LibreRTDETRv4s.pt", [None, 512]),
]
# Square sizes the entry says fail during export.
REJECTED_SQUARE = {("rtdetrv2", "detect"): 512, ("rtdetrv2", "obb"): 640}

entry = c.load_entry(FAMILY)
SECOND_IMAGE = str(Path(SAMPLE_IMAGE).with_name("guggenheim-bilbao.jpg"))
BATCH1_NOTE = (
    "Batch size 1 only. The graph declares a dynamic batch axis, but batches of 2 or "
    "more fail in 1.7.0."
)
image = cv2.imread(str(SAMPLE_IMAGE))
results = []


def variant_for(key: str, task: str) -> dict:
    for variant in entry["variants"]:
        if key in variant["keys"] and task in variant["tasks"]:
            return variant
    raise KeyError(f"{FAMILY}.json has no variant for {key!r} {task!r}")


# Pillow Image.resize(BILINEAR) in fixed point, as the OBB notes say: width pass,
# then height pass, weights rounded to 22 fractional bits, integer sums, 8-bit after
# each pass. _common's float version is close but not exact, and the OBB model
# turns one box on the test image by 0.2 rad with it.
PRECISION_BITS = 22


def _pillow_pass(img: np.ndarray, axis: int, out_size: int) -> np.ndarray:
    in_size = img.shape[axis]
    s = in_size / out_size
    f = max(s, 1.0)
    ksize = int(math.ceil(f)) * 2 + 1
    idx = np.zeros((out_size, ksize), np.int64)
    wts = np.zeros((out_size, ksize), np.int64)
    for i in range(out_size):
        center = (i + 0.5) * s
        lo = max(int(center - f + 0.5), 0)
        hi = min(int(center + f + 0.5), in_size)
        raw = [max(0.0, 1.0 - abs(j + 0.5 - center) / f) for j in range(lo, hi)]
        total = 0.0
        for w in raw:
            total += w
        for t, (j, w) in enumerate(zip(range(lo, hi), raw)):
            w = w / total if total != 0.0 else w
            idx[i, t] = j
            wts[i, t] = int(w * (1 << PRECISION_BITS) + (0.5 if w >= 0 else -0.5))
    moved = np.moveaxis(img, axis, 1).astype(np.int64)  # (other, in, C)
    acc = (moved[:, idx, :] * wts[None, :, :, None]).sum(axis=2) + (1 << (PRECISION_BITS - 1))
    return np.moveaxis(np.clip(acc >> PRECISION_BITS, 0, 255).astype(np.uint8), 1, axis)


def pillow_bilinear_fixed(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if out_w != img.shape[1]:
        img = _pillow_pass(img, 1, out_w)
    if out_h != img.shape[0]:
        img = _pillow_pass(img, 0, out_h)
    return img


def preprocess(image_bgr: np.ndarray, inp: dict, H: int, W: int, metadata: dict, resize=None):
    """``resize`` replaces the documented OBB resize, only to test the notes' claims."""
    if inp["resize"] != "custom":
        return c.preprocess(image_bgr, inp, H, W, metadata)
    # LibreRTDETRv2 OBB: r = min(height / h, width / w), resize to
    # (round(w * r), round(h * r)), paste top-left on a canvas of pad_value.
    assert inp["pad_position"] == "topleft", inp["pad_position"]
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if inp["color"] == "RGB" else image_bgr
    oh, ow = img.shape[:2]
    r = min(H / oh, W / ow)
    nw, nh = round(ow * r), round(oh * r)
    canvas = np.full((H, W, 3), inp["pad_value"], dtype=np.uint8)
    assert inp["interpolation"] == "pillow-bilinear", inp["interpolation"]
    canvas[:nh, :nw] = (resize or pillow_bilinear_fixed)(img, nw, nh)
    x = canvas.astype(np.float32) / float(inp["scale"])
    if inp.get("mean") is not None:
        x = (x - np.asarray(inp["mean"], np.float32)) / np.asarray(inp["std"], np.float32)
    geo = {"orig_w": ow, "orig_h": oh, "input_w": W, "input_h": H, "r": r}
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), geo


# --- Decoding, following the postprocess steps ---


def top_k(logits: np.ndarray):
    """Sigmoid, top max_det over queries x classes, then score > conf."""
    nc = logits.shape[-1]
    flat = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).reshape(-1)
    order = np.argsort(-flat, kind="stable")[: min(MAX_DET, flat.size)]
    scores = flat[order]
    keep = scores > CONF
    return order[keep] // nc, order[keep] % nc, scores[keep]


def decode_detect(outs: dict, geo: dict, b: int = 0) -> dict:
    query, cls, scores = top_k(outs["pred_logits"][b])
    cx, cy, w, h = outs["pred_boxes"][b][query].astype(np.float64).T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] *= geo["orig_w"]
    boxes[:, [1, 3]] *= geo["orig_h"]
    return {"boxes": boxes, "scores": scores, "classes": cls}


def corners(xywhr: np.ndarray) -> np.ndarray:
    cx, cy, w, h, a = xywhr.T
    cos, sin = np.cos(a), np.sin(a)
    out = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out.append(np.stack([cx + dx * cos - dy * sin, cy + dx * sin + dy * cos], axis=-1))
    return np.stack(out, axis=1)  # (n, 4, 2)


def decode_obb(outs: dict, geo: dict, b: int = 0) -> dict:
    query, cls, scores = top_k(outs["pred_logits"][b])
    q = outs["pred_boxes"][b][query].astype(np.float64)
    W, H, r = geo["input_w"], geo["input_h"], geo["r"]
    xywhr = np.stack(
        [q[:, 0] * W / r, q[:, 1] * H / r, q[:, 2] * W / r, q[:, 3] * H / r, q[:, 4] * math.pi],
        axis=1,
    )
    pts = corners(xywhr)
    boxes = np.concatenate([pts.min(axis=1), pts.max(axis=1)], axis=1)
    return {"xywhr": xywhr, "corners": pts, "boxes": boxes, "scores": scores, "classes": cls}


def reference_obb(result) -> dict:
    data = c._np(result.obb.data).astype(np.float64)
    return {
        "xywhr": data[:, :5],
        "corners": c._np(result.obb.xyxyxyxy).astype(np.float64).reshape(-1, 4, 2),
        "boxes": c._np(result.boxes.xyxy).astype(np.float64).reshape(-1, 4),
        "scores": data[:, 5],
        "classes": data[:, 6].astype(np.int64),
    }


def obb_metrics(ours: dict, ref: dict) -> dict:
    """Greedy same-class match on (cx, cy, w, h), worst differences over the pairs."""
    n_ours, n_ref = len(ours["xywhr"]), len(ref["xywhr"])
    used, unmatched = set(), 0
    worst_xywh = worst_angle = worst_corner = worst_box = worst_score = 0.0
    for i in np.argsort(-ref["scores"], kind="stable"):
        cands = [j for j in range(n_ours) if j not in used and ours["classes"][j] == ref["classes"][i]]
        if not cands:
            unmatched += 1
            continue
        cost = [np.abs(ours["xywhr"][j, :4] - ref["xywhr"][i, :4]).max() for j in cands]
        j = cands[int(np.argmin(cost))]
        used.add(j)
        worst_xywh = max(worst_xywh, float(np.abs(ours["xywhr"][j, :4] - ref["xywhr"][i, :4]).max()))
        worst_angle = max(worst_angle, float(abs(ours["xywhr"][j, 4] - ref["xywhr"][i, 4])))
        worst_corner = max(worst_corner, float(np.abs(ours["corners"][j] - ref["corners"][i]).max()))
        worst_box = max(worst_box, float(np.abs(ours["boxes"][j] - ref["boxes"][i]).max()))
        worst_score = max(worst_score, abs(float(ours["scores"][j] - ref["scores"][i])))
    return dict(
        n_ours=n_ours, n_ref=n_ref, unmatched=unmatched, xywh=worst_xywh, angle=worst_angle,
        corner=worst_corner, box=worst_box, score=worst_score,
    )


def compare_obb(ours: dict, ref: dict, label: str) -> bool:
    """Pass when counts agree and every pair is within 1 px on cx, cy, w, h, the
    corners and the enclosing box, and within 0.01 rad on the angle."""
    m = obb_metrics(ours, ref)
    ok = (
        m["n_ours"] == m["n_ref"]
        and m["unmatched"] == 0
        and max(m["xywh"], m["corner"], m["box"]) <= 1.0
        and m["angle"] <= 0.01
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] {label}: ours={m['n_ours']} ref={m['n_ref']} unmatched={m['unmatched']} "
        f"max_xywh_diff_px={m['xywh']:.3f} max_angle_diff_rad={m['angle']:.5f} "
        f"max_corner_diff_px={m['corner']:.3f} max_enclosing_box_diff_px={m['box']:.3f} "
        f"max_score_diff={m['score']:.4f}"
    )
    return ok


def detect_metrics(ours: dict, ref: dict) -> dict:
    """Quiet version of _common.compare_detections: counts and worst matched box diff."""
    worst, used = 0.0, set()
    if len(ours["boxes"]) and len(ref["boxes"]):
        ious = c.iou_matrix(ref["boxes"], ours["boxes"])
        for i in np.argsort(-ref["scores"]):
            cands = [j for j in np.argsort(-ious[i]) if j not in used and ours["classes"][j] == ref["classes"][i]]
            if cands and ious[i, cands[0]] >= 0.5:
                used.add(cands[0])
                worst = max(worst, float(np.abs(ours["boxes"][cands[0]] - ref["boxes"][i]).max()))
    return dict(n_ours=len(ours["boxes"]), n_ref=len(ref["boxes"]), box=worst)


def check_resize_claims(sess, inp, meta, H, W, task, ref, tag) -> list:
    """The Pillow notes claim OpenCV resizes miss the reference on the test image
    (detect: two extra detections, boxes off by more than 2 px), and for OBB that
    float Pillow weights turn a box by about 0.2 rad. Prove each claim."""
    names = [o.name for o in sess.get_outputs()]

    def run(x, geo):
        outs = dict(zip(names, sess.run(None, {inp["name"]: x})))
        return decode_obb(outs, geo) if task == "obb" else decode_detect(outs, geo)

    checks = []
    for label, interp in (("cv2 INTER_LINEAR", cv2.INTER_LINEAR), ("cv2 INTER_AREA", cv2.INTER_AREA)):
        if task == "obb":
            x, geo = preprocess(image, inp, H, W, meta, resize=lambda im, w, h: cv2.resize(im, (w, h), interpolation=interp))
            m = obb_metrics(run(x, geo), ref)
            ok = m["n_ours"] != m["n_ref"] and m["xywh"] > 2.0
            detail = f"ours={m['n_ours']} ref={m['n_ref']} max_xywh_diff_px={m['xywh']:.3f}"
        else:
            name = {cv2.INTER_LINEAR: "bilinear", cv2.INTER_AREA: "area"}[interp]
            x, geo = preprocess(image, dict(inp, interpolation=name), H, W, meta)
            m = detect_metrics(run(x, geo), ref)
            ok = m["n_ours"] - m["n_ref"] == 2 and m["box"] > 2.0
            detail = f"ours={m['n_ours']} ref={m['n_ref']} max_box_diff_px={m['box']:.3f}"
        print(f"[{'PASS' if ok else 'FAIL'}] {tag} claim: {label} instead of Pillow misses the reference: {detail}")
        checks.append(ok)
    if task == "obb":
        x, geo = preprocess(image, inp, H, W, meta, resize=c.pillow_bilinear_resize)
        m = obb_metrics(run(x, geo), ref)
        ok = 0.1 <= m["angle"] <= 0.3
        print(
            f"[{'PASS' if ok else 'FAIL'}] {tag} claim: float Pillow weights turn a box by about 0.2 rad: "
            f"max_angle_diff_rad={m['angle']:.4f} max_xywh_diff_px={m['xywh']:.3f}"
        )
        checks.append(ok)
    return checks


def check_outputs(sess, export: dict, nc: int, label: str) -> bool:
    """Output names in order, and every numeric or known symbolic dim, as the entry says."""
    want = export["outputs"]
    got = sess.get_outputs()
    ok = [o.name for o in got] == [o["name"] for o in want]
    for o, w in zip(got, want):
        dims = [d.strip() for d in w["shape"].strip("[]").split(",")]
        ok &= len(dims) == len(o.shape)
        for d, actual in zip(dims, o.shape):
            expected = {"queries": QUERIES, "nc": nc}.get(d, int(d) if d.isdigit() else None)
            if expected is not None:
                ok &= actual == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: outputs {[(o.name, o.shape) for o in got]}")
    return bool(ok)


def check_value_ranges(outs: dict, label: str) -> bool:
    """pred_boxes normalized 0 to 1 (angle column included), pred_logits not yet sigmoided."""
    boxes, logits = outs["pred_boxes"], outs["pred_logits"]
    ok = bool(boxes.min() >= 0.0 and boxes.max() <= 1.0 and logits.min() < 0.0)
    print(
        f"[{'PASS' if ok else 'FAIL'}] {label}: pred_boxes in [{boxes.min():.4f}, {boxes.max():.4f}], "
        f"pred_logits in [{logits.min():.2f}, {logits.max():.2f}]"
    )
    return ok


def expect_export_error(ckpt: str, label: str, error: type, **kwargs) -> bool:
    path = WORK / "rejected.onnx"
    try:
        LibreYOLO(ckpt, device=DEVICE).export(format="onnx", output_path=str(path), **kwargs)
    except error as exc:
        print(f"[PASS] {label}: export raises {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {label}: expected {error.__name__}, got {type(exc).__name__}: {exc}")
        return False
    finally:
        path.unlink(missing_ok=True)
    print(f"[FAIL] {label}: export succeeded, the entry says it is rejected")
    return False


def check_batch2(sess, inp, meta, H, W, task, model, ref, tag) -> bool:
    """Run the default graph on two different images stacked. Each element must
    decode like the .pt on its own image, unless the entry says batch 1 only."""
    claims_batch1 = BATCH1_NOTE in inp["notes"]
    x0, geo0 = preprocess(image, inp, H, W, meta)
    x1, geo1 = preprocess(cv2.imread(SECOND_IMAGE), inp, H, W, meta)
    try:
        raw = sess.run(None, {inp["name"]: np.concatenate([x0, x1])})
    except Exception as exc:  # noqa: BLE001
        print(f"[{'PASS' if claims_batch1 else 'FAIL'}] {tag} batch=2 fails: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        return claims_batch1
    if claims_batch1:
        print(f"[FAIL] {tag} batch=2 runs, but the entry says batch 1 only")
        return False
    outs = dict(zip([o.name for o in sess.get_outputs()], raw))
    second = model.predict(SECOND_IMAGE, conf=CONF, max_det=MAX_DET)
    if task == "obb":
        ok = compare_obb(decode_obb(outs, geo0, 0), ref, f"{tag} batch=2 element 0 vs .pt")
        ok &= compare_obb(decode_obb(outs, geo1, 1), reference_obb(second), f"{tag} batch=2 element 1 vs .pt")
    else:
        ok = c.compare_detections(decode_detect(outs, geo0, 0), ref, label=f"{tag} batch=2 element 0 vs .pt")
        ok &= c.compare_detections(
            decode_detect(outs, geo1, 1), c.reference_detections(second), label=f"{tag} batch=2 element 1 vs .pt"
        )
    return bool(ok)


for key, task, ckpt, sizes in CASES:
    variant = variant_for(key, task)
    inp = variant["input"]
    export = variant["tasks"][task][0]
    assert export["when"] == "default"
    for imgsz in sizes:
        tag = f"{key} {task} imgsz={imgsz or 'default'}"
        model = LibreYOLO(ckpt, device=DEVICE)
        result = model.predict(SAMPLE_IMAGE, imgsz=imgsz, conf=CONF, max_det=MAX_DET)
        ref = reference_obb(result) if task == "obb" else c.reference_detections(result)

        path = WORK / f"{Path(ckpt).stem}_{imgsz or 'default'}.onnx"
        kwargs = {} if imgsz is None else {"imgsz": imgsz}
        model.export(format="onnx", output_path=str(path), **kwargs)
        meta = c.onnx_metadata(path)
        sess = c.session(path)
        assert sess.get_inputs()[0].name == inp["name"]
        H, W = c.input_hw(sess, meta)
        assert H == W, (H, W)
        nc = int(meta["nc"])
        results.append(check_outputs(sess, export, nc, f"{tag} input={H}x{W}"))

        x, geo = preprocess(image, inp, H, W, meta)
        outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {inp["name"]: x})))
        results.append(check_value_ranges(outs, tag))

        if task == "obb":
            ours = decode_obb(outs, geo)
            results.append(compare_obb(ours, ref, f"{tag} input={H}x{W} ours vs .pt"))
            backend = reference_obb(LibreYOLO(str(path)).predict(SAMPLE_IMAGE, imgsz=imgsz, conf=CONF))
            results.append(compare_obb(backend, ref, f"{tag} LibreYOLO(.onnx).predict vs .pt"))
        else:
            ours = decode_detect(outs, geo)
            results.append(c.compare_detections(ours, ref, label=f"{tag} input={H}x{W} ours vs .pt"))
            backend = c.reference_detections(LibreYOLO(str(path)).predict(SAMPLE_IMAGE, imgsz=imgsz, conf=CONF))
            results.append(c.compare_detections(backend, ref, label=f"{tag} LibreYOLO(.onnx).predict vs .pt"))
        if imgsz is None:
            results.append(check_batch2(sess, inp, meta, H, W, task, model, ref, tag))
            if inp["interpolation"] == "pillow-bilinear":
                results.extend(check_resize_claims(sess, inp, meta, H, W, task, ref, tag))
        path.unlink()

    # What the entry says export refuses.
    results.append(expect_export_error(ckpt, f"{key} {task} imgsz=(384, 640)", NotImplementedError, imgsz=(384, 640)))
    results.append(expect_export_error(ckpt, f"{key} {task} nms=True", NotImplementedError, nms=True))
    if (key, task) in REJECTED_SQUARE:
        size = REJECTED_SQUARE[(key, task)]
        results.append(expect_export_error(ckpt, f"{key} {task} imgsz={size}", RuntimeError, imgsz=size))

c.finish(results)
