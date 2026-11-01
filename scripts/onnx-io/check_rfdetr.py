"""Check src/data/docs/onnx-io/rfdetr.json against LibreYOLO predictions.

Usage: python scripts/onnx-io/check_rfdetr.py <work_dir>

Covers detect, segment, pose and obb with the smallest checkpoint of each task,
at the checkpoint's native square imgsz. RF-DETR rejects non-square imgsz and
nms=True at export; both rejections are checked too, as the JSON claims them.
"""

import math
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
CONF = 0.25
LOW_CONF = 0.01  # second pass: hundreds of entries, boxes past the image edge
TASKS = [
    ("detect", "LibreRFDETRn.pt"),
    ("segment", "LibreRFDETRn-seg.pt"),
    ("pose", "LibreRFDETRx-pose.pt"),
    ("obb", "LibreRFDETRn-obb.pt"),
]

variant = c.load_variant("rfdetr", "rfdetr")
INP = variant["input"]
image = cv2.imread(str(SAMPLE_IMAGE))
results = []
library = []  # LibreYOLO's own ONNX backend vs .pt: reported, not part of the JSON proof


def report(ok: bool, label: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ---------------------------------------------------------------- input


def preprocess(image_bgr: np.ndarray, task: str, H: int, W: int):
    """Build the input tensor from the JSON input block and its notes."""
    assert INP["resize"] == "custom" and INP["pad_value"] is None
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if INP["color"] == "RGB" else image_bgr
    x = img.astype(np.float32) / float(INP["scale"])  # divide first, then resize
    if task == "pose":
        # Same per-axis weights as Pillow BILINEAR, applied in float with no rounding.
        wy = c.pillow_bilinear_weights(x.shape[0], H)
        wx = c.pillow_bilinear_weights(x.shape[1], W)
        x = np.einsum("oh,hwc->owc", wy, x.astype(np.float64))
        x = np.einsum("pw,owc->opc", wx, x).astype(np.float32)
    else:
        x = cv2.resize(x, (W, H), interpolation=c.INTERPOLATION[INP["interpolation"]])
    x = (x - np.asarray(INP["mean"], np.float32)) / np.asarray(INP["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None].astype(np.float32))


# ---------------------------------------------------------------- outputs


def coco_ids_from_json() -> list[int]:
    """The COCO id list is parsed from the JSON step, so a typo there fails."""
    step = next(s for s in variant["tasks"]["detect"][0]["postprocess"] if "in this list:" in s)
    ids = [int(v) for v in re.findall(r"\d+", step.split("in this list:")[1].split(".")[0])]
    assert len(ids) == 80, len(ids)
    return ids


COCO_IDS = coco_ids_from_json()


def select_topk(labels: np.ndarray):
    """Sigmoid, top k = queries over queries x columns. Returns query, column, score."""
    scores = sigmoid(labels)  # [queries, columns]
    queries, columns = scores.shape
    flat = scores.reshape(-1)
    order = np.argsort(-flat, kind="stable")[:queries]
    return order // columns, order % columns, flat[order]


def map_classes(columns_idx: np.ndarray, n_columns: int, nc: int):
    """91 columns + nc 80: COCO id to class index, drop unused ids."""
    if n_columns == 91 and nc == 80:
        lut = {cid: i for i, cid in enumerate(COCO_IDS)}
        mapped = np.array([lut.get(int(v), -1) for v in columns_idx], dtype=np.int64)
        return mapped, mapped >= 0
    return columns_idx.astype(np.int64), np.ones(len(columns_idx), dtype=bool)


def cxcywh_to_original_xyxy(boxes: np.ndarray, ow: int, oh: int) -> np.ndarray:
    cx, cy, w, h = boxes.astype(np.float64).T
    return np.stack([(cx - w / 2) * ow, (cy - h / 2) * oh, (cx + w / 2) * ow, (cy + h / 2) * oh], 1)


def decode_detect(outs: dict, nc: int, ow: int, oh: int, conf: float) -> dict:
    labels = outs["labels"][0]
    q, col, score = select_topk(labels)
    keep = score > conf
    q, col, score = q[keep], col[keep], score[keep]
    cls, valid = map_classes(col, labels.shape[1], nc)
    q, cls, score = q[valid], cls[valid], score[valid]
    return {
        "boxes": cxcywh_to_original_xyxy(outs["dets"][0][q], ow, oh),
        "scores": score,
        "classes": cls,
        "query": q,
    }


def decode_masks(outs: dict, query: np.ndarray, ow: int, oh: int) -> np.ndarray:
    masks = []
    for logits in outs["masks"][0][query]:
        up = cv2.resize(logits.astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR)
        masks.append(up > 0)
    return np.asarray(masks, dtype=bool).reshape(-1, oh, ow)


def decode_pose(outs: dict, meta: dict, ow: int, oh: int, conf: float) -> dict:
    labels = outs["labels"][0]
    q, col, score = select_topk(labels)
    per_class = [int(v) for v in re.findall(r"\d+", meta["num_keypoints_per_class"])]
    K = max(per_class)
    kp_all = outs["keypoints"][0].astype(np.float64)  # [queries, columns * K, 8]
    assert kp_all.shape[1] == labels.shape[1] * K
    score = score.copy()
    blocks = []
    for n, (qi, ci) in enumerate(zip(q, col)):
        block = kp_all[qi, ci * K : ci * K + K]
        used = per_class[ci]
        if used > 0:
            r = block[:used]
            a, b, d = r[:, 4], r[:, 5], r[:, 6]
            p = np.maximum(sigmoid(r[:, 2]), 1e-12)
            sigma = np.exp(-2 * a) + np.exp(-2 * d) + b**2 * np.exp(-2 * a - 2 * d)
            trace = math.log(float((p * sigma).sum() / p.sum()))
            score[n] = score[n] * math.exp(-0.2 * trace)
        out = np.zeros((K, 3), dtype=np.float64)
        out[:used, 0] = block[:used, 0] * ow
        out[:used, 1] = block[:used, 1] * oh
        out[:used, 2] = sigmoid(block[:used, 2])
        blocks.append(out)
    blocks = np.asarray(blocks)
    keep = (col != 0) & (score > conf)
    return {
        "boxes": cxcywh_to_original_xyxy(outs["dets"][0][q[keep]], ow, oh),
        "scores": score[keep],
        "classes": (col[keep] - 1).astype(np.int64),
        "keypoints": blocks[keep],
    }


def xywhr_corners(rects: np.ndarray) -> np.ndarray:
    cx, cy, w, h, a = rects.T
    cos, sin = np.cos(a), np.sin(a)
    u = np.stack([cos * w / 2, sin * w / 2], -1)
    v = np.stack([-sin * h / 2, cos * h / 2], -1)
    ctr = np.stack([cx, cy], -1)
    return np.stack([ctr - u - v, ctr + u - v, ctr + u + v, ctr - u + v], 1)


def decode_obb(outs: dict, ow: int, oh: int, conf: float) -> dict:
    labels = outs["labels"][0]
    q, col, score = select_topk(labels)
    keep = score > conf
    q, col, score = q[keep], col[keep], score[keep]
    rel = np.concatenate([outs["dets"][0][q], outs["angles"][0][q]], 1).astype(np.float64)
    corners = xywhr_corners(rel)
    corners[..., 0] *= ow
    corners[..., 1] *= oh
    rects = []
    for pts in corners.astype(np.float32):
        (cx, cy), (w, h), deg = cv2.minAreaRect(pts)
        w, h, a = max(w, 1e-4), max(h, 1e-4), math.radians(deg)
        if h > w:
            w, h, a = h, w, a + math.pi / 2
        a = (a + math.pi / 2) % math.pi - math.pi / 2
        rects.append([cx, cy, w, h, a])
    return {"xywhr": np.asarray(rects, dtype=np.float64).reshape(-1, 5), "scores": score, "classes": col.astype(np.int64)}


# ---------------------------------------------------------------- comparisons


def match_pairs(ours: dict, ref: dict) -> list[tuple[int, int]]:
    """Same greedy same-class IoU matching as _common.compare_detections."""
    pairs, used = [], set()
    if not len(ours["boxes"]) or not len(ref["boxes"]):
        return pairs
    ious = c.iou_matrix(ref["boxes"], ours["boxes"])
    for i in np.argsort(-ref["scores"]):
        cand = [j for j in np.argsort(-ious[i]) if j not in used and ours["classes"][j] == ref["classes"][i]]
        if cand and ious[i, cand[0]] >= 0.5:
            used.add(cand[0])
            pairs.append((int(i), int(cand[0])))
    return pairs


def compare_masks(ours_masks, ref_masks, pairs, label) -> bool:
    ious = []
    for i, j in pairs:
        a, b = ref_masks[i], ours_masks[j]
        union = np.logical_or(a, b).sum()
        ious.append(1.0 if union == 0 else np.logical_and(a, b).sum() / union)
    worst = min(ious) if ious else 0.0
    return report(bool(ious) and worst >= 0.99, f"{label}: pairs={len(pairs)} min_mask_iou={worst:.4f}")


def compare_keypoints(ours_kp, ref_kp, pairs, label) -> bool:
    xy, conf = 0.0, 0.0
    for i, j in pairs:
        xy = max(xy, float(np.abs(ours_kp[j][:, :2] - ref_kp[i][:, :2]).max()))
        conf = max(conf, float(np.abs(ours_kp[j][:, 2] - ref_kp[i][:, 2]).max()))
    ok = bool(pairs) and xy <= 1.0 and conf <= 0.01
    return report(ok, f"{label}: pairs={len(pairs)} max_kpt_xy_diff_px={xy:.3f} max_kpt_conf_diff={conf:.4f}")


def angle_diff(a, b):
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def compare_obb(ours: dict, ref: dict, label: str) -> bool:
    n_o, n_r = len(ours["xywhr"]), len(ref["xywhr"])
    used, worst_param, worst_angle, worst_corner, worst_score, unmatched = set(), 0.0, 0.0, 0.0, 0.0, 0
    for i in np.argsort(-ref["scores"]):
        cand = [j for j in range(n_o) if j not in used and ours["classes"][j] == ref["classes"][i]]
        if not cand:
            unmatched += 1
            continue
        dist = [np.hypot(*(ours["xywhr"][j][:2] - ref["xywhr"][i][:2])) for j in cand]
        j = cand[int(np.argmin(dist))]
        used.add(j)
        o, r = ours["xywhr"][j], ref["xywhr"][i]
        worst_param = max(worst_param, float(np.abs(o[:4] - r[:4]).max()))
        worst_angle = max(worst_angle, angle_diff(o[4], r[4]))
        co, cr = xywhr_corners(o[None])[0], xywhr_corners(r[None])[0]
        worst_corner = max(worst_corner, float(max(np.linalg.norm(cr - p, axis=1).min() for p in co)))
        worst_score = max(worst_score, abs(float(ours["scores"][j] - ref["scores"][i])))
    params_ok = worst_param <= 1.0 and worst_angle <= 0.01
    ok = n_o == n_r and unmatched == 0 and (params_ok or worst_corner <= 1.0)
    return report(
        ok,
        f"{label}: ours={n_o} ref={n_r} unmatched={unmatched} max_cxcywh_diff_px={worst_param:.3f} "
        f"max_angle_diff_rad={worst_angle:.4f} max_corner_diff_px={worst_corner:.3f} max_score_diff={worst_score:.4f}",
    )


def np_(x):
    return c._np(x)


def reference(result, task: str) -> dict:
    if task == "obb":
        return {
            "xywhr": np_(result.obb.xywhr).reshape(-1, 5).astype(np.float64),
            "scores": np_(result.obb.conf).reshape(-1).astype(np.float64),
            "classes": np_(result.obb.cls).reshape(-1).astype(np.int64),
        }
    ref = c.reference_detections(result)
    if task == "segment":
        ref["masks"] = np_(result.masks.data).astype(bool)
    if task == "pose":
        ref["keypoints"] = np_(result.keypoints.data).astype(np.float64)
    return ref


def compare(ours: dict, ref: dict, task: str, label: str) -> list[bool]:
    if task == "obb":
        return [compare_obb(ours, ref, label)]
    checks = [c.compare_detections(ours, ref, label=label)]
    pairs = match_pairs(ours, ref)
    if task == "segment":
        checks.append(compare_masks(ours["masks"], ref["masks"], pairs, f"{label} masks"))
    if task == "pose":
        checks.append(compare_keypoints(ours["keypoints"], ref["keypoints"], pairs, f"{label} keypoints"))
    return checks


# ---------------------------------------------------------------- run

for task, weights in TASKS:
    model = LibreYOLO(weights)
    refs = {conf: reference(model.predict(SAMPLE_IMAGE, conf=conf), task) for conf in (CONF, LOW_CONF)}
    entry = variant["tasks"][task][0]
    assert entry["when"] == "default"
    path = WORK / f"rfdetr_{task}.onnx"
    model.export(format="onnx", output_path=str(path))
    meta = c.onnx_metadata(path)
    sess = c.session(path)
    assert sess.get_inputs()[0].name == INP["name"], sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    assert names == [o["name"] for o in entry["outputs"]], names
    H, W = c.input_hw(sess, meta)
    assert H == W
    x = preprocess(image, task, H, W)
    outs = dict(zip(names, sess.run(None, {INP["name"]: x})))
    shapes = {k: v.shape for k, v in outs.items()}
    oh, ow = image.shape[:2]
    nc = int(meta["nc"])

    print(f"rfdetr {task} {weights}: input={H}x{W} outputs={shapes}")
    for conf, ref in refs.items():
        if task == "obb":
            ours = decode_obb(outs, ow, oh, conf)
        elif task == "pose":
            ours = decode_pose(outs, meta, ow, oh, conf)
        else:
            ours = decode_detect(outs, nc, ow, oh, conf)
            if task == "segment":
                ours["masks"] = decode_masks(outs, ours["query"], ow, oh)
        results.extend(compare(ours, ref, task, f"rfdetr {task} default conf={conf}"))

    # The batch axis is dynamic: a batch of 2 gives the same first item.
    outs2 = sess.run(None, {INP["name"]: np.concatenate([x, x], 0)})
    diff = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(outs2, outs.values()))
    results.append(report(all(a.shape[0] == 2 for a in outs2) and outs2[0].shape[1] == shapes["dets"][1]
                          and float(np.abs(outs2[0][0] - outs["dets"][0]).max()) < 1e-3,
                          f"rfdetr {task} batch=2 runs, max_output_diff_item0={diff:.2e}"))

    # LibreYOLO's own ONNX backend on the same file, vs the .pt.
    lib = reference(LibreYOLO(str(path)).predict(SAMPLE_IMAGE, conf=CONF), task)
    library.extend(compare(lib, refs[CONF], task, f"(library, not part of the JSON proof) LibreYOLO onnx backend {task}"))
    del sess
    path.unlink()

# Claims in the input notes: square only, divisibility, no nms=True.
for kwargs, expected, label in [
    ({"imgsz": (384, 640)}, NotImplementedError, "non-square imgsz (384, 640) rejected"),
    ({"imgsz": 400}, ValueError, "detect imgsz 400 (not a multiple of 32) rejected"),
    ({"nms": True}, NotImplementedError, "nms=True rejected"),
]:
    path = WORK / "rfdetr_reject.onnx"
    try:
        LibreYOLO("LibreRFDETRn.pt").export(format="onnx", output_path=str(path), **kwargs)
        results.append(report(False, f"{label}: export succeeded"))
    except expected as exc:
        results.append(report(True, f"{label}: {type(exc).__name__}: {str(exc).splitlines()[0][:140]}"))
    path.unlink(missing_ok=True)

print(f"library backend checks: {sum(library)}/{len(library)} match the .pt")
c.finish(results)
