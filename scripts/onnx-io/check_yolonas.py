"""Check src/data/docs/onnx-io/yolonas.json against LibreYOLO predictions.

Usage: python scripts/onnx-io/check_yolonas.py <work_dir> [detect] [pose] [obb]

YOLO-NAS does not use the shared letterbox: each task resizes the longest side
to its own target (636, 640 or 1024), rounds the new size with Python round,
pads with its own value at its own position, and detect/pose resample with
Pillow BILINEAR. The input is built here by ``build_input`` from the JSON
input block and its notes, with numpy and OpenCV only (Pillow's filter comes
from ``_common.resize_to``).

Each task runs the default graph at batch 1 on two square sizes, and at
batch 2 (the sample image and its mirror) on the default size, decoding every
row against its own ``.pt`` reference.
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
TASKS = [a for a in sys.argv[2:] if a in ("detect", "pose", "obb")] or ["detect", "pose", "obb"]
CONF, IOU, MAX_DET = 0.25, 0.45, 300
# DOTA classes on a street photo: nothing reaches the 0.25 default, so the OBB
# check runs both sides at a lower threshold to exercise thresholding, top-k
# and rotated NMS on real candidates.
TASK_CONF = {"detect": CONF, "pose": CONF, "obb": 0.05}
CHECKPOINTS = {
    "detect": "LibreYOLONASs.pt",
    "pose": "LibreYOLONASn-pose.pt",
    "obb": "LibreYOLONASs-obb.pt",
}
# Square sizes to export per task, default first. The second size proves the
# "longest side to min(target, height, width)" rule.
SIZES = {"detect": (640, 512), "pose": (640, 512), "obb": (1024, 768)}

ENTRY = c.load_entry("yolonas")
mirror_path = WORK / "yolonas_mirror.png"  # lossless, so cv2 and LibreYOLO see the same pixels
cv2.imwrite(str(mirror_path), cv2.flip(cv2.imread(str(SAMPLE_IMAGE)), 1))
IMAGES = {
    "sample": (str(SAMPLE_IMAGE), cv2.imread(str(SAMPLE_IMAGE))),
    "mirror": (str(mirror_path), cv2.imread(str(mirror_path))),
}
results = []


def variant_for(task: str) -> dict:
    for variant in ENTRY["variants"]:
        if "yolonas" in variant["keys"] and task in variant["tasks"]:
            return variant
    raise KeyError(f"yolonas.json has no yolonas variant for task {task!r}")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def build_input(image_bgr: np.ndarray, inp: dict, H: int, W: int):
    """Follow the input block and its notes. Returns (CHW tensor, geometry)."""
    notes = " ".join(inp["notes"])
    assert inp["resize"] == "custom", inp["resize"]
    target = int(re.search(r"min\((\d+), height, width\)", notes).group(1))
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if inp["color"] == "RGB" else image_bgr
    oh, ow = img.shape[:2]
    R = min(target, H, W)
    r = min(R / oh, R / ow)
    nw, nh = int(round(ow * r)), int(round(oh * r))
    if (nw, nh) == (ow, oh):
        resized = img
    else:
        # "pillow-bilinear" or an OpenCV name, as the input block says.
        resized = c.resize_to(img, nw, nh, inp["interpolation"])
    if inp["pad_position"] == "center":
        left, top = (W - nw) // 2, (H - nh) // 2
    elif inp["pad_position"] == "topleft":
        left, top = 0, 0
    else:
        raise ValueError(inp["pad_position"])
    canvas = np.full((H, W, 3), inp["pad_value"], np.uint8)
    canvas[top : top + nh, left : left + nw] = resized
    x = canvas.astype(np.float32) / float(inp["scale"])
    assert inp["mean"] is None and inp["std"] is None
    geo = {"orig_w": ow, "orig_h": oh, "r": r, "left": left, "top": top}
    return np.ascontiguousarray(x.transpose(2, 0, 1)), geo


# ---------------------------------------------------------------------------
# Graph contract: names and shapes must match the JSON outputs.
# ---------------------------------------------------------------------------


def check_io(sess, inp, export, H, W, batch, meta, outs) -> bool:
    ok = sess.get_inputs()[0].name == inp["name"]
    names = [o.name for o in sess.get_outputs()]
    ok &= names == [o["name"] for o in export["outputs"]]
    anchors = sum((H // s) * (W // s) for s in (8, 16, 32))
    symbols = {"batch": batch, "anchors": anchors, "nc": int(meta["nc"]), "num_keypoints": 17}
    for spec in export["outputs"]:
        dims = [d.strip() for d in spec["shape"].strip("[]").split(",")]
        want = tuple(int(d) if d.isdigit() else symbols[d] for d in dims)
        got = outs[spec["name"]].shape if spec["name"] in outs else None
        if got != want:
            print(f"  shape mismatch for {spec['name']}: graph {got}, JSON {want}")
            ok = False
    if not ok:
        print(f"  graph outputs {names}, JSON {[o['name'] for o in export['outputs']]}")
    return bool(ok)


def run(path, inp, export, imgsz, keys):
    meta = c.onnx_metadata(path)
    sess = c.session(path)
    H, W = c.input_hw(sess, meta)
    assert (H, W) == (imgsz, imgsz), (H, W)
    built = [build_input(IMAGES[k][1], inp, H, W) for k in keys]
    x = np.stack([b[0] for b in built])
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {inp["name"]: x})))
    io_ok = check_io(sess, inp, export, H, W, len(keys), meta, outs)
    return outs, [b[1] for b in built], io_ok


def nms_raises(name: str, task: str, imgsz: int) -> bool:
    try:
        LibreYOLO(name).export(format="onnx", imgsz=imgsz, nms=True, output_path=str(WORK / "nms.onnx"))
    except NotImplementedError as exc:
        print(f"[PASS] yolonas {task} nms=True raises NotImplementedError: {exc}")
        return True
    print(f"[FAIL] yolonas {task} nms=True exported, but the JSON says it raises")
    (WORK / "nms.onnx").unlink(missing_ok=True)
    return False


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


def decode_detect(outs, geo, b, conf):
    boxes = outs["pred_logits"][b].astype(np.float64)  # [anchors, 4] x1 y1 x2 y2
    scores = outs["pred_boxes"][b]  # [anchors, nc], sigmoid applied
    best, cls = scores.max(1), scores.argmax(1)
    m = best > conf
    boxes, best, cls = boxes[m], best[m], cls[m]
    order = np.argsort(-best, kind="stable")[:1000]
    boxes, best, cls = boxes[order], best[order], cls[order]
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - geo["left"]) / geo["r"]).clip(0, geo["orig_w"])
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - geo["top"]) / geo["r"]).clip(0, geo["orig_h"])
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, best, cls = boxes[valid], best[valid], cls[valid]
    keep = c.class_aware_nms(boxes, best, cls, IOU, MAX_DET)
    return {"boxes": boxes[keep], "scores": best[keep], "classes": cls[keep]}


def compare_detect(ours, ref, label) -> bool:
    return c.compare_detections(ours, ref, label=label)


# ---------------------------------------------------------------------------
# pose
# ---------------------------------------------------------------------------


def decode_pose(outs, geo, b, conf):
    boxes = outs["boxes"][b].astype(np.float64)  # [anchors, 4] x1 y1 x2 y2
    scores = outs["scores"][b]  # [anchors, nc], sigmoid applied
    kxy = outs["keypoints_xy"][b].astype(np.float64)  # [anchors, K, 2]
    kconf = outs["keypoints_conf"][b].astype(np.float64)  # [anchors, K]
    best, cls = scores.max(1), scores.argmax(1)
    m = best >= conf
    boxes, best, cls, kxy, kconf = boxes[m], best[m], cls[m], kxy[m], kconf[m]
    order = np.argsort(-best, kind="stable")[:1000]
    boxes, best, cls, kxy, kconf = boxes[order], best[order], cls[order], kxy[order], kconf[order]
    # Image sits in the top-left corner: no pad offset, divide by r.
    boxes = boxes / geo["r"]
    kxy = kxy / geo["r"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, geo["orig_w"])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, geo["orig_h"])
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, best, cls, kxy, kconf = boxes[valid], best[valid], cls[valid], kxy[valid], kconf[valid]
    keep = c.class_aware_nms(boxes, best, cls, IOU, MAX_DET)
    return {
        "boxes": boxes[keep], "scores": best[keep], "classes": cls[keep],
        "keypoints": np.concatenate([kxy[keep], kconf[keep][..., None]], -1),
    }


def compare_pose(ours, ref, label) -> bool:
    """Detections as usual, then keypoints of each best same-class IoU pair."""
    ok = c.compare_detections(ours, ref, label=label)
    if len(ref["boxes"]) == 0 or len(ours["boxes"]) != len(ref["boxes"]):
        print(f"[{'PASS' if ok else 'FAIL'}] {label} keypoints: pairs={len(ref['boxes'])}")
        return ok
    ious = c.iou_matrix(ref["boxes"], ours["boxes"])
    xy, vis = 0.0, 0.0
    for i in range(len(ref["boxes"])):
        j = int(np.argmax(np.where(ours["classes"] == ref["classes"][i], ious[i], -1)))
        xy = max(xy, float(np.abs(ours["keypoints"][j, :, :2] - ref["keypoints"][i, :, :2]).max()))
        vis = max(vis, float(np.abs(ours["keypoints"][j, :, 2] - ref["keypoints"][i, :, 2]).max()))
    kp_ok = xy <= 1.0 and vis <= 1e-3
    print(
        f"[{'PASS' if kp_ok else 'FAIL'}] {label} keypoints: "
        f"max_keypoint_xy_diff_px={xy:.3f} max_keypoint_conf_diff={vis:.5f}"
    )
    return ok and kp_ok


def reference_pose(res) -> dict:
    ref = c.reference_detections(res)
    ref["keypoints"] = c._np(res.keypoints.data).astype(np.float64).reshape(-1, 17, 3)
    return ref


# ---------------------------------------------------------------------------
# obb
# ---------------------------------------------------------------------------


def rotated_iou(a, b) -> float:
    area_a, area_b = float(a[2] * a[3]), float(b[2] * b[3])
    if area_a <= 0 or area_b <= 0:
        return 0.0
    ra = ((float(a[0]), float(a[1])), (float(a[2]), float(a[3])), math.degrees(float(a[4])))
    rb = ((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), math.degrees(float(b[4])))
    _, pts = cv2.rotatedRectangleIntersection(ra, rb)
    if pts is None:
        return 0.0
    inter = float(cv2.contourArea(pts.astype(np.float32)))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def corners(xywhr):
    cx, cy, w, h, a = xywhr
    wv = np.array([math.cos(a), math.sin(a)]) * w / 2
    hv = np.array([-math.sin(a), math.cos(a)]) * h / 2
    ctr = np.array([cx, cy])
    return np.stack([ctr - wv - hv, ctr + wv - hv, ctr + wv + hv, ctr - wv + hv])


def decode_obb(outs, geo, b, conf):
    boxes = outs["boxes"][b].astype(np.float64)  # [anchors, 5] cx cy w h angle
    scores = outs["scores"][b]  # [anchors, nc], sigmoid applied
    best, cls = scores.max(1), scores.argmax(1)
    m = best >= conf
    boxes, best, cls = boxes[m], best[m], cls[m]
    order = np.argsort(-best, kind="stable")[:1000]
    boxes, best, cls = boxes[order], best[order], cls[order]
    # Image sits in the top-left corner: no pad offset, the angle is unchanged.
    boxes[:, 0] = ((boxes[:, 0] - geo["left"]) / geo["r"]).clip(0, geo["orig_w"])
    boxes[:, 1] = ((boxes[:, 1] - geo["top"]) / geo["r"]).clip(0, geo["orig_h"])
    boxes[:, 2:4] /= geo["r"]
    valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    boxes, best, cls = boxes[valid], best[valid], cls[valid]
    # Long side first, angle in [-pi/2, pi/2).
    swap = boxes[:, 3] > boxes[:, 2]
    boxes[swap, 2], boxes[swap, 3] = boxes[swap, 3].copy(), boxes[swap, 2].copy()
    boxes[swap, 4] += math.pi / 2
    boxes[:, 4] = np.mod(boxes[:, 4] + math.pi / 2, math.pi) - math.pi / 2
    # Greedy per-class NMS on exact rotated IoU.
    pending = list(np.argsort(-best, kind="stable"))
    keep = []
    while pending and len(keep) < MAX_DET:
        i = pending.pop(0)
        keep.append(i)
        pending = [j for j in pending if cls[j] != cls[i] or rotated_iou(boxes[i], boxes[j]) <= IOU]
    keep = np.asarray(keep, np.int64)
    return {"xywhr": boxes[keep], "scores": best[keep], "classes": cls[keep]}


def compare_obb(ours, ref, label) -> bool:
    n_ours, n_ref = len(ours["xywhr"]), len(ref["xywhr"])
    used, unmatched = set(), 0
    worst_xywh, worst_angle, worst_corner, worst_score = 0.0, 0.0, 0.0, 0.0
    for i in np.argsort(-ref["scores"]):
        cand = [j for j in range(n_ours) if j not in used and ours["classes"][j] == ref["classes"][i]]
        if not cand:
            unmatched += 1
            continue
        j = max(cand, key=lambda k: rotated_iou(ours["xywhr"][k], ref["xywhr"][i]))
        if rotated_iou(ours["xywhr"][j], ref["xywhr"][i]) < 0.5:
            unmatched += 1
            continue
        used.add(j)
        a, b = ours["xywhr"][j], ref["xywhr"][i]
        d_xywh = float(np.abs(a[:4] - b[:4]).max())
        d_angle = abs((a[4] - b[4] + math.pi / 2) % math.pi - math.pi / 2)
        # Our corners follow the JSON's angle convention. LibreYOLO's own
        # xyxyxyxy proves the convention, up to the starting corner.
        d_corner = min(np.abs(np.roll(corners(a), k, 0) - ref["corners"][i]).max() for k in range(4))
        # Near-square boxes can canonicalize either way; then corners decide.
        if (d_xywh <= 1.0 and d_angle <= 0.01) or d_corner > 1.0:
            worst_xywh, worst_angle = max(worst_xywh, d_xywh), max(worst_angle, d_angle)
        worst_corner = max(worst_corner, float(d_corner))
        worst_score = max(worst_score, abs(float(ours["scores"][j] - ref["scores"][i])))
    ok = n_ours == n_ref and unmatched == 0 and worst_xywh <= 1.0 and worst_angle <= 0.01 and worst_corner <= 1.0
    print(
        f"[{'PASS' if ok else 'FAIL'}] {label}: ours={n_ours} ref={n_ref} unmatched={unmatched} "
        f"max_xywh_diff_px={worst_xywh:.3f} max_angle_diff_rad={worst_angle:.5f} "
        f"max_corner_diff_px={worst_corner:.3f} max_score_diff={worst_score:.4f}"
    )
    return ok


def reference_obb(res) -> dict:
    return {
        "xywhr": c._np(res.obb.xywhr).reshape(-1, 5).astype(np.float64),
        "scores": c._np(res.obb.conf).reshape(-1).astype(np.float64),
        "classes": c._np(res.obb.cls).reshape(-1).astype(np.int64),
        "corners": c._np(res.obb.xyxyxyxy).reshape(-1, 4, 2).astype(np.float64),
    }


def obb_angle_ok(outs) -> bool:
    """The JSON says the raw angle is (sigmoid - 0.25) x pi."""
    angle = outs["boxes"][..., 4]
    ok = bool(angle.min() >= -math.pi / 4 - 1e-5 and angle.max() <= 3 * math.pi / 4 + 1e-5)
    if not ok:
        print(f"  raw angle range {angle.min():.4f} to {angle.max():.4f} is outside -pi/4 to 3pi/4")
    return ok


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

TASK_FUNCS = {
    "detect": (decode_detect, compare_detect, c.reference_detections),
    "pose": (decode_pose, compare_pose, reference_pose),
    "obb": (decode_obb, compare_obb, reference_obb),
}


def check_task(task: str) -> None:
    decode, compare, reference = TASK_FUNCS[task]
    variant = variant_for(task)
    exports = variant["tasks"][task]
    assert [e["when"] for e in exports] == ["default"]
    name, conf = CHECKPOINTS[task], TASK_CONF[task]
    pt = LibreYOLO(name)
    refs = {}

    def ref_for(key, imgsz):
        if (key, imgsz) not in refs:
            refs[key, imgsz] = reference(pt.predict(IMAGES[key][0], imgsz=imgsz, conf=conf, iou=IOU))
        return refs[key, imgsz]

    for imgsz in SIZES[task]:
        default_size = imgsz == SIZES[task][0]
        path = WORK / f"yolonas_{task}_{imgsz}.onnx"
        LibreYOLO(name).export(format="onnx", imgsz=imgsz, output_path=str(path))
        for keys in [("sample",)] + ([("sample", "mirror")] if default_size else []):
            try:
                outs, geos, io_ok = run(path, variant["input"], exports[0], imgsz, keys)
            except Exception as exc:  # a batch the graph cannot take
                print(f"[FAIL] yolonas {task} default imgsz={imgsz} batch={len(keys)}: {type(exc).__name__}: {exc}")
                results.append(False)
                continue
            if task == "obb":
                io_ok &= obb_angle_ok(outs)
            for b, key in enumerate(keys):
                label = f"yolonas {task} default imgsz={imgsz} batch={len(keys)} row={b} ({key}) conf={conf}"
                if not io_ok:
                    print(f"[FAIL] {label}: graph names, shapes or ranges differ from the JSON")
                ok = compare(decode(outs, geos[b], b, conf), ref_for(key, imgsz), label)
                results.append(bool(ok and io_ok))
        if default_size:
            lib = reference(LibreYOLO(str(path)).predict(SAMPLE_IMAGE, conf=conf, iou=IOU))
            label = f"LibreYOLO(onnx).predict vs .pt {task} imgsz={imgsz} conf={conf}"
            results.append(bool(compare(lib, ref_for("sample", imgsz), label)))
        path.unlink()
    assert "nms=True is not available" in " ".join(exports[0]["postprocess"])
    results.append(nms_raises(name, task, SIZES[task][0]))


for task in TASKS:
    check_task(task)
mirror_path.unlink()
c.finish(results)
