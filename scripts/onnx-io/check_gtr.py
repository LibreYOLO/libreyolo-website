"""Check src/data/docs/onnx-io/gtr.json against LibreYOLO predictions.

Usage: python scripts/onnx-io/check_gtr.py <work_dir> [--tasks detect,segment,pose,obb,depth,semantic]

GTR has four input contracts, so gtr.json has four variants that share the
registry key "gtr": detection, segmentation and pose (Pillow stretch, ImageNet
normalization), OBB (top-left letterbox with round() sizes, pad 0, ImageNet),
depth (Pillow stretch, normalization in the graph) and semantic (OpenCV
top-left letterbox on a 1024 x 2048 canvas, normalization in the graph).

The input is built from the JSON input block with numpy and OpenCV only
(_common.preprocess, plus preprocess_obb below for the round() letterbox), and
the outputs are decoded as the JSON postprocess steps say. LibreYOLO is used for
export and for the reference .predict on the .pt only.

"pillow-bilinear" is checked twice. The main runs use Pillow's fixed-point
arithmetic, as the JSON note spells it out (pillow_fixed_point below, bit-exact
with Pillow 13). Then _common's float weights are run at the default threshold,
where the JSON says they are good enough.

Besides the JSON checks, the script runs LibreYOLO("x.onnx").predict against the
.pt (library parity, reported apart, does not change the exit code) and exports
each task at batch=2 to check that both images decode like two batch-1 runs.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

from libreyolo import SAMPLE_IMAGE, LibreYOLO  # noqa: E402  (export + reference only)

parser = argparse.ArgumentParser()
parser.add_argument("work", nargs="?", default=".")
parser.add_argument("--tasks", default="detect,segment,pose,obb,depth,semantic")
args = parser.parse_args()
WORK = Path(args.work).resolve()
WORK.mkdir(parents=True, exist_ok=True)
CONF, IOU, MAX_DET = 0.25, 0.45, 300
CHECKPOINTS = {
    "detect": "LibreGTRs.pt",
    "segment": "LibreGTRs-seg.pt",
    "pose": "LibreGTRs-pose.pt",
    "obb": "LibreGTRs-obb.pt",
    "depth": "LibreGTRs-depth.pt",
    "semantic": "LibreGTRs-sem.pt",
}

ENTRY = c.load_entry("gtr")
IMAGE = cv2.imread(str(SAMPLE_IMAGE))
FLIPPED = np.ascontiguousarray(IMAGE[:, ::-1])  # second image of the batch-2 run
OH, OW = IMAGE.shape[:2]
results = []  # JSON checks: these decide the exit code
lib_results = []  # LibreYOLO("x.onnx").predict vs the .pt, reported apart


def record(ok, bucket=results):
    bucket.append(bool(ok))
    return bool(ok)


def say(ok, label, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def variant_for(task):
    found = [v for v in ENTRY["variants"] if "gtr" in v["keys"] and task in v["tasks"]]
    assert len(found) == 1, f"expected one gtr variant for {task}, got {len(found)}"
    return found[0]


# ---------------------------------------------------------------- graph ----


def export(task, path, **kw):
    LibreYOLO(CHECKPOINTS[task]).export(format="onnx", output_path=str(path), **kw)
    return c.onnx_metadata(path), c.session(path)


def assert_rejected(task, label, **kw):
    path = WORK / "rejected.onnx"
    try:
        LibreYOLO(CHECKPOINTS[task]).export(format="onnx", output_path=str(path), **kw)
    except (NotImplementedError, ValueError) as exc:
        return say(True, label, f"export rejected ({type(exc).__name__}: {str(exc)[:100]})")
    path.unlink(missing_ok=True)
    return say(False, label, "export was accepted")


def check_io(sess, variant, spec, symbols, label):
    """The graph's input and output names and static shapes must equal the JSON's."""
    def dims(shape):
        return [eval(d, {"__builtins__": {}}, symbols) for d in shape.strip("[]").split(",")]

    inp = sess.get_inputs()[0]
    ok = inp.name == variant["input"]["name"] and list(inp.shape) == dims(variant["input"]["shape"])
    ok &= [o.name for o in sess.get_outputs()] == [o["name"] for o in spec["outputs"]]
    for want, out in zip(spec["outputs"], sess.get_outputs()):
        ok &= list(out.shape) == dims(want["shape"])
    graph = [(o.name, o.shape) for o in [inp, *sess.get_outputs()]]
    return say(ok, label, f"graph {graph} {'matches' if ok else 'differs from'} the JSON")


def run(sess, x):
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, {sess.get_inputs()[0].name: x})))


def element(outs, b):
    return {k: v[b] for k, v in outs.items()}


# --------------------------------------------------------------- decode ----


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def cxcywh_to_xyxy(b):
    b = np.asarray(b, np.float64)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def topk_pairs(logits, k=MAX_DET):
    """Top-k of sigmoid scores over all queries x classes: (query, class, score)."""
    flat = sigmoid(logits).reshape(-1)
    nc = logits.shape[-1]
    idx = np.argsort(-flat, kind="stable")[: min(k, flat.size)]
    return idx // nc, idx % nc, flat[idx]


def decode_detect(o, ow, oh, conf):
    """o is one batch element: pred_logits [queries, nc], pred_boxes [queries, 4]."""
    q, cls, score = topk_pairs(o["pred_logits"])
    keep = score > conf
    boxes = cxcywh_to_xyxy(o["pred_boxes"][q[keep]])
    boxes[:, [0, 2]] *= ow
    boxes[:, [1, 3]] *= oh
    return {"boxes": boxes, "scores": score[keep], "classes": cls[keep], "query": q[keep]}


def decode_segment(o, ow, oh, conf):
    det = decode_detect(o, ow, oh, conf)
    det["masks"] = np.stack([
        cv2.resize(o["pred_masks"][q].astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR) > 0
        for q in det["query"]
    ]) if len(det["query"]) else np.zeros((0, oh, ow), bool)
    return det


def decode_pose(o, ow, oh, conf, num_keypoints):
    score = sigmoid(o["pred_logits"][:, 1])
    order = np.argsort(-score, kind="stable")[:MAX_DET]
    order = order[score[order] >= conf]
    xy = o["pred_keypoints"][order].reshape(-1, num_keypoints, 2).astype(np.float64)
    xy[..., 0] *= ow
    xy[..., 1] *= oh
    kpts = np.concatenate([xy, np.ones(xy.shape[:2] + (1,))], -1)  # visibility 1.0
    boxes = np.concatenate([xy.min(1), xy.max(1)], 1) if len(order) else np.zeros((0, 4))
    return {"boxes": boxes, "scores": score[order], "classes": np.zeros(len(order), np.int64), "kpts": kpts}


def corners(xywhr):
    cx, cy, w, h, a = (xywhr[:, i] for i in range(5))
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append(np.stack([cx + dx * np.cos(a) - dy * np.sin(a), cy + dx * np.sin(a) + dy * np.cos(a)], 1))
    return np.stack(pts, 1)


def decode_obb(o, r, H, W, conf):
    q, cls, score = topk_pairs(o["pred_logits"])
    keep = score > conf
    b = o["pred_boxes"][q[keep]].astype(np.float64)
    xywh = b[:, :4] * np.array([W, H, W, H]) / r
    angle = b[:, 4] * math.pi
    ext_x = np.abs(np.cos(angle)) * xywh[:, 2] / 2 + np.abs(np.sin(angle)) * xywh[:, 3] / 2
    ext_y = np.abs(np.sin(angle)) * xywh[:, 2] / 2 + np.abs(np.cos(angle)) * xywh[:, 3] / 2
    boxes = np.stack([xywh[:, 0] - ext_x, xywh[:, 1] - ext_y, xywh[:, 0] + ext_x, xywh[:, 1] + ext_y], 1)
    xywhr = np.concatenate([xywh, angle[:, None]], 1)
    return {"boxes": boxes, "scores": score[keep], "classes": cls[keep], "xywhr": xywhr, "corners": corners(xywhr)}


# -------------------------------------------------------------- compare ----


def from_result(result, task):
    """A LibreYOLO Results object in the same dict layout as the decoders."""
    d = c.reference_detections(result)
    if task == "segment":
        d["masks"] = c._np(result.masks.data).astype(bool).reshape(-1, OH, OW)
    if task == "pose":
        d["kpts"] = c._np(result.keypoints.data).astype(np.float64)
    if task == "obb":
        d["xywhr"] = c._np(result.obb.xywhr).astype(np.float64)
        d["corners"] = c._np(result.obb.xyxyxyxy).astype(np.float64)
    return d


def match(ours, ref):
    """Greedy same-class IoU matching, as in _common.compare_detections."""
    pairs, used = [], set()
    if not len(ours["boxes"]) or not len(ref["boxes"]):
        return pairs
    ious = c.iou_matrix(ref["boxes"], ours["boxes"])
    for i in np.argsort(-ref["scores"]):
        cand = [j for j in np.argsort(-ious[i]) if j not in used and ours["classes"][j] == ref["classes"][i]]
        if cand and ious[i, cand[0]] >= 0.5:
            used.add(cand[0])
            pairs.append((i, cand[0]))
    return pairs


def compare(ours, ref, task, label):
    """Boxes per _common.compare_detections, plus the task's extra bar."""
    ok = c.compare_detections(ours, ref, label=label)
    pairs = match(ours, ref)
    full = len(pairs) == len(ref["boxes"])
    if task == "segment":
        ious = [(ref["masks"][i] & ours["masks"][j]).sum() / max((ref["masks"][i] | ours["masks"][j]).sum(), 1) for i, j in pairs]
        worst = min(ious, default=1.0 if full else 0.0)
        ok &= say(full and worst >= 0.99, f"{label} masks", f"matched={len(pairs)}/{len(ref['boxes'])} min_mask_iou={worst:.5f}")
    if task == "pose":
        xy = max([np.abs(ours["kpts"][j, :, :2] - ref["kpts"][i, :, :2]).max() for i, j in pairs], default=0.0)
        vis = max([np.abs(ours["kpts"][j, :, 2] - ref["kpts"][i, :, 2]).max() for i, j in pairs], default=0.0)
        ok &= say(full and xy <= 1.0 and vis <= 1e-6, f"{label} keypoints",
                  f"matched={len(pairs)}/{len(ref['boxes'])} max_xy_diff_px={xy:.4f} max_visibility_diff={vis:.4f}")
    if task == "obb":
        d_xywh = max([np.abs(ours["xywhr"][j, :4] - ref["xywhr"][i, :4]).max() for i, j in pairs], default=0.0)
        d_ang = max([abs(ours["xywhr"][j, 4] - ref["xywhr"][i, 4]) for i, j in pairs], default=0.0)
        d_cor = max([np.abs(ours["corners"][j] - ref["corners"][i]).max() for i, j in pairs], default=0.0)
        ok &= say(full and d_xywh <= 1 and d_ang <= 0.01 and d_cor <= 1, f"{label} rotated",
                  f"matched={len(pairs)}/{len(ref['boxes'])} max_cxcywh_diff_px={d_xywh:.4f} "
                  f"max_angle_diff_rad={d_ang:.5f} max_corner_diff_px={d_cor:.4f}")
    return ok


# ---------------------------------------------------------------- input ----

_FLOAT_RESIZE = c.resize_to


def _fixed_point_pass(img, out_size, axis):
    """One Pillow pass: weights * 2^22 rounded half away from zero, integer sum
    starting at 2^21, shift right by 22, clip to 0..255."""
    src = np.moveaxis(img, axis, 0).astype(np.int64)
    weights = c.pillow_bilinear_weights(src.shape[0], out_size)
    fixed = np.trunc(weights * (1 << 22) + np.where(weights < 0, -0.5, 0.5)).astype(np.int64)
    fixed[weights == 0] = 0
    acc = np.tensordot(fixed, src, axes=(1, 0)) + (1 << 21)
    return np.moveaxis(np.clip(acc >> 22, 0, 255).astype(np.uint8), 0, axis)


def pillow_fixed_point(img, out_w, out_h):
    """Image.resize((out_w, out_h), BILINEAR) on uint8 HWC: width pass, then height."""
    if out_w != img.shape[1]:
        img = _fixed_point_pass(img, out_w, 1)
    if out_h != img.shape[0]:
        img = _fixed_point_pass(img, out_h, 0)
    return img


def use_pillow(mode):
    """Route _common's "pillow-bilinear" to the fixed-point form or its float weights."""
    def resize_to(img, out_w, out_h, interpolation):
        if interpolation == "pillow-bilinear" and mode == "fixed":
            return pillow_fixed_point(img, out_w, out_h)
        return _FLOAT_RESIZE(img, out_w, out_h, interpolation)

    c.resize_to = resize_to


use_pillow("fixed")


def preprocess_obb(image_bgr, inp, H, W):
    """Top-left letterbox with Python round() sizes, Pillow bilinear, pad, then normalize."""
    assert inp["resize"] == "custom" and inp["pad_position"] == "topleft"
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if inp["color"] == "RGB" else image_bgr
    oh, ow = rgb.shape[:2]
    r = min(H / oh, W / ow)
    nw, nh = max(1, int(round(ow * r))), max(1, int(round(oh * r)))
    canvas = np.full((H, W, 3), inp["pad_value"], np.uint8)
    canvas[:nh, :nw] = c.resize_to(rgb, nw, nh, inp["interpolation"])
    x = canvas.astype(np.float32) / float(inp["scale"])
    x = (x - np.asarray(inp["mean"], np.float32)) / np.asarray(inp["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r


def make_input(task, variant, image_bgr, H, W, meta):
    """Returns (tensor, geometry) where geometry is what the decoder needs."""
    if task == "obb":
        x, r = preprocess_obb(image_bgr, variant["input"], H, W)
        return x, {"r": r}
    x, geo = c.preprocess(image_bgr, variant["input"], H, W, meta)
    return x, geo


# ------------------------------------------------------- instance tasks ----


QUERIES = {"detect": 300, "segment": 300, "pose": 60, "obb": 300}


def instance_decoder(task, meta, H, W):
    K = int(meta.get("num_keypoints") or 0)

    def decode(o, geo, conf):
        if task == "detect":
            return decode_detect(o, OW, OH, conf)
        if task == "segment":
            return decode_segment(o, OW, OH, conf)
        if task == "pose":
            return decode_pose(o, OW, OH, conf, K)
        return decode_obb(o, geo["r"], H, W, conf)

    return decode


def run_instance(task):
    variant = variant_for(task)
    spec = variant["tasks"][task]
    assert [e["when"] for e in spec] == ["default"], "GTR has no nms=True export"
    spec = spec[0]
    path = WORK / f"gtrs_{task}.onnx"
    meta, sess = export(task, path)
    H, W = c.input_hw(sess, meta)
    symbols = {"batch": 1, "height": H, "width": W, "queries": QUERIES[task], "nc": int(meta["nc"]),
               "num_keypoints": int(meta.get("num_keypoints") or 0), "mask_h": H // 4, "mask_w": W // 4}
    record(check_io(sess, variant, spec, symbols, f"gtr {task} io"))
    decode = instance_decoder(task, meta, H, W)
    x, geo = make_input(task, variant, IMAGE, H, W, meta)
    outs = run(sess, x)
    model = LibreYOLO(CHECKPOINTS[task])
    # Lower thresholds compare more boxes. The sample photo is not aerial, so
    # GTR-S OBB finds a single object at 0.25.
    confs = {"detect": (CONF, 0.05), "obb": (CONF, 0.05, 0.01)}.get(task, (CONF,))
    for conf in confs:
        ref = from_result(model.predict(SAMPLE_IMAGE, conf=conf, iou=IOU), task)
        ours = decode(element(outs, 0), geo, conf)
        record(compare(ours, ref, task, f"gtr {task} conf={conf} input={H}x{W}"))
        if task == "detect" and conf == 0.05:
            out = ((ours["boxes"][:, :2] < 0) | (ours["boxes"][:, 2:] > [OW, OH])).any(1).sum()
            print(f"[INFO] gtr detect conf=0.05: {out} of {len(ours['boxes'])} boxes reach outside the image and are not clipped")

    ref = from_result(model.predict(SAMPLE_IMAGE, conf=CONF, iou=IOU), task)
    use_pillow("float")
    xf, geof = make_input(task, variant, IMAGE, H, W, meta)
    use_pillow("fixed")
    record(compare(decode(element(run(sess, xf), 0), geof, CONF), ref, task,
                   f"gtr {task} float Pillow weights conf={CONF}"))
    if task == "obb":
        # JSON note: at conf 0.01 the float weights add two borderline OBB boxes.
        n_ref = len(from_result(model.predict(SAMPLE_IMAGE, conf=0.01, iou=IOU), task)["boxes"])
        n_float = len(decode(element(run(sess, xf), 0), geof, 0.01)["boxes"])
        record(say(n_float == n_ref + 2, "gtr obb float Pillow weights conf=0.01 (JSON note)",
                   f"{n_float} boxes instead of {n_ref}"))
    if task == "detect":
        # JSON note: OpenCV INTER_LINEAR is not a drop-in, it turns 8 detections into 10.
        xcv, geocv = c.preprocess(IMAGE, {**variant["input"], "interpolation": "bilinear"}, H, W, meta)
        n_cv = len(decode(element(run(sess, xcv), 0), geocv, CONF)["boxes"])
        record(say(len(ref["boxes"]) == 8 and n_cv == 10, "gtr detect cv2 INTER_LINEAR input (JSON note)",
                   f"{n_cv} detections instead of {len(ref['boxes'])}"))
    got = from_result(LibreYOLO(str(path)).predict(SAMPLE_IMAGE, conf=CONF, iou=IOU), task)
    record(compare(got, ref, task, f"gtr {task} library parity LibreYOLO(onnx) vs .pt"), lib_results)

    # Batch 2: export with batch=2, feed the image and its mirror, and decode
    # each element like a batch-1 run of the same image.
    path2 = WORK / f"gtrs_{task}_b2.onnx"
    try:
        meta2, sess2 = export(task, path2, batch=2)
        x2, geo2 = make_input(task, variant, FLIPPED, H, W, meta)
        outs2 = run(sess2, np.concatenate([x, x2]))
        single2 = run(sess, x2)
        record(check_io(sess2, variant, spec, {**symbols, "batch": 2}, f"gtr {task} batch=2 io"))
        for b, (g, single) in enumerate(((geo, outs), (geo2, single2))):
            record(compare(decode(element(outs2, b), g, CONF), decode(element(single, 0), g, CONF), task,
                           f"gtr {task} batch=2 element {b} vs batch 1"))
    except Exception as exc:  # report, keep going
        record(say(False, f"gtr {task} batch=2", f"{type(exc).__name__}: {str(exc)[:300]}"))
    path2.unlink(missing_ok=True)
    path.unlink()
    if task == "detect":
        record(assert_rejected(task, "gtr detect nms=True", nms=True))
    if task in ("detect", "obb"):
        record(assert_rejected(task, f"gtr {task} imgsz=(384, 640), square input only", imgsz=(384, 640)))


# ---------------------------------------------------------------- depth ----


def resize_align_corners(m, oh, ow):
    """Bilinear resize with align_corners=True: source = dest * (in - 1) / (out - 1)."""
    ih, iw = m.shape
    ys = np.arange(oh) * ((ih - 1) / (oh - 1) if oh > 1 else 0.0)
    xs = np.arange(ow) * ((iw - 1) / (ow - 1) if ow > 1 else 0.0)
    y0 = np.floor(ys).astype(int).clip(0, ih - 1)
    x0 = np.floor(xs).astype(int).clip(0, iw - 1)
    y1, x1 = (y0 + 1).clip(max=ih - 1), (x0 + 1).clip(max=iw - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    m = m.astype(np.float64)
    top = m[y0][:, x0] * (1 - wx) + m[y0][:, x1] * wx
    bottom = m[y1][:, x0] * (1 - wx) + m[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def depth_stats(ours, ref):
    rel = np.abs(ours - ref) / np.abs(ref)
    return rel.max(), np.median(rel)


def run_depth():
    variant = variant_for("depth")
    spec = variant["tasks"]["depth"][0]
    path = WORK / "gtrs_depth.onnx"
    meta, sess = export("depth", path)
    H, W = c.input_hw(sess, meta)
    record(check_io(sess, variant, spec, {"batch": 1, "height": H, "width": W}, "gtr depth io"))
    x, _ = c.preprocess(IMAGE, variant["input"], H, W, meta)
    ours = resize_align_corners(run(sess, x)["depth"][0, 0], OH, OW)
    ref = c._np(LibreYOLO(CHECKPOINTS["depth"]).predict(SAMPLE_IMAGE).depth_map.data).astype(np.float64)
    mx, med = depth_stats(ours, ref)
    record(say(mx <= 1e-3, f"gtr depth input={H}x{W}",
               f"max_rel_diff={mx:.2e} median_rel_diff={med:.2e} (bar: max relative diff <= 1e-3 at {OW}x{OH})"))

    use_pillow("float")
    xf, _ = c.preprocess(IMAGE, variant["input"], H, W, meta)
    use_pillow("fixed")
    mx_f, med_f = depth_stats(resize_align_corners(run(sess, xf)["depth"][0, 0], OH, OW), ref)
    record(say(mx_f <= 1e-3, "gtr depth float Pillow weights", f"max_rel_diff={mx_f:.2e} median_rel_diff={med_f:.2e}"))

    # What a reader gets with OpenCV instead of Pillow.
    cv_inp = {**variant["input"], "interpolation": "bilinear"}
    xcv, _ = c.preprocess(IMAGE, cv_inp, H, W, meta)
    mx_cv, med_cv = depth_stats(resize_align_corners(run(sess, xcv)["depth"][0, 0], OH, OW), ref)
    print(f"[INFO] gtr depth with cv2 INTER_LINEAR input instead: max_rel_diff={mx_cv:.2e} median_rel_diff={med_cv:.2e}")

    got = c._np(LibreYOLO(str(path)).predict(SAMPLE_IMAGE).depth_map.data).astype(np.float64)
    mx_lib, med_lib = depth_stats(got, ref)
    record(say(mx_lib <= 1e-3, "gtr depth library parity LibreYOLO(onnx) vs .pt",
               f"max_rel_diff={mx_lib:.2e} median_rel_diff={med_lib:.2e}"), lib_results)
    path.unlink()
    record(assert_rejected("depth", "gtr depth batch=2, batch size 1 only", batch=2))
    record(assert_rejected("depth", "gtr depth imgsz=(384, 640), square input only", imgsz=(384, 640)))


# ------------------------------------------------------------- semantic ----


def decode_semantic(logits, geo):
    """logits [nc, height, width]: crop the image area, resize to the original, argmax."""
    vh = min(logits.shape[1], max(1, int(round(OH * geo["scale_y"]))))
    vw = min(logits.shape[2], max(1, int(round(OW * geo["scale_x"]))))
    crop = np.ascontiguousarray(logits[:, :vh, :vw].transpose(1, 2, 0), dtype=np.float32)
    resized = cv2.resize(crop, (OW, OH), interpolation=cv2.INTER_LINEAR)
    return resized.argmax(-1)


def run_semantic():
    variant = variant_for("semantic")
    spec = variant["tasks"]["semantic"][0]
    path = WORK / "gtrs_sem.onnx"
    meta, sess = export("semantic", path)
    H, W = c.input_hw(sess, meta)
    symbols = {"batch": 1, "height": H, "width": W, "nc": int(meta["nc"])}
    record(check_io(sess, variant, spec, symbols, "gtr semantic io"))
    x, geo = c.preprocess(IMAGE, variant["input"], H, W, meta)
    outs = run(sess, x)
    ours = decode_semantic(outs["semantic_logits"][0], geo)
    ref = c._np(LibreYOLO(CHECKPOINTS["semantic"]).predict(SAMPLE_IMAGE).semantic_mask.data).reshape(OH, OW)
    agree = (ours == ref).mean()
    record(say(agree >= 0.999, f"gtr semantic input={H}x{W} (height before width)",
               f"pixel_label_agreement={agree * 100:.4f}% (bar: >= 99.9 % at {OW}x{OH})"))
    # JSON note: Pillow input instead of OpenCV drops the agreement below 99.7 %.
    xp, _ = c.preprocess(IMAGE, {**variant["input"], "interpolation": "pillow-bilinear"}, H, W, meta)
    agree_p = (decode_semantic(run(sess, xp)["semantic_logits"][0], geo) == ref).mean()
    record(say(agree_p < 0.997, "gtr semantic Pillow input instead (JSON note)",
               f"pixel_label_agreement={agree_p * 100:.4f}%"))
    got = c._np(LibreYOLO(str(path)).predict(SAMPLE_IMAGE).semantic_mask.data).reshape(OH, OW)
    lib_agree = (got == ref).mean()
    record(say(lib_agree >= 0.999, "gtr semantic library parity LibreYOLO(onnx) vs .pt",
               f"pixel_label_agreement={lib_agree * 100:.4f}%"), lib_results)

    path2 = WORK / "gtrs_sem_b2.onnx"
    try:
        _, sess2 = export("semantic", path2, batch=2)
        x2, geo2 = c.preprocess(FLIPPED, variant["input"], H, W, meta)
        outs2 = run(sess2, np.concatenate([x, x2]))
        record(check_io(sess2, variant, spec, {**symbols, "batch": 2}, "gtr semantic batch=2 io"))
        single2 = run(sess, x2)["semantic_logits"][0]
        for b, (g, single) in enumerate(((geo, outs["semantic_logits"][0]), (geo2, single2))):
            a = (decode_semantic(outs2["semantic_logits"][b], g) == decode_semantic(single, g)).mean()
            record(say(a >= 0.999, f"gtr semantic batch=2 element {b} vs batch 1", f"pixel_label_agreement={a * 100:.4f}%"))
    except Exception as exc:
        record(say(False, "gtr semantic batch=2", f"{type(exc).__name__}: {str(exc)[:300]}"))
    path2.unlink(missing_ok=True)
    path.unlink()


def check_float_weights():
    """JSON note: float weights differ from Pillow by a level or two on up to about 1 % of pixels."""
    rgb = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)
    worst_level, worst_share = 0, 0.0
    for w, h in ((640, 640), (1024, 682)):  # the detection stretch and the OBB letterbox
        diff = np.abs(_FLOAT_RESIZE(rgb, w, h, "pillow-bilinear").astype(int) - pillow_fixed_point(rgb, w, h))
        worst_level, worst_share = max(worst_level, diff.max()), max(worst_share, (diff > 0).mean())
    record(say(worst_level <= 2 and worst_share <= 0.013, "gtr float Pillow weights vs fixed point (JSON note)",
               f"max_level_diff={worst_level} share_of_values_off={worst_share * 100:.2f}%"))


RUNNERS = {
    "detect": lambda: run_instance("detect"),
    "segment": lambda: run_instance("segment"),
    "pose": lambda: run_instance("pose"),
    "obb": lambda: run_instance("obb"),
    "depth": run_depth,
    "semantic": run_semantic,
}

check_float_weights()
for task in args.tasks.split(","):
    RUNNERS[task]()

print(f"library parity (LibreYOLO onnx vs .pt, not counted): {sum(lib_results)}/{len(lib_results)} passed")
c.finish(results)
