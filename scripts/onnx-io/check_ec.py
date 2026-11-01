"""Check src/data/docs/onnx-io/ec.json against LibreYOLO predictions.

Usage: python scripts/onnx-io/check_ec.py <work_dir>

EC stretches the image with Pillow's BILINEAR resize, built here with
``_common.preprocess`` (``interpolation: pillow-bilinear``). The ONNX outputs
are decoded per the entry's postprocess steps and compared with the ``.pt``.
The OpenCV INTER_LINEAR / INTER_AREA lines are INFO only: they show why the
entry insists on Pillow's filter.
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

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
WORK.mkdir(parents=True, exist_ok=True)
CONF, IOU, MAX_DET = 0.25, 0.45, 300
CHECKPOINTS = {"detect": "LibreECs.pt", "segment": "LibreECs-seg.pt", "pose": "LibreECs-pose.pt"}
BATCH1_NOTE = (
    "Detect and segment: batch size 1 only. The graph declares a dynamic batch axis, "
    "but batches of 2 or more fail in 1.7.0. Pose runs with any batch size."
)
BATCH1_ONLY = {"detect", "segment"}
EXACT_NOTE_START = "For a bit-exact match, use Pillow's integer math"

variant = c.load_variant("ec", "ec")
inp = variant["input"]
assert inp["resize"] == "stretch" and inp["interpolation"] == "pillow-bilinear", inp
image = cv2.imread(str(SAMPLE_IMAGE))
results = []
assert BATCH1_NOTE in inp["notes"], "entry must state the batch limit"
assert any(n.startswith(EXACT_NOTE_START) for n in inp["notes"])


def pillow_fixed_point_resize(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """The entry's bit-exact recipe: float weights from _common, then Pillow's integers."""
    a = img.astype(np.int64)
    for axis, n_out in ((1, out_w), (0, out_h)):
        w = c.pillow_bilinear_weights(a.shape[axis], n_out) * (1 << 22)
        w = np.where(w < 0, w - 0.5, w + 0.5).astype(np.int64)  # round half away from zero
        a = np.einsum("ow,hwc->hoc", w, a) if axis == 1 else np.einsum("oh,hwc->owc", w, a)
        a = np.clip((a + (1 << 21)) >> 22, 0, 255)
    return a.astype(np.uint8)


def preprocess_exact(image_bgr: np.ndarray, H: int, W: int):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if inp["color"] == "RGB" else image_bgr
    x = pillow_fixed_point_resize(img, W, H).astype(np.float32) / float(inp["scale"])
    x = (x - np.asarray(inp["mean"], np.float32)) / np.asarray(inp["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), {"orig_w": img.shape[1], "orig_h": img.shape[0]}


# --------------------------------------------------------------- decode ----


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def cxcywh_to_original_xyxy(b: np.ndarray, geo: dict) -> np.ndarray:
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float64)
    xyxy[:, [0, 2]] *= geo["orig_w"]
    xyxy[:, [1, 3]] *= geo["orig_h"]
    return xyxy  # LibreYOLO does not clip EC boxes


def decode_detect(outs: dict, geo: dict):
    scores = sigmoid(outs["pred_logits"][0])  # [queries, nc]
    nc = scores.shape[1]
    flat = scores.reshape(-1)
    top = np.argsort(-flat, kind="stable")[: min(MAX_DET, flat.size)]
    top = top[flat[top] > CONF]
    query, cls = top // nc, top % nc
    boxes = cxcywh_to_original_xyxy(outs["pred_boxes"][0][query], geo)
    return {"boxes": boxes, "scores": flat[top], "classes": cls.astype(np.int64)}, query


def decode_segment(outs: dict, geo: dict):
    dets, query = decode_detect(outs, geo)
    logits = outs["pred_masks"][0][query].astype(np.float32)  # [n, mask_h, mask_w]
    size = (geo["orig_w"], geo["orig_h"])
    masks = np.stack([cv2.resize(m, size, interpolation=cv2.INTER_LINEAR) > 0.0 for m in logits]) if len(logits) else np.zeros((0, geo["orig_h"], geo["orig_w"]), bool)
    return dets, masks


def decode_pose(outs: dict, geo: dict, num_keypoints: int):
    person = sigmoid(outs["pred_logits"][0])[:, -1]  # [queries], last logit is the person class
    top = np.argsort(-person, kind="stable")[: min(MAX_DET, person.size)]
    top = top[person[top] >= CONF]
    kpts = outs["pred_keypoints"][0][top].reshape(-1, num_keypoints, 2).astype(np.float64)
    kpts[..., 0] *= geo["orig_w"]
    kpts[..., 1] *= geo["orig_h"]
    boxes = np.concatenate([kpts.min(axis=1), kpts.max(axis=1)], axis=1) if len(top) else np.zeros((0, 4))
    dets = {"boxes": boxes, "scores": person[top], "classes": np.zeros(len(top), np.int64)}
    return dets, kpts


# -------------------------------------------------------------- compare ----


def match(ours: dict, ref: dict):
    """Pair each reference detection with its matched ONNX detection (as compare_detections)."""
    pairs, used = [], set()
    if len(ours["boxes"]) and len(ref["boxes"]):
        ious = c.iou_matrix(ref["boxes"], ours["boxes"])
        for i in np.argsort(-ref["scores"]):
            cand = [j for j in np.argsort(-ious[i]) if j not in used and ours["classes"][j] == ref["classes"][i]]
            if cand and ious[i, cand[0]] >= 0.5:
                used.add(cand[0])
                pairs.append((int(i), int(cand[0])))
    return pairs


def compare_masks(ours_dets, our_masks, ref_dets, ref_masks, label):
    pairs = match(ours_dets, ref_dets)
    worst = 1.0
    for i, j in pairs:
        a, b = ref_masks[i], our_masks[j]
        union = np.logical_or(a, b).sum()
        worst = min(worst, 1.0 if union == 0 else np.logical_and(a, b).sum() / union)
    ok = len(pairs) == len(ref_dets["boxes"]) and worst >= 0.99
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: matched={len(pairs)} min_mask_iou={worst:.5f}")
    return ok


def compare_keypoints(ours_dets, our_kpts, ref_dets, ref_kpts, ref_vis, label):
    pairs = match(ours_dets, ref_dets)
    worst = max((float(np.abs(our_kpts[j] - ref_kpts[i]).max()) for i, j in pairs), default=0.0)
    vis_ok = ref_vis is None or bool(np.all(ref_vis == 1.0))
    ok = len(pairs) == len(ref_dets["boxes"]) and worst <= 1.0 and vis_ok
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: matched={len(pairs)} max_kpt_diff_px={worst:.3f} ref_visibility_all_1={vis_ok}")
    return ok


def reference(result, task):
    ref = c.reference_detections(result)
    if task == "segment":
        ref["masks"] = c._np(result.masks.data).astype(bool)
    if task == "pose":
        ref["kpts"] = c._np(result.keypoints.xy).astype(np.float64)
        conf = getattr(result.keypoints, "conf", None)
        ref["vis"] = None if conf is None else c._np(conf)
    return ref


def expect_raise(label, fn):
    try:
        fn()
    except NotImplementedError as e:
        print(f"[PASS] {label}: NotImplementedError ({str(e)[:70]}...)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")
        return False
    print(f"[FAIL] {label}: export did not raise")
    return False


# ------------------------------------------------------------------ run ----

for task in variant["tasks"]:
    ckpt = CHECKPOINTS[task]
    model = LibreYOLO(ckpt)
    assert model.task == task, (ckpt, model.task)
    pred = model.predict(SAMPLE_IMAGE, conf=CONF, iou=IOU, max_det=MAX_DET)
    ref = reference(pred[0] if isinstance(pred, list) else pred, task)

    for export in variant["tasks"][task]:
        assert export["when"] == "default"
        path = WORK / f"ec_{task}.onnx"
        LibreYOLO(ckpt).export(format="onnx", output_path=str(path))
        meta = c.onnx_metadata(path)
        sess = c.session(path)
        assert sess.get_inputs()[0].name == inp["name"]
        assert [o.name for o in sess.get_outputs()] == [o["name"] for o in export["outputs"]], [o.name for o in sess.get_outputs()]
        H, W = c.input_hw(sess, meta)
        assert H == W == 640, (H, W)
        label = f"ec {task} {export['when']} input={H}x{W}"
        if task == "pose":
            assert meta["num_keypoints"] == "17" and meta["keypoint_dim"] == "2", meta

        for interp in ("pillow-bilinear", "pillow-fixed-point", "bilinear", "area"):
            tag = {"pillow-bilinear": "", "pillow-fixed-point": " pillow fixed-point"}.get(interp, f" [info: cv2 {interp} instead]")
            if interp == "pillow-fixed-point":
                x, geo = preprocess_exact(image, H, W)
            else:
                x, geo = c.preprocess(image, {**inp, "interpolation": interp}, H, W, meta)
            if interp == inp["interpolation"]:
                x1 = x
            outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {inp["name"]: x})))
            assert outs["pred_logits"].shape[1] == (60 if task == "pose" else 300), outs["pred_logits"].shape
            if task == "segment":
                assert outs["pred_masks"].shape[2:] == (H // 4, W // 4), outs["pred_masks"].shape
            if task == "pose":
                assert outs["pred_logits"].shape[2] == 2 and outs["pred_keypoints"].shape[2] == 17 * 2
            if task == "detect":
                ours, _ = decode_detect(outs, geo)
            elif task == "segment":
                ours, masks = decode_segment(outs, geo)
            else:
                ours, kpts = decode_pose(outs, geo, int(meta.get("num_keypoints", 17)))
            if "info" in tag:
                # Not a gate: shows why the entry says Pillow resize, not cv2 INTER_LINEAR.
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    c.compare_detections(ours, ref, label=label + tag)
                print("[INFO]" + buf.getvalue().split("]", 1)[1].rstrip())
                continue
            ok = c.compare_detections(ours, ref, label=label + tag)
            if task == "segment":
                ok = compare_masks(ours, masks, ref, ref["masks"], label + tag + " masks") and ok
            if task == "pose":
                ok = compare_keypoints(ours, kpts, ref, ref["kpts"], ref["vis"], label + tag + " keypoints") and ok
            results.append(ok)

        # Batch 2: two copies of the input stacked. Each row must decode like batch 1.
        try:
            outs2 = sess.run(None, {inp["name"]: np.concatenate([x1, x1])})
            rows = [{o.name: v[k : k + 1] for o, v in zip(sess.get_outputs(), outs2)} for k in (0, 1)]
            outs1 = sess.run(None, {inp["name"]: x1})
            diff = max(float(np.abs(r[o.name] - v).max()) for r in rows for o, v in zip(sess.get_outputs(), outs1))
            batch_ok, batch_msg = diff <= 1e-3, f"runs, max abs diff vs batch 1 = {diff:.2e}"
        except Exception as e:  # noqa: BLE001
            batch_ok, batch_msg = False, f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        claims_batch1 = task in BATCH1_ONLY
        ok = (not batch_ok) if claims_batch1 else batch_ok
        print(f"[{'PASS' if ok else 'FAIL'}] ec {task} batch=2 (entry says {'batch 1 only' if claims_batch1 else 'any batch'}): {batch_msg}")
        results.append(ok)

        # LibreYOLO's own ONNX backend on the same file, against the .pt.
        onnx_pred = LibreYOLO(str(path)).predict(SAMPLE_IMAGE, conf=CONF, iou=IOU, max_det=MAX_DET)
        onnx_ref = reference(onnx_pred[0] if isinstance(onnx_pred, list) else onnx_pred, task)
        ok = c.compare_detections(onnx_ref, ref, label=f"ec {task} LibreYOLO(.onnx).predict vs .pt")
        if task == "segment":
            ok = compare_masks(onnx_ref, onnx_ref["masks"], ref, ref["masks"], f"ec {task} LibreYOLO(.onnx) masks") and ok
        if task == "pose":
            ok = compare_keypoints(onnx_ref, onnx_ref["kpts"], ref, ref["kpts"], onnx_ref["vis"], f"ec {task} LibreYOLO(.onnx) keypoints") and ok
        results.append(ok)
        path.unlink()

    # Claims in the entry: no embedded NMS, square input only.
    results.append(expect_raise(f"ec {task} nms=True rejected", lambda: LibreYOLO(ckpt).export(format="onnx", nms=True, output_path=str(WORK / "nms.onnx"))))
    results.append(expect_raise(f"ec {task} imgsz=(384, 640) rejected", lambda: LibreYOLO(ckpt).export(format="onnx", imgsz=(384, 640), output_path=str(WORK / "rect.onnx"))))
    for leftover in WORK.glob("*.onnx"):
        leftover.unlink()

c.finish(results)
