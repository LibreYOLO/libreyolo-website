"""Shared helpers for the ONNX inputs and outputs checks.

Every docs entry in ``src/data/docs/onnx-io/<family>.json`` is checked by a
``check_<family>.py`` script that:

1. exports a small checkpoint with LibreYOLO (export only),
2. builds the input tensor from the entry's ``input`` block with
   :func:`preprocess` below (numpy + OpenCV, no LibreYOLO code),
3. runs the graph in plain onnxruntime and decodes the outputs exactly as the
   entry's ``postprocess`` steps say, again without LibreYOLO code,
4. compares the result with ``LibreYOLO(<file>.pt).predict`` on the same image.

An entry is published only when its check passes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src" / "data" / "docs" / "onnx-io"

INTERPOLATION = {
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}
# Plus "pillow-bilinear": Pillow's BILINEAR filter, which widens the kernel when
# shrinking. OpenCV has no equivalent; see pillow_bilinear_resize below.


def pillow_bilinear_weights(in_size: int, out_size: int) -> np.ndarray:
    """Pillow BILINEAR (Image.resize) weights for one axis, as an [out, in] matrix.

    s = in / out, f = max(s, 1). Output pixel i has center c = (i + 0.5) * s and
    reads source pixels int(c - f + 0.5) .. int(c + f + 0.5) - 1, clipped to the
    image, each weighted max(0, 1 - |j + 0.5 - c| / f), normalized to sum 1.
    """
    s = in_size / out_size
    f = max(s, 1.0)
    weights = np.zeros((out_size, in_size), np.float64)
    for i in range(out_size):
        center = (i + 0.5) * s
        lo = max(int(center - f + 0.5), 0)
        hi = min(int(center + f + 0.5), in_size)
        j = np.arange(lo, hi)
        w = np.maximum(0.0, 1.0 - np.abs(j + 0.5 - center) / f)
        weights[i, lo:hi] = w / w.sum()
    return weights


def pillow_bilinear_resize(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Width pass, round to uint8, height pass, round to uint8 (Pillow's order)."""
    x = img.astype(np.float64)
    x = np.clip(np.floor(np.einsum("ow,hwc->hoc", pillow_bilinear_weights(img.shape[1], out_w), x) + 0.5), 0, 255)
    x = np.clip(np.floor(np.einsum("oh,hwc->owc", pillow_bilinear_weights(img.shape[0], out_h), x) + 0.5), 0, 255)
    return x.astype(np.uint8)


def resize_to(img: np.ndarray, out_w: int, out_h: int, interpolation: str) -> np.ndarray:
    if interpolation == "pillow-bilinear":
        return pillow_bilinear_resize(img, out_w, out_h)
    return cv2.resize(img, (out_w, out_h), interpolation=INTERPOLATION[interpolation])


def load_entry(family: str) -> dict:
    return json.loads((DATA / f"{family}.json").read_text())


def load_variant(family: str, key: str) -> dict:
    for variant in load_entry(family)["variants"]:
        if key in variant["keys"]:
            return variant
    raise KeyError(f"{family}.json has no variant for registry key {key!r}")


def onnx_metadata(path) -> dict:
    import onnx

    return {p.key: p.value for p in onnx.load(str(path)).metadata_props}


def session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def input_hw(sess, metadata: dict) -> tuple[int, int]:
    shape = sess.get_inputs()[0].shape
    h, w = shape[2], shape[3]
    if not isinstance(h, int) or not isinstance(w, int):
        h = int(metadata.get("imgsz_h") or metadata["imgsz"])
        w = int(metadata.get("imgsz_w") or metadata["imgsz"])
    return int(h), int(w)


def preprocess(image_bgr: np.ndarray, inp: dict, H: int, W: int, metadata: dict):
    """Build the NCHW input tensor from the entry's input block only.

    Returns ``(tensor, geometry)``. ``geometry`` holds the original size, the
    x/y scale applied and the pad offsets, for mapping outputs back.
    """
    if inp["color"] == "BGR":
        img = image_bgr
    elif inp["color"] == "RGB":
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"color={inp['color']!r}")
    oh, ow = img.shape[:2]
    interp = inp.get("interpolation", "bilinear")
    geo = {"orig_w": ow, "orig_h": oh, "input_w": W, "input_h": H}

    if inp["resize"] == "letterbox":
        r = min(H / oh, W / ow)
        nh, nw = max(int(oh * r), 1), max(int(ow * r), 1)
        position = inp.get("pad_position", "topleft")
        if position == "metadata":
            position = metadata.get("letterbox_pad", "topleft")
        if position == "center":
            left, top = (W - nw) // 2, (H - nh) // 2
        elif position == "topleft":
            left, top = 0, 0
        else:
            raise ValueError(f"pad_position={position!r}")
        canvas = np.full((H, W, 3), inp.get("pad_value", 114), dtype=np.uint8)
        canvas[top : top + nh, left : left + nw] = resize_to(img, nw, nh, interp)
        geo.update(scale_x=r, scale_y=r, pad_left=left, pad_top=top)
    elif inp["resize"] == "stretch":
        canvas = resize_to(img, W, H, interp)
        geo.update(scale_x=W / ow, scale_y=H / oh, pad_left=0, pad_top=0)
    else:
        raise NotImplementedError(
            f"resize={inp['resize']!r} is not in _common; implement it in the "
            "family check script, following the entry's notes word for word."
        )

    x = canvas.astype(np.float32) / float(inp.get("scale", 255))
    if inp.get("mean") is not None:
        x = (x - np.asarray(inp["mean"], np.float32)) / np.asarray(inp["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), geo


def to_original_xyxy(boxes: np.ndarray, geo: dict) -> np.ndarray:
    """Map xyxy boxes in input-canvas pixels back to the original image."""
    out = np.asarray(boxes, dtype=np.float64).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - geo["pad_left"]) / geo["scale_x"]
    out[:, [1, 3]] = (out[:, [1, 3]] - geo["pad_top"]) / geo["scale_y"]
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, geo["orig_w"])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, geo["orig_h"])
    return out


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Plain greedy NMS. Returns kept indices sorted by descending score."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        ious = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][ious <= iou]
    return np.asarray(keep, dtype=np.int64)


def class_aware_nms(boxes, scores, classes, iou: float, max_det: int) -> np.ndarray:
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        keep.extend(idx[nms_xyxy(boxes[idx], scores[idx], iou)])
    keep = np.asarray(keep, dtype=np.int64)
    return keep[np.argsort(-scores[keep], kind="stable")][:max_det]


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def reference_detections(result) -> dict:
    """Pull boxes/scores/classes out of a LibreYOLO Results object."""
    return {
        "boxes": _np(result.boxes.xyxy).reshape(-1, 4).astype(np.float64),
        "scores": _np(result.boxes.conf).reshape(-1).astype(np.float64),
        "classes": _np(result.boxes.cls).reshape(-1).astype(np.int64),
    }


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clip(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def compare_detections(ours: dict, ref: dict, tol_px: float = 1.0, label: str = "") -> bool:
    """Greedy same-class IoU matching. Pass when every detection pairs up and
    the worst box corner differs by at most ``tol_px`` original-image pixels."""
    n_ours, n_ref = len(ours["boxes"]), len(ref["boxes"])
    worst, worst_score, unmatched = 0.0, 0.0, 0
    used = set()
    if n_ours and n_ref:
        ious = iou_matrix(ref["boxes"], ours["boxes"])
        for i in np.argsort(-ref["scores"]):
            candidates = [
                j for j in np.argsort(-ious[i])
                if j not in used and ours["classes"][j] == ref["classes"][i]
            ]
            if not candidates or ious[i, candidates[0]] < 0.5:
                unmatched += 1
                continue
            j = candidates[0]
            used.add(j)
            worst = max(worst, float(np.abs(ours["boxes"][j] - ref["boxes"][i]).max()))
            worst_score = max(worst_score, abs(float(ours["scores"][j] - ref["scores"][i])))
    else:
        unmatched = n_ref
    ok = n_ours == n_ref and unmatched == 0 and worst <= tol_px
    print(
        f"[{'PASS' if ok else 'FAIL'}] {label}: ours={n_ours} ref={n_ref} "
        f"unmatched={unmatched} max_box_diff_px={worst:.3f} max_score_diff={worst_score:.4f}"
    )
    return ok


def finish(results: list[bool]) -> None:
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) and results else 1)
