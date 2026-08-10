"""Render the task artwork /models shows, by running LibreYOLO for real.

    .venv/Scripts/python.exe scripts/build-task-art/generate-task-art.py

Nine tasks on /models had no artwork and fell back to a framed caption. This
renders one 16:9 still per task from an actual model run, so the picture on the
page is the library's own output rather than a stock illustration.

Sources are the library's own demo photographs, so the stills sit next to the
three existing clips without looking borrowed, and nothing new needs licensing.

Rendered today: semantic, panoptic, normal, matte, restore.

Still outstanding, each for its own reason:

  edge   DexiNed and TEED are BIPED-trained and LibreYOLO deliberately does not
         mirror those checkpoints, so there is nothing to download. Needs a
         locally converted checkpoint via weights/convert_teed_weights.py.
  mesh   LibreSAM3DBody weights exist on the org but the loader cannot parse a
         size out of the filename, so auto-download never starts.
  ocr    Needs text in frame. The RF100-VL thumbnails that carry text are
         photographs of real commercial packaging, which is not something to
         put on a marketing page, and PP-OCR returned no boxes through the
         attribute this script reads. Needs an unbranded source first.
  embed  Deliberately absent. Embeddings have no image output, so a still would
         be an invented diagram rather than a model run.

Each task is independent: one failing model does not stop the rest, and the
script reports at the end which stills exist and which did not render.
"""

from __future__ import annotations

import gc
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

SITE = Path(__file__).resolve().parents[2]
SHOWCASE = SITE / "public" / "showcase"
DATASETS = SITE / "public" / "articles" / "rf100vl-benchmark" / "datasets"

# The library's own demo assets, and the important thing about them is that
# they are CLEAN. The /showcase posters already have boxes and masks burned in,
# so running a model over one produces an annotation of an annotation.
ASSETS = Path("C:/Users/Usuario/Documents/GitHub/libreyolo/libreyolo/assets")
SCENE = ASSETS / "guggenheim-bilbao.jpg"  # person, architecture, sky, trees, water
PERSON = ASSETS / "parkour.jpg"

# 16:9 at a size that stays crisp on a 280px column at 2x without shipping a
# needlessly large file.
OUT_W, OUT_H = 1280, 720


def letterbox_16x9(img: np.ndarray) -> np.ndarray:
    """Centre-crop to 16:9, then resize. Keeps subjects centred rather than squashed."""
    h, w = img.shape[:2]
    target = OUT_W / OUT_H
    if w / h > target:
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        img = img[:, x0 : x0 + new_w]
    else:
        new_h = int(w / target)
        y0 = (h - new_h) // 2
        img = img[y0 : y0 + new_h, :]
    return cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


def save(name: str, img: np.ndarray) -> Path:
    out = SHOWCASE / name
    cv2.imwrite(str(out), letterbox_16x9(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def predict(weights: str, source: Path, **kw):
    """Load, predict, free the GPU, hand back the result and the source image."""
    from libreyolo import LibreYOLO

    model = LibreYOLO(weights)
    try:
        res = model.predict(str(source), verbose=False, **kw)[0]
        return res, cv2.imread(str(source))
    finally:
        del model
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def run_plot(weights: str, source: Path, **kw) -> np.ndarray:
    """For the tasks whose Results.plot() is implemented (edge, normal).

    Those two return a PIL image rather than the BGR array the box-drawing
    tasks hand back, so normalise before anything downstream touches it.
    """
    res, _ = predict(weights, source, **kw)
    out = res.plot()
    if not isinstance(out, np.ndarray):  # PIL.Image
        out = cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)
    return out


def as_array(obj) -> np.ndarray:
    """Unwrap a result payload that may be a tensor, a wrapper, or already an array."""
    for attr in ("array", "data"):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
            break
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu()
    return np.asarray(obj)


# A stable, readable palette. Deterministic per class id so re-running the
# script does not reshuffle the colours in the committed stills.
def class_colors(n: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    cols = rng.integers(60, 235, size=(max(n, 1), 3), dtype=np.uint8)
    return cols


def colorize_labels(img: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend an integer label map over the frame, leaving background untouched."""
    labels = labels.astype(np.int32)
    if labels.shape[:2] != img.shape[:2]:
        labels = cv2.resize(labels, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    ids = [i for i in np.unique(labels) if i >= 0 and i != 255]
    palette = class_colors(int(max(ids) + 1) if ids else 1)

    overlay = img.copy()
    for i in ids:
        if i == 0:  # background class stays as the photograph
            continue
        mask = labels == i
        overlay[mask] = palette[i]
        # A thin contour keeps regions legible once the fill is blended back.
        edges = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        overlay[edges > 0] = np.clip(palette[i].astype(np.int32) + 60, 0, 255).astype(np.uint8)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# ── per-task recipes ────────────────────────────────────────────────────────
# Each returns the rendered BGR image. Anything that raises is reported and
# skipped; the page falls back to its framed caption for that task.


def art_semantic():
    # PIDNet is Cityscapes-trained, so it labels road, building, sky and
    # vegetation on an urban frame. DeepLabv3 is Pascal VOC and finds only the
    # person here, which reads as instance segmentation and misses the point.
    # SegFormer would also fit but its weights are non-commercial, and this
    # still ships on a marketing page.
    res, img = predict("LibrePIDNetl-sem.pt", SCENE)
    return colorize_labels(img, as_array(res.semantic_mask))


def art_panoptic():
    res, img = predict("LibreEoMTs-panoptic.pt", SCENE)
    payload = res.panoptic
    arr = as_array(payload)
    if arr.ndim == 3 and arr.shape[0] in (1, 2):  # (id, category) stacks
        arr = arr[0]
    return colorize_labels(img, arr)


def art_ocr():
    # Needs text in frame. These thumbnails already ship on the RF100-VL
    # article page, so no new asset licensing decision is involved.
    src = next((DATASETS / c for c in ("wine-labels.webp", "invoice-processing.webp",
                                       "water-meter.webp", "signatures.webp")
                if (DATASETS / c).exists()), None)
    if src is None:
        raise FileNotFoundError("no text-bearing source thumbnail found")

    res, img = predict("LibrePPOCRl-ocr.pt", src)
    ocr = res.ocr
    out = img.copy()
    # Boxes are quadrilaterals for text detection, so draw polygons rather than
    # rectangles; slanted labels on a bottle are the whole point of the task.
    polys = getattr(ocr, "boxes", None)
    polys = as_array(polys) if polys is not None else np.empty((0,))
    for poly in polys:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=(80, 220, 120), thickness=2)
    return out


def art_matte():
    res, img = predict("LibreBiRefNetl-matte.pt", PERSON)
    alpha = as_array(res.matte).astype(np.float32)
    if alpha.shape[:2] != img.shape[:2]:
        alpha = cv2.resize(alpha, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0, 1)[..., None]

    # Composite onto a flat neutral so the cutout is the subject of the still,
    # which is what matting actually produces. Light rather than dark: the
    # subjects wear black, and on a dark ground the alpha edge disappears.
    bg = np.full_like(img, (228, 226, 222), dtype=np.uint8)
    return (img * alpha + bg * (1 - alpha)).astype(np.uint8)


def art_restore():
    # Super-resolution needs something worth restoring, so degrade first and
    # let the model undo it, then show the two halves against each other.
    #
    # Crop into the architecture before degrading: at full-frame scale a 4x
    # round trip is barely visible in a 280px column, and the panels and edges
    # of the museum are exactly where the difference shows.
    full = cv2.imread(str(SCENE))
    fh, fw = full.shape[:2]
    img = full[int(fh * 0.30) : int(fh * 0.95), int(fw * 0.00) : int(fw * 0.55)]
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    tmp = Path(__file__).parent / "_restore-input.png"
    cv2.imwrite(str(tmp), small)
    try:
        res, _ = predict("LibreRealESRGANx4-restore.pt", tmp)
    finally:
        tmp.unlink(missing_ok=True)

    restored = as_array(res.restored)
    if restored.ndim == 3 and restored.shape[0] in (1, 3):  # CHW
        restored = np.transpose(restored, (1, 2, 0))
    if restored.dtype != np.uint8:
        restored = np.clip(restored * (255 if restored.max() <= 1.01 else 1), 0, 255).astype(np.uint8)
    # The restore head hands back RGB while everything here is BGR; without
    # this the sky comes out orange and the still looks like a filter, not a
    # super-resolution result.
    restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
    restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_AREA)

    # Left half is the degraded input blown back up, right half is the model.
    naive = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = restored.copy()
    out[:, : w // 2] = naive[:, : w // 2]
    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    return out


def art_edge():
    # DexiNed and TEED publish no weights in the LibreYOLO org, so this only
    # works if the library can reach an upstream checkpoint.
    try:
        return run_plot("LibreDexiNedb-edge.pt", SCENE)
    except Exception:
        return run_plot("LibreTEEDt-edge.pt", SCENE)


def art_normal():
    return run_plot("LibreMoGe2l-normal.pt", SCENE)


def art_mesh():
    return run_plot("LibreSAM3DBodyd3-mesh.pt", PERSON)


RECIPES = {
    "task-semantic.jpg": ("semantic", art_semantic),
    "task-panoptic.jpg": ("panoptic", art_panoptic),
    "task-ocr.jpg": ("ocr", art_ocr),
    "task-matte.jpg": ("matte", art_matte),
    "task-restore.jpg": ("restore", art_restore),
    "task-edge.jpg": ("edge", art_edge),
    "task-normal.jpg": ("normal", art_normal),
    "task-mesh.jpg": ("mesh", art_mesh),
    # 'embed' is deliberately absent: embeddings have no image output, so a
    # still would have to be an invented diagram rather than a model run.
}


def main() -> int:
    only = sys.argv[1:]
    done, failed = [], []

    for filename, (task, recipe) in RECIPES.items():
        if only and task not in only:
            continue
        print(f"\n=== {task} -> {filename} ===", flush=True)
        try:
            img = recipe()
            if img is None:
                raise RuntimeError("recipe returned nothing")
            out = save(filename, img)
            print(f"    wrote {out.relative_to(SITE)}", flush=True)
            done.append(task)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=2)
            failed.append((task, f"{type(exc).__name__}: {exc}"))

    print("\n" + "=" * 60)
    print(f"rendered ({len(done)}): {', '.join(done) or 'none'}")
    print(f"failed   ({len(failed)}):")
    for task, why in failed:
        print(f"    {task}: {why[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
