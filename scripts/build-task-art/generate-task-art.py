"""Render the task artwork /models shows, by running LibreYOLO for real.

    .venv/Scripts/python.exe scripts/build-task-art/generate-task-art.py

Nine tasks on /models had no artwork and fell back to a framed caption. This
renders one 16:9 still per task from an actual model run, so the picture on the
page is the library's own output rather than a stock illustration.

Sources are the library's own demo photographs, so the stills sit next to the
three existing clips without looking borrowed, and nothing new needs licensing.

Rendered: semantic, panoptic, normal, matte, restore, ocr, embed.

Two are still outstanding:

  edge   DexiNed and TEED are BIPED-trained and LibreYOLO deliberately does not
         mirror those checkpoints, so there is nothing to download. Convert one
         you are licensed to use with weights/convert_teed_weights.py, or mirror
         the checkpoints on the org and this recipe starts working as written.
  mesh   LibreSAM3DBodyd3-mesh is public on the org but the file is named
         model.ckpt while the loader expects LibreSAM3DBodyd3-mesh.pt, so it
         fails with "could not determine download URL". Library bug, not a
         licence problem.

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
    # SegFormer-B5 on ADE20K: 150 classes, so it parses sky, building, tree,
    # road and person separately instead of collapsing the scene.
    #
    # Its weights are non-commercial (NVIDIA Source Code License), which is a
    # restriction on use, not a bar on the model being here: LibreYOLO already
    # hosts LibreSegformer on its own org under that licence. Earlier passes
    # used PIDNet (Cityscapes) and DeepLabv3 (Pascal VOC, which finds only the
    # person and reads as instance segmentation); both were worse.
    for weights in ("LibreSegformerb5-sem.pt", "LibreSegformerb2-sem.pt", "LibrePIDNetl-sem.pt"):
        try:
            res, img = predict(weights, SCENE)
            print(f"    semantic via {weights}", flush=True)
            return colorize_labels(img, as_array(res.semantic_mask))
        except Exception as exc:
            print(f"    {weights} failed: {type(exc).__name__}: {str(exc)[:110]}", flush=True)
    raise RuntimeError("no semantic model could be loaded")


def art_panoptic():
    res, img = predict("LibreEoMTs-panoptic.pt", SCENE)
    payload = res.panoptic
    arr = as_array(payload)
    if arr.ndim == 3 and arr.shape[0] in (1, 2):  # (id, category) stacks
        arr = arr[0]
    return colorize_labels(img, arr)


def make_ocr_document(path: Path) -> None:
    """Render the sheet PP-OCR reads.

    Every text-bearing photograph to hand was of real commercial packaging,
    which is not something to put on a marketing page, so the input is drawn
    here instead: an invented calibration certificate with no real company,
    product or person on it. The input is synthetic; the boxes and the strings
    on top of it are a real PP-OCRv5 run.
    """
    from PIL import Image, ImageDraw, ImageFont

    def font(size, bold=False):
        for name in (("arialbd.ttf" if bold else "arial.ttf"), "segoeui.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    # Drawn at 16:9 so the saved still needs no crop; at 1400x900 the centre
    # crop took the top off the heading.
    w, h = 1400, 788
    doc = Image.new("RGB", (w, h), (250, 249, 246))
    d = ImageDraw.Draw(doc)
    d.rectangle([0, 0, w, 88], fill=(237, 235, 230))
    d.text((60, 26), "CALIBRATION REPORT", font=font(38, True), fill=(24, 24, 28))
    d.text((1030, 34), "No. 4471-B", font=font(25), fill=(70, 70, 76))

    rows = [
        ("Instrument", "Optical bench K-12"),
        ("Serial number", "SN 8842 1907"),
        ("Date of test", "2026-08-10"),
        ("Ambient temperature", "21.4 C"),
        ("Relative humidity", "43 %"),
        ("Measured deviation", "0.018 mm"),
        ("Tolerance", "0.050 mm"),
        ("Result", "PASS"),
    ]
    y = 124
    for key, value in rows:
        d.text((60, y), key, font=font(26), fill=(92, 92, 98))
        d.text((560, y), value, font=font(26, True), fill=(20, 20, 24))
        d.line([(60, y + 44), (w - 60, y + 44)], fill=(223, 221, 216), width=2)
        y += 72

    d.text((60, y + 20), "Signed by the calibration technician", font=font(23), fill=(122, 122, 128))
    doc.save(path)


def art_ocr():
    from libreyolo import LibreYOLO

    src = Path(__file__).parent / "_ocr-document.png"
    make_ocr_document(src)

    # PP-OCR returns one result per text region rather than one result holding
    # every region, so the whole list is the detection set.
    model = LibreYOLO("LibrePPOCRl-ocr.pt")
    try:
        results = model.predict(str(src), verbose=False)
    finally:
        del model
        gc.collect()

    img = cv2.imread(str(src))
    src.unlink(missing_ok=True)

    out = img.copy()
    drawn = 0
    for res in results:
        regions = getattr(res, "ocr", None)
        if regions is None:
            continue
        polys = as_array(getattr(regions, "polygons"))
        texts = list(getattr(regions, "texts", []) or [])
        for i, poly in enumerate(polys):
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=(90, 200, 110), thickness=3)
            drawn += 1
            # Echo the recognised string above its box, which is the half of
            # OCR a box alone does not show.
            if i < len(texts) and texts[i]:
                x, y = pts[:, 0].min(), pts[:, 1].min()
                cv2.putText(out, str(texts[i])[:28], (int(x), max(int(y) - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (28, 120, 44), 2, cv2.LINE_AA)

    if drawn == 0:
        raise RuntimeError("PP-OCR found no text regions")
    print(f"    {drawn} text regions read", flush=True)
    return out


def art_embed():
    """Embeddings have no picture of their own, so show what they are for.

    One query thumbnail, then the four nearest of the other ninety-nine by
    cosine similarity in CLIP space, scored. Every number on it comes from a
    real forward pass; nothing here is a diagram of how embeddings work.
    """
    from libreyolo import LibreYOLO

    thumbs = sorted(DATASETS.glob("*.webp"))
    if len(thumbs) < 12:
        raise FileNotFoundError("not enough dataset thumbnails to retrieve against")

    model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
    try:
        results = model.predict([str(p) for p in thumbs], verbose=False)
    finally:
        del model
        gc.collect()

    vectors, kept = [], []
    for path, res in zip(thumbs, results):
        emb = getattr(res, "embeddings", None)
        if emb is None:
            continue
        vec = as_array(emb).astype(np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        vectors.append(vec / norm)
        kept.append(path)

    if len(vectors) < 6:
        raise RuntimeError(f"only {len(vectors)} embeddings came back")

    matrix = np.stack(vectors)
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -1)

    # Query the image whose best match is strongest, so the still shows the
    # retrieval working rather than an arbitrary and possibly poor example.
    query = int(sims.max(axis=1).argmax())
    order = np.argsort(-sims[query])[:4]

    # Composed at the final 16:9 directly, because a wide strip would be
    # centre-cropped on save and the query tile would be the first thing lost.
    # Query large on the left, its four nearest in a block on the right, which
    # fills the frame instead of leaving two thirds of it empty.
    panel = np.full((OUT_H, OUT_W, 3), 246, np.uint8)
    big, small, gap = 470, 250, 24
    font = cv2.FONT_HERSHEY_SIMPLEX

    def place(idx, x, y, size, score=None, is_query=False):
        img = cv2.imread(str(kept[idx]))
        if img is None:
            return
        panel[y : y + size, x : x + size] = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (x, y), (x + size, y + size),
                      (40, 40, 44) if is_query else (203, 201, 197), 3 if is_query else 1)
        head = "query" if is_query else f"{score:.3f}"
        cv2.putText(panel, head, (x, y - 12), font, 0.78 if is_query else 0.68,
                    (28, 28, 32) if is_query else (108, 106, 110), 2, cv2.LINE_AA)
        cv2.putText(panel, kept[idx].stem[:24], (x, y + size + 24), font, 0.5,
                    (122, 120, 124), 1, cv2.LINE_AA)

    qx, qy = gap * 2, (OUT_H - big) // 2
    place(query, qx, qy, big, is_query=True)

    # Each neighbour needs a score above it and a name below, so a row is
    # taller than its tile. Sizing on the tile alone clipped both.
    label_h, caption_h, row_gap = 30, 28, 26
    row_h = label_h + small + caption_h
    gx = qx + big + gap * 3
    top = (OUT_H - (row_h * 2 + row_gap)) // 2 + label_h

    for n, idx in enumerate(order):
        col, row = n % 2, n // 2
        place(int(idx), gx + col * (small + gap * 2), top + row * (row_h + row_gap),
              small, score=float(sims[query][idx]))

    print(f"    query={kept[query].stem}, top={[kept[i].stem for i in order]}", flush=True)
    return panel


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
    "task-embed.jpg": ("embed", art_embed),
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
