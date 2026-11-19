"""Render the augmentation guide's figures with the library's own pipelines.

Usage, from the website root, in the same environment as collect.py:

    python scripts/build-augmentation/render_figures.py

Writes ``public/augmentation/*.webp`` and ``src/data/docs/augmentation-figures.json``.

Every panel is produced by LibreYOLO's training code at the installed version,
fed the one sample image the docs use (``guggenheim-bilbao.jpg``) through the
library's own ``YOLODataset``. The person box is not drawn by hand: it is what
``LibreYOLO9t.pt`` detects on that image, written as a YOLO label file. With a
one-image dataset every mosaic tile and every MixUp partner is that same photo.

Three techniques keep the panels comparable and honest:

- Fixed seeds. Panels in one figure share a seed, so the mosaic layout and the
  other random draws are the same and only the knob under study changes.
- Pinned draws. The geometric knobs are drawn from a range (``degrees=10``
  means any angle in -10..+10). To show what a value can do, the figure passes
  the edge of the range as a one-point range, ``(10, 10)``, which the
  library's ``get_aug_params`` accepts and samples exactly. Those panels show
  the strongest effect a value allows, and their labels give the drawn value
  (``+10°``, ``0.5×``) rather than a setting.
- Chosen draws. Where a figure shows ordinary random draws, the script picks
  the seeds by a stated rule (a visible effect, a blend ratio near one half)
  and the manifest records what was drawn.

Panels with boxes record the size of the largest box, the most complete copy of
the person, in pixels at the 640-pixel training resolution.
"""

from __future__ import annotations

import contextlib
import json
import random
import shutil
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import libreyolo  # noqa: E402
from libreyolo import SAMPLE_IMAGE  # noqa: E402

OUT_DIR = Path("public/augmentation")
MANIFEST = Path("src/data/docs/augmentation-figures.json")
PANEL = 480  # pixels per square panel; shown at about half that on screen
IMGSZ = 640
WEBP_QUALITY = 80

BOX_COLOR = (212, 182, 6)  # libre-400 #06b6d4, in BGR
BOX_EDGE = (20, 20, 20)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def detect_person() -> dict:
    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9t.pt")
    result = model.predict(SAMPLE_IMAGE)
    result = result[0] if isinstance(result, list) else result
    boxes = result.boxes
    if len(boxes) != 1:
        raise SystemExit(f"expected one detection on the sample image, got {len(boxes)}")
    xyxy = [round(float(v), 1) for v in boxes.xyxy[0].tolist()]
    return {"model": "LibreYOLO9t.pt", "class": "person", "xyxy": xyxy,
            "conf": round(float(boxes.conf[0]), 3)}


def make_dataset_dir(root: Path, box: dict) -> Path:
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    shutil.copy(SAMPLE_IMAGE, images / "guggenheim-bilbao.jpg")
    h, w = cv2.imread(str(SAMPLE_IMAGE)).shape[:2]
    x1, y1, x2, y2 = box["xyxy"]
    cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
    (labels / "guggenheim-bilbao.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return root


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def chw_float_to_bgr(img, mean=None, std=None) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32).transpose(1, 2, 0)
    if mean is not None:
        arr = arr * np.asarray(std, dtype=np.float32).reshape(-1) + np.asarray(mean, dtype=np.float32).reshape(-1)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_boxes(bgr: np.ndarray, boxes_xyxy: list) -> np.ndarray:
    out = bgr.copy()
    for x1, y1, x2, y2 in boxes_xyxy:
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, BOX_EDGE, 5, cv2.LINE_AA)
        cv2.rectangle(out, p1, p2, BOX_COLOR, 3, cv2.LINE_AA)
    return out


def largest(boxes_xyxy: list):
    if not boxes_xyxy:
        return None
    w, h = max(((x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes_xyxy), key=lambda s: s[0] * s[1])
    return [int(round(w)), int(round(h))]


class Panel:
    def __init__(self, bgr, boxes=None, label="", note=None, report_box=True):
        self.bgr, self.boxes, self.label, self.note = bgr, boxes, label, note
        self.report_box = report_box


def diff(a: Panel, b: Panel) -> float:
    return float(np.abs(a.bgr.astype(np.int16) - b.bgr.astype(np.int16)).mean())


# --------------------------------------------------------------------------
# Pipelines, each built the way its trainer builds it
# --------------------------------------------------------------------------

@contextlib.contextmanager
def recording_scale(record: list):
    """Record the scale each affine warp draws, without changing it."""
    from libreyolo.data.augment import geometry

    original = geometry.get_affine_matrix

    def spy(*args, **kwargs):
        matrix, scale = original(*args, **kwargs)
        record.append(float(scale))
        return matrix, scale

    geometry.get_affine_matrix = spy
    try:
        yield
    finally:
        geometry.get_affine_matrix = original


@contextlib.contextmanager
def recording_beta(record: list):
    """Record YOLOv9's MixUp ratio, a numpy Beta(1, 1) draw, without changing it."""
    original = np.random.beta

    def spy(a, b, *args, **kwargs):
        value = original(a, b, *args, **kwargs)
        record.append(float(value))
        return value

    np.random.beta = spy
    try:
        yield
    finally:
        np.random.beta = original


def yolo9_sample(dataset, seed: int, *, mosaic_prob=1.0, degrees=0.0, translate=0.0,
                 mosaic_scale=(0.5, 0.5), shear=0.0, perspective=0.0, mixup_prob=0.0,
                 hsv_prob=0.0, flip_prob=0.0, flipud=0.0) -> tuple[Panel, dict]:
    """One sample from YOLO9MosaicMixupDataset + YOLO9TrainTransform."""
    from libreyolo.data.augment.yolo9 import YOLO9MosaicMixupDataset, YOLO9TrainTransform

    # As YOLO9Trainer.create_transforms builds it.
    preproc = YOLO9TrainTransform(max_labels=300, flip_prob=flip_prob, vertical_flip_prob=flipud,
                                  hsv_prob=hsv_prob, rot90_prob=0.0)
    wrapper = YOLO9MosaicMixupDataset(
        dataset=dataset, img_size=(IMGSZ, IMGSZ), mosaic=mosaic_prob > 0, preproc=preproc,
        degrees=degrees, translate=translate, mosaic_scale=mosaic_scale, mixup_scale=(0.5, 1.5),
        shear=shear, enable_mixup=mixup_prob > 0, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob,
        perspective=perspective,
    )
    scales, lams = [], []
    seed_all(seed)
    with recording_scale(scales), recording_beta(lams):
        img, labels, *_ = wrapper[0]
    real = labels[labels[:, 0] >= 0]
    boxes = [list(map(float, row[1:5] * IMGSZ)) for row in real]
    return Panel(chw_float_to_bgr(img), boxes), {"scale": scales[0] if scales else None,
                                                 "lam": lams[0] if lams else None}


def detr_transform():
    from libreyolo.models.dfine.transforms import DFINETrainTransform

    # As DFINETrainer.create_transforms builds it for detection. flip_prob is
    # set to 0 here so the strong ops are the only change.
    return DFINETrainTransform(max_labels=120, flip_prob=0.0, imgsz=IMGSZ)


def detr_run(dataset, seed: int, keep=None) -> Panel:
    """Run the D-FINE transform; ``keep`` names the strong ops left in."""
    from torchvision.transforms import v2 as tv2

    transform = detr_transform()
    if keep is not None:
        ops = [op for op in transform._strong.transforms
               if type(op).__name__ in keep or type(op).__name__ == "SanitizeBoundingBoxes"
               or (type(op).__name__ == "RandomApply" and "RandomIoUCrop" in keep)]
        transform._strong = tv2.Compose(ops)
    img, label, *_ = dataset.pull_item(0)
    seed_all(seed)
    out, padded = transform(img, label.copy(), (IMGSZ, IMGSZ))
    rows = padded[padded[:, 3] > 0]
    boxes = [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for _, cx, cy, w, h in rows.tolist()]
    return Panel(chw_float_to_bgr(out), boxes)


def rfdetr_run(box: dict, seed: int, crop_resize_prob: float) -> Panel:
    from libreyolo.data.augment.constants import IMAGENET_MEAN, IMAGENET_STD
    from libreyolo.data.augment.rfdetr import RFDETRDetTransform

    # RF-DETR receives the original image (wants_unresized_image) and resizes
    # straight to its square canvas, 512 here.
    transform = RFDETRDetTransform(max_labels=300, flip_prob=0.0, imgsz=512,
                                   crop_resize_prob=crop_resize_prob)
    image = cv2.imread(str(SAMPLE_IMAGE))
    target = np.asarray([box["xyxy"] + [0.0]], dtype=np.float32)
    seed_all(seed)
    out, padded = transform(image, target, None)[:2]
    bgr = chw_float_to_bgr(out, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    scale = 640.0 / bgr.shape[1]
    bgr = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_CUBIC)
    rows = padded[padded[:, 3] > 0]
    boxes = [[(cx - w / 2) * scale, (cy - h / 2) * scale, (cx + w / 2) * scale, (cy + h / 2) * scale]
             for _, cx, cy, w, h in rows.tolist()]
    return Panel(bgr, boxes)


CLS_SIZE = 448


def classify_image(seed: int, **knobs):
    from PIL import Image

    from libreyolo.data.augment.classify import IMAGENET_MEAN, IMAGENET_STD, build_classify_transforms

    transform = build_classify_transforms(CLS_SIZE, True, **knobs)
    seed_all(seed)
    tensor = transform(Image.open(SAMPLE_IMAGE).convert("RGB"))
    return tensor, chw_float_to_bgr(tensor.numpy(), mean=IMAGENET_MEAN, std=IMAGENET_STD)


def cls_panel(label: str, seed: int, note=None, **knobs) -> Panel:
    _, bgr = classify_image(seed, **knobs)
    return Panel(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_CUBIC), None, label, note,
                 report_box=False)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

YOLO9_DEFAULTS = {"translate": 0.1, "mosaic_scale": (0.5, 1.5)}  # YOLO9Config at 1.8.0


def figures(dataset, box: dict) -> dict[str, list[Panel]]:
    figs: dict[str, list[Panel]] = {}

    def y9(label, seed, note=None, **kw) -> Panel:
        panel, _ = yolo9_sample(dataset, seed, **kw)
        panel.label, panel.note = label, note
        return panel

    # Mosaic: off, the whole 2x canvas, then two draws at YOLOv9's defaults,
    # one zoomed out and one zoomed in (first seeds from 1 with a visible box).
    mosaic = [y9("mosaic_prob=0", 7, mosaic_prob=0.0), y9("mosaic_prob=1", 7, note="0.5×")]
    wanted = [lambda s: s < 0.75, lambda s: s > 1.2]
    for seed in range(1, 200):
        if not wanted:
            break
        panel, drawn = yolo9_sample(dataset, seed, **YOLO9_DEFAULTS)
        if panel.boxes and wanted[0](drawn["scale"]):
            panel.label, panel.note = "mosaic_prob=1", f"{drawn['scale']:.2f}×"
            mosaic.append(panel)
            wanted.pop(0)
    figs["mosaic"] = mosaic

    # Zooming in crops the canvas to its top-left corner, so pick the mosaic
    # layout (seed) that keeps the most of a person in frame at 1.5×.
    def box_area(seed):
        panel, _ = yolo9_sample(dataset, seed, mosaic_scale=(1.5, 1.5))
        size = largest(panel.boxes or [])
        return size[0] * size[1] if size else 0

    zoom_seed = max(range(1, 60), key=box_area)
    figs["mosaic_scale"] = [y9(f"{s:g}×", zoom_seed, mosaic_scale=(s, s)) for s in (0.1, 0.5, 1.0, 1.5)]
    figs["degrees"] = [y9(f"{d:+g}°" if d else "0°", 7, degrees=(float(d), float(d))) for d in (0, 10, 45)]
    figs["translate"] = [y9(f"+{t * IMGSZ:g} px" if t else "0 px", 7, translate=(t, t))
                         for t in (0.0, 0.1, 0.25)]
    figs["shear"] = [y9(f"{s:+g}°" if s else "0°", 7, shear=(float(s), float(s))) for s in (0, 2, 10)]
    # Perspective: the corner offsets are uniform(0, max) draws. One seed, so
    # every panel has the same random corners, scaled by the value.
    figs["perspective"] = [y9(f"perspective={p:g}", 12, perspective=p) for p in (0.0, 0.0005, 0.001, 0.002)]

    base = y9("hsv_prob=0", 1, mosaic_prob=0.0)
    draws = []
    for seed in range(1, 120):
        panel = y9("hsv_prob=1", seed, mosaic_prob=0.0, hsv_prob=1.0)
        draws.append((diff(panel, base), seed, panel))
    # The three most visible draws, so the figure shows the range of the jitter.
    figs["hsv"] = [base] + [p for _, _, p in sorted(draws, key=lambda d: -d[0])[:3]]

    figs["flip"] = [
        y9("flip_prob=0", 3, mosaic_prob=0.0),
        y9("flip_prob=1", 3, mosaic_prob=0.0, flip_prob=1.0),
        y9("flipud=1", 3, mosaic_prob=0.0, flipud=1.0),
    ]

    # MixUp on a whole-canvas mosaic, with the first draws whose blend ratio is
    # near one half and near four fifths.
    mix = [y9("mixup_prob=0", 7)]
    targets = [(0.45, 0.55), (0.75, 0.85)]
    for seed in range(1, 400):
        if not targets:
            break
        panel, drawn = yolo9_sample(dataset, seed, mixup_prob=1.0)
        lo, hi = targets[0]
        if drawn["lam"] is not None and lo <= drawn["lam"] <= hi:
            panel.label, panel.note = "mixup_prob=1", f"λ={drawn['lam']:.2f}"
            mix.append(panel)
            targets.pop(0)
    figs["mixup"] = mix

    # DETR-style: the D-FINE transform with one strong op at a time (the first
    # seed where that op fires), then three draws of the whole chain.
    detr = [detr_run(dataset, 1, keep=())]
    detr[0].label = "Resize"
    for name in ("RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"):
        for seed in range(1, 100):
            panel = detr_run(dataset, seed, keep=(name,))
            if diff(panel, detr[0]) > 1.0:
                break
        panel.label = name
        detr.append(panel)
    figs["detr_ops"] = detr
    chain = []
    for seed in (10, 11, 12):
        panel = detr_run(dataset, seed)
        panel.label = "D-FINE"
        chain.append(panel)
    figs["detr_samples"] = chain

    # RF-DETR: no crop, then the two crops that differ most from it.
    rf_base = rfdetr_run(box, 1, 0.0)
    rf_base.label = "crop_resize_prob=0"
    crops = []
    for seed in range(1, 80):
        panel = rfdetr_run(box, seed, 1.0)
        crops.append((diff(panel, rf_base), seed, panel))
    picked = [p for _, _, p in sorted(crops, key=lambda c: -c[0])[:2]]
    for panel in picked:
        panel.label = "crop_resize_prob=1"
    figs["rfdetr_crop"] = [rf_base] + picked

    # Classification, shown at 448 px so the effects are legible.
    figs["cls_crop"] = [
        cls_panel("scale=(0.5, 1.0)", 1, flip_prob=0.0),
        cls_panel("scale=(0.5, 1.0)", 2, flip_prob=0.0),
        cls_panel("scale=0.08", 3, flip_prob=0.0, scale=0.08),
    ]
    # scale=1.0 makes the crop deterministic (the whole frame), so the policy
    # or the erasing op is the only change.
    figs["auto_augment"] = [cls_panel("auto_augment=None", 5, flip_prob=0.0, scale=1.0)] + [
        cls_panel(f'auto_augment="{p}"', 5, flip_prob=0.0, scale=1.0, auto_augment=p)
        for p in ("randaugment", "autoaugment", "augmix")
    ]
    erasing = [cls_panel("erasing=0", 5, flip_prob=0.0, scale=1.0)]
    for seed in range(5, 200):
        if len(erasing) == 3:
            break
        panel = cls_panel("erasing=0.5", seed, flip_prob=0.0, scale=1.0, erasing=0.5)
        if diff(panel, erasing[0]) > 0.5:  # it fires with probability 0.5
            erasing.append(panel)
    figs["erasing"] = erasing
    figs["batch_mix"] = batch_mix_panels()
    return figs


def batch_mix_panels() -> list[Panel]:
    from libreyolo.data.augment.classify import IMAGENET_MEAN, IMAGENET_STD, ClassifyBatchMixer

    a, a_bgr = classify_image(1, flip_prob=0.0, scale=(0.3, 0.3))
    b, b_bgr = classify_image(4, flip_prob=1.0, scale=(0.3, 0.3))
    batch = [(a, 0), (b, 1)]
    panels = [Panel(cv2.resize(a_bgr, (640, 640)), None, "A", report_box=False),
              Panel(cv2.resize(b_bgr, (640, 640)), None, "B", report_box=False)]
    for label, kw in (("mixup=1", {"mixup": 1.0, "cutmix": 0.0}), ("cutmix=1", {"mixup": 0.0, "cutmix": 1.0})):
        mixer = ClassifyBatchMixer(num_classes=2, **kw)
        # The first draw whose label weight for A is near one half.
        for seed in range(1, 400):
            seed_all(seed)
            imgs, targets, *_ = mixer(batch)
            weight = float(targets[0][0])
            if 0.4 <= weight <= 0.6:
                break
        bgr = chw_float_to_bgr(imgs[0].numpy(), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        panels.append(Panel(cv2.resize(bgr, (640, 640)), None, label,
                            note=f"{weight:.2f} A + {1 - weight:.2f} B", report_box=False))
    return panels


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write(figs: dict[str, list[Panel]], box: dict) -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    manifest = {
        "libreyolo_version": libreyolo.__version__,
        "image": "guggenheim-bilbao.jpg",
        "box": box,
        "panel_px": PANEL,
        "figures": {},
    }
    for name, panels in figs.items():
        entries = []
        for i, panel in enumerate(panels, 1):
            bgr = panel.bgr if not panel.boxes else draw_boxes(panel.bgr, panel.boxes)
            bgr = cv2.resize(bgr, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
            filename = f"{name.replace('_', '-')}-{i}.webp"
            ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])
            if not ok:
                raise SystemExit(f"webp encode failed for {filename}")
            (OUT_DIR / filename).write_bytes(buf.tobytes())
            entry = {"src": f"/augmentation/{filename}", "width": PANEL, "height": PANEL,
                     "label": panel.label}
            if panel.note:
                entry["note"] = panel.note
            if panel.report_box:
                entry["boxes"] = len(panel.boxes or [])
                entry["largest_box"] = largest(panel.boxes or [])
            entries.append(entry)
        manifest["figures"][name] = entries
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    total = sum(p.stat().st_size for p in OUT_DIR.iterdir())
    print(f"wrote {sum(len(v) for v in figs.values())} panels in {len(figs)} figures "
          f"({total / 1024:.0f} KiB) and {MANIFEST}")


def main() -> None:
    from libreyolo.data.dataset import YOLODataset

    box = detect_person()
    with tempfile.TemporaryDirectory() as tmp:
        root = make_dataset_dir(Path(tmp) / "one-image", box)
        dataset = YOLODataset(data_dir=str(root), split="train", img_size=(IMGSZ, IMGSZ))
        write(figures(dataset, box), box)


if __name__ == "__main__":
    main()
