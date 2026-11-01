"""Build the Speridlabs Idol gallery: compact 3D objects, photos and a manifest.

Usage (any Python with numpy and Pillow):
    python scripts/idol-gallery/build.py SOURCE_DIR

SOURCE_DIR holds the Speridlabs Idol results listed in objects.json: the .ply
each object was saved as, and the photo it was generated from. Writes
public/idol-gallery/<slug>.lyg and <slug>.webp, and src/data/idol-gallery.json.

The .lyg format is gzip over a 40-byte header and four arrays, about 17 bytes
per Gaussian before compression instead of 56 in the .ply:
    header: b"LYG1", uint32 count, float32 pos_min[3], pos_max[3],
            float32 log_scale_min, log_scale_max
    uint16 position[count][3]   linear over [pos_min, pos_max]
    uint8  log_scale[count][3]  linear over [log_scale_min, log_scale_max]
    uint8  rotation[count][4]   w, x, y, z of the unit quaternion, 128 + 128 q
    uint8  rgba[count][4]       color and opacity, 255 = 1
"""

import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SH_C0 = 0.28209479177387814
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public" / "idol-gallery"
MANIFEST = ROOT / "src" / "data" / "idol-gallery.json"
PHOTO_SIDE = 480


def read_ply(path):
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("latin1").splitlines()
    names = [line.split()[2] for line in header if line.startswith("property float ")]
    meta = {}
    for line in header:
        if line.startswith("comment libreyolo "):
            meta = json.loads(line[line.index("{"):])
    rows = np.frombuffer(data[end:], dtype="<f4").reshape(-1, len(names))
    return {name: rows[:, i] for i, name in enumerate(names)}, meta


def encode(columns):
    pos = np.stack([columns["x"], columns["y"], columns["z"]], 1).astype(np.float64)
    log_scale = np.stack([columns[f"scale_{k}"] for k in range(3)], 1).astype(np.float64)
    quat = np.stack([columns[f"rot_{k}"] for k in range(4)], 1).astype(np.float64)
    rgb = np.clip(0.5 + SH_C0 * np.stack([columns[f"f_dc_{k}"] for k in range(3)], 1), 0, 1)
    alpha = 1 / (1 + np.exp(-columns["opacity"].astype(np.float64)))

    keep = alpha >= 1 / 255  # the renderer skips the rest anyway
    pos, log_scale, quat, rgb, alpha = pos[keep], log_scale[keep], quat[keep], rgb[keep], alpha[keep]
    count = len(pos)

    pos_min, pos_max = pos.min(0), pos.max(0)
    span = np.maximum(pos_max - pos_min, 1e-9)
    pos_q = np.round((pos - pos_min) / span * 65535).astype("<u2")
    s_min, s_max = float(log_scale.min()), float(log_scale.max())
    scale_q = np.round((log_scale - s_min) / max(s_max - s_min, 1e-9) * 255).astype(np.uint8)
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    quat_q = np.clip(np.round(quat * 128 + 128), 0, 255).astype(np.uint8)
    rgba_q = np.round(np.concatenate([rgb, alpha[:, None]], 1) * 255).astype(np.uint8)

    header = b"LYG1" + struct.pack("<I3f3f2f", count, *pos_min, *pos_max, s_min, s_max)
    body = header + pos_q.tobytes() + scale_q.tobytes() + quat_q.tobytes() + rgba_q.tobytes()
    return gzip.compress(body, compresslevel=9, mtime=0), count


def square_photo(src, dst):
    image = Image.open(src).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((PHOTO_SIDE, PHOTO_SIDE), Image.LANCZOS)
    image.save(dst, "WEBP", quality=80, method=6)


def main(source):
    objects = json.loads((Path(__file__).parent / "objects.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for obj in objects:
        columns, meta = read_ply(source / obj["ply"])
        blob, count = encode(columns)
        (OUT / f"{obj['slug']}.lyg").write_bytes(blob)
        square_photo(source / obj["photo"], OUT / f"{obj['slug']}.webp")
        manifest.append({
            "slug": obj["slug"],
            "title": obj["title"],
            "artist": obj["artist"],
            "file": f"/idol-gallery/{obj['slug']}.lyg",
            "photo": f"/idol-gallery/{obj['slug']}.webp",
            "photo_credit": obj["photo_credit"],
            "gaussians": count,
            "bytes": len(blob),
            "seed": meta.get("seed"),
            "steps": meta.get("steps"),
            "mode": meta.get("mode"),
            "seconds": obj["seconds"],
            "camera": meta.get("camera", {}),
        })
        print(f"{obj['slug']:24s} {count:7d} Gaussians  {len(blob) / 1e6:5.2f} MB")
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
