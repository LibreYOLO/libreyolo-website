"""Extract the augmentation facts the docs render from an installed LibreYOLO.

Usage, from the website root, in a Python environment where the release being
documented is installed (``pip install "libreyolo[rfdetr]==1.8.0"``):

    python scripts/build-augmentation/collect.py

Writes ``src/data/docs/augmentation.json``. The augmentation matrix page and
the augmentations guide render their tables from that file, so nothing in them
is typed by hand.

Sources, all read from the installed package:

- ``libreyolo.data.augment.spec``: the knob list, each family's status for
  each knob, and the notes behind them. The library pins this table to its
  pipelines with ``tests/unit/test_augment_spec.py``.
- Each family's ``TrainConfig`` subclass: the defaults a run starts from.
- ``libreyolo train``'s option list: which knobs the CLI accepts.

Two kinds of fact cannot be read off a data structure: the meaningful range of
each knob, and the places where the spec and the pipeline disagree. Both are
declared below, and both are checked against the library at run time, so this
script fails rather than publishing a claim the installed version no longer
supports.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import libreyolo  # noqa: E402
from libreyolo.data.augment import spec  # noqa: E402
from libreyolo.training import config as train_config  # noqa: E402

OUT = Path("src/data/docs/augmentation.json")

# Display order and grouping. Every key must be in spec.AUG_KNOBS and every
# spec knob must be here; the script checks both directions.
GROUPS = [
    ("detection", [
        "mosaic_prob", "mixup_prob", "hsv_prob", "flip_prob", "flipud",
        "degrees", "translate", "mosaic_scale", "shear", "perspective",
        "mixup_scale",
    ]),
    ("schedule", ["no_aug_epochs"]),
    ("classification", ["auto_augment", "erasing", "mixup", "cutmix", "scale", "crop_pct"]),
]

# The meaningful range of each knob, in notation that needs no translation.
# "checked" says whether the library rejects a value outside it; the checks
# below prove each claim against the installed version.
RANGES = {
    "mosaic_prob": ("[0, 1]", False),
    "mixup_prob": ("[0, 1]", False),
    "hsv_prob": ("[0, 1]", False),
    "flip_prob": ("[0, 1]", False),
    "flipud": ("[0, 1]", False),
    "degrees": ("≥ 0", False),
    "translate": ("≥ 0", False),
    "mosaic_scale": ("(min, max), 0 < min ≤ max", False),
    "shear": ("[0, 90)", False),
    "perspective": ("≥ 0", False),
    "mixup_scale": ("(min, max), 0 < min ≤ max", False),
    "no_aug_epochs": ("≥ 0", False),
    "auto_augment": ('None, "randaugment", "autoaugment", "augmix"', True),
    "erasing": ("[0, 1)", True),
    "mixup": ("[0, 1], mixup + cutmix ≤ 1", True),
    "cutmix": ("[0, 1], mixup + cutmix ≤ 1", True),
    "scale": ("0 < min ≤ max ≤ 1", True),
    "crop_pct": ("(0, 1], None", True),
}

# CLI spellings that differ from the config field. The classification
# ``mixup`` field shares its CLI name with the detection alias; the CLI routes
# it by task.
CLI_ALIASES = {"mosaic_prob": ["mosaic"], "mixup_prob": ["mixup"], "flip_prob": ["flip_prob", "fliplr"]}

# Families whose config cannot be found through the registry: task-dependent
# configs, and families whose model class needs an extra to import.
CONFIG_FALLBACK = {
    "yolonas": ("libreyolo.training.config", "YOLONASConfig"),
    "ec": ("libreyolo.training.config", "ECConfig"),
    "rfdetr": ("libreyolo.models.rfdetr.config", "RFDETRConfig"),
    "dinov2": ("libreyolo.models.dinov2.config", "DINOv2Config"),
}

# Multi-task families whose spec notes say a knob is read by one task only
# ("task='pose' only"). Their defaults table shows that task's config value,
# since the family's default config is the one that never reads it.
TASK_CONFIGS = {
    ("ec", "pose"): ("libreyolo.training.config", "ECPoseConfig"),
    ("dinov2", "classify"): ("libreyolo.models.dinov2.config", "DINOv2Config"),
}


def task_only(note: str) -> str | None:
    import re

    match = re.search(r"task='(\w+)' only", note or "")
    return match.group(1) if match else None


# Families documented on a lineage page rather than their own. The script
# checks each target page exists.
LINEAGE_PAGES = {
    "yolo9_e2e": "yolov9",
    "yolo9_p2": "yolov9",
    "rtdetrv2": "rt-detr",
    "rtdetrv4": "rt-detr",
    "deimv2": "deim",
}


def display_name(family: str) -> str:
    """The family's model page title where it has a page of its own, so the
    tables spell names as the rest of the site does; the spec's name otherwise."""
    import re

    registry = json.loads(Path("src/data/docs/registry.json").read_text())
    slug = (registry["families"].get(family) or {}).get("slug")
    if slug:
        page = Path(f"content/docs/models/{slug}.md").read_text()
        match = re.search(r'^title:\s*"?(.*?)"?\s*$', page, re.M)
        if match:
            return match.group(1)
    return spec.display_name(family)


def model_page(family: str) -> str | None:
    registry = json.loads(Path("src/data/docs/registry.json").read_text())
    slug = (registry["families"].get(family) or {}).get("slug") or LINEAGE_PAGES.get(family)
    if slug and not Path(f"content/docs/models/{slug}.md").exists():
        fail(f"{family} points at /docs/models/{slug}, which has no page")
    return f"/docs/models/{slug}" if slug else None


# Where the shipped spec says more than the pipeline does. Each entry is
# verified below; when the library is fixed the check fails and the entry is
# removed, so the page follows the spec again.
CORRECTIONS = [
    {
        "knob": "mixup_scale",
        "status": spec.IGNORED,
        "dataset": "libreyolo.data.augment.yolo9.YOLO9MosaicMixupDataset",
        "reason": (
            "This family trains through YOLO9MosaicMixupDataset, which stores "
            "mixup_scale but never reads it: its MixUp blends two full frames "
            "at a Beta(1, 1) ratio without rescaling the partner image."
        ),
    },
]


def fail(message: str) -> None:
    sys.exit(f"collect.py: {message}")


def default_of(field: dataclasses.Field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    return None


def py(value) -> str:
    """A default as a reader would type it in Python: 1.0, (0.5, 1.5), None."""
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(py(v) for v in value) + ")"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return repr(value)


def registry_classes():
    from libreyolo.models.base.model import BaseModel

    by_family: dict[str, list[type]] = {}
    for cls in BaseModel._registry:
        family = getattr(cls, "FAMILY", None)
        if family:
            by_family.setdefault(family, []).append(cls)
    return by_family


def family_config(family: str, classes: dict[str, list[type]]):
    for cls in classes.get(family, []):
        cfg = getattr(cls, "TRAIN_CONFIG", None)
        if cfg is not None and dataclasses.is_dataclass(cfg):
            return cfg
    if family in CONFIG_FALLBACK:
        module, name = CONFIG_FALLBACK[family]
        return getattr(importlib.import_module(module), name)
    return None


def trainers_by_family() -> dict[str, list[type]]:
    """Every trainer class in the package, keyed by the family it reports."""
    import pkgutil

    import libreyolo.models as models
    from libreyolo.training.trainer import BaseTrainer

    found: dict[str, list[type]] = {}
    for info in pkgutil.walk_packages(models.__path__, models.__name__ + "."):
        if not info.name.endswith("trainer"):
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception:  # a family behind a missing extra
            continue
        for obj in vars(module).values():
            if (isinstance(obj, type) and issubclass(obj, BaseTrainer)
                    and obj is not BaseTrainer and obj.__module__ == module.__name__):
                try:
                    family = obj.get_model_family(None)
                except Exception:
                    continue
                found.setdefault(family, []).append(obj)
    return found


# The archetype a family's table is built from, read from spec.py's own source:
# ``dict(_YOLOX_STYLE)`` or ``{**_DETR_STYLE, ...}``. A family written out as a
# literal table is its own archetype, except PP-YOLOE, which the spec comments
# describe as reusing YOLO-NAS's affine and MixUp wrapper.
ARCHETYPE_BASES = {
    "_YOLOX_STYLE": "yolox_style",
    "_DETR_STYLE": "detr_style",
    "_CLASSIFY_STYLE": "classification",
    "_SEMANTIC_STYLE": "semantic",
}
LITERAL_ARCHETYPES = {"yolonas": "yolonas", "ppyoloe": "yolonas", "nafnet": "restore"}


def archetypes_from_source() -> dict[str, str]:
    import ast

    tree = ast.parse(inspect.getsource(spec))
    table = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "FAMILY_AUG_SUPPORT":
            table = node.value
    if not isinstance(table, ast.Dict):
        fail("could not find FAMILY_AUG_SUPPORT in spec.py")
    result = {}
    for key_node, value in zip(table.keys, table.values):
        family = ast.literal_eval(key_node)
        base = None
        if isinstance(value, ast.Call) and value.args and isinstance(value.args[0], ast.Name):
            base = value.args[0].id
        elif isinstance(value, ast.Dict):
            spreads = [v.id for k, v in zip(value.keys, value.values) if k is None and isinstance(v, ast.Name)]
            base = next((s for s in spreads if s in ARCHETYPE_BASES), None)
        if base in ARCHETYPE_BASES:
            result[family] = ARCHETYPE_BASES[base]
        elif family in LITERAL_ARCHETYPES:
            result[family] = LITERAL_ARCHETYPES[family]
        else:
            fail(f"{family} has a literal table and no archetype; add it to LITERAL_ARCHETYPES")
    return result


def check_ranges() -> None:
    """Prove each "checked" range claim, and the unchecked ones too."""
    from libreyolo.data.augment.classify import (
        build_classify_transforms,
        normalize_auto_augment,
        normalize_crop_scale,
        validate_mix_probabilities,
    )
    from libreyolo.data.augment.geometry import get_affine_matrix

    def rejects(fn) -> bool:
        try:
            fn()
        except ValueError:
            return True
        return False

    # Detection knobs: the config takes anything.
    for knob, bad in [("mosaic_prob", 1.5), ("degrees", -5.0), ("erasing", 1.0), ("shear", 120.0)]:
        try:
            train_config.TrainConfig(**{knob: bad})
        except Exception as exc:  # pragma: no cover - a behavior change
            fail(f"TrainConfig now rejects {knob}={bad} ({exc}); update RANGES")
    # Scale must be positive and perspective non-negative when drawn.
    if not rejects(lambda: get_affine_matrix((64, 64), 0.0, 0.0, (0.0, 0.0), 0.0)):
        fail("get_affine_matrix no longer rejects a zero scale")
    if not rejects(lambda: get_affine_matrix((64, 64), 0.0, 0.0, (1.0, 1.0), 0.0, perspective=-0.001)):
        fail("get_affine_matrix no longer rejects a negative perspective")
    # Classification knobs are validated when the pipeline is built.
    if not rejects(lambda: build_classify_transforms(64, True, erasing=1.0)):
        fail("erasing=1.0 is accepted; RANGES says [0, 1)")
    if rejects(lambda: build_classify_transforms(64, True, erasing=0.99)):
        fail("erasing=0.99 is rejected; RANGES says [0, 1)")
    if not rejects(lambda: validate_mix_probabilities(0.6, 0.6)):
        fail("mixup + cutmix > 1 is accepted")
    if not rejects(lambda: normalize_auto_augment("foo")):
        fail("an unknown auto_augment policy is accepted")
    if not rejects(lambda: normalize_crop_scale(1.5)) or normalize_crop_scale(0.5) != (0.5, 1.0):
        fail("scale normalization changed")
    if not rejects(lambda: build_classify_transforms(64, False, crop_pct=1.5)):
        fail("crop_pct=1.5 is accepted")


def check_corrections() -> list[dict]:
    trainers = trainers_by_family()
    applied = []
    for corr in CORRECTIONS:
        module, name = corr["dataset"].rsplit(".", 1)
        dataset = getattr(importlib.import_module(module), name)
        mixup_source = inspect.getsource(dataset._mixup)
        if corr["knob"] in mixup_source:
            fail(f"{name}._mixup now reads {corr['knob']}; remove the correction")
        families = []
        for family in spec.FAMILY_AUG_SUPPORT:
            sources = [inspect.getsource(t.create_transforms) for t in trainers.get(family, [])
                       if hasattr(t, "create_transforms")]
            if any(name in s for s in sources):
                families.append(family)
        if not families:
            fail(f"no family trains through {name}; remove the correction")
        for family in families:
            if spec.FAMILY_AUG_SUPPORT[family][corr["knob"]].status == corr["status"]:
                fail(f"spec already marks {family}.{corr['knob']} {corr['status']}; remove the correction")
        applied.append({**{k: v for k, v in corr.items() if k != "dataset"}, "families": families,
                        "spec_status": spec.FAMILY_AUG_SUPPORT[families[0]][corr["knob"]].status})
    return applied


def cli_options() -> set[str]:
    from libreyolo.cli.commands.train import train_cmd

    return set(inspect.signature(train_cmd).parameters)


def main() -> None:
    order = [k for _, keys in GROUPS for k in keys]
    if set(order) != set(spec.AUG_KNOBS):
        fail(f"knob list drifted: missing {set(spec.AUG_KNOBS) - set(order)}, extra {set(order) - set(spec.AUG_KNOBS)}")
    if set(RANGES) != set(order):
        fail("RANGES does not cover every knob")

    check_ranges()
    classes = registry_classes()
    corrections = check_corrections()
    corrected = {(f, c["knob"]): c for c in corrections for f in c["families"]}
    archetypes = archetypes_from_source()

    base_defaults = {f.name: default_of(f) for f in dataclasses.fields(train_config.TrainConfig)}
    options = cli_options()

    knobs = []
    for group, keys in GROUPS:
        for key in keys:
            spellings = CLI_ALIASES.get(key, [key])
            cli = [s for s in spellings if s in options]
            if key == "mixup":
                cli = ["mixup"] if "mixup" in options else []
            range_text, checked = RANGES[key]
            knobs.append({
                "key": key,
                "group": group,
                "cli": cli,
                "default": py(base_defaults.get(key)),
                "range": range_text,
                "checked": checked,
                "description": spec.AUG_KNOBS[key],
            })

    families = []
    for family, table in spec.FAMILY_AUG_SUPPORT.items():
        cfg = family_config(family, classes)
        if cfg is None:
            fail(f"no TrainConfig found for {family}")
        defaults = {f.name: default_of(f) for f in dataclasses.fields(cfg)}
        default_tasks = {}
        for key in order:
            task = task_only(table[key].note)
            if task is None:
                continue
            if (family, task) not in TASK_CONFIGS:
                fail(f"{family}.{key} is read by task={task} only; add its config to TASK_CONFIGS")
            module, name = TASK_CONFIGS[(family, task)]
            task_cfg = getattr(importlib.import_module(module), name)
            defaults[key] = default_of(next(f for f in dataclasses.fields(task_cfg) if f.name == key))
            default_tasks[key] = task
        archetype = archetypes[family]
        support, notes = {}, {}
        for key in order:
            status, note = table[key].status, table[key].note
            if (family, key) in corrected:
                status, note = corrected[(family, key)]["status"], corrected[(family, key)]["reason"]
            support[key] = status
            if note:
                notes[key] = note
        families.append({
            "key": family,
            "name": display_name(family),
            "page": model_page(family),
            "archetype": archetype,
            "config": cfg.__name__,
            "support": support,
            "notes": notes,
            "defaults": {k: py(defaults.get(k)) for k in order},
            "default_tasks": default_tasks,
            "extra_knobs": {
                k: {"description": d, "default": py(defaults.get(k))}
                for k, d in spec.FAMILY_EXTRA_AUG_KNOBS.get(family, {}).items()
            },
        })

    data = {
        "libreyolo_version": libreyolo.__version__,
        "statuses": [spec.USED, spec.GATED_BY_MOSAIC, spec.IGNORED],
        "groups": [g for g, _ in GROUPS],
        "knobs": knobs,
        "families": families,
        "corrections": corrections,
    }
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(knobs)} knobs, {len(families)} families, "
          f"{sum(len(c['families']) for c in corrections)} corrected cells, LibreYOLO {libreyolo.__version__}")


if __name__ == "__main__":
    main()
