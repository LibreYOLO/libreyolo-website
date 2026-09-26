---
title: LibreMODUS
families: [libremodus]
seo_title: "LibreMODUS in LibreYOLO: any-to-any image analysis"
description: "Use LibreMODUS in LibreYOLO for depth, normals, edges and detection, and to compose them with any2any(). Inference-only; weights load from EPFL-VILAB."
lead: "LibreMODUS is an inference-only integration of the MODUS 14B-A7B checkpoint, an any-to-any model that turns one image-derived input into another: RGB in, depth out; depth in, normals out; any of those plus a phrase, boxes out. LibreYOLO supports four tasks through the standard predict API and a wider set through any2any()."
keywords: [LibreMODUS, MODUS, any-to-any, depth estimation, surface normals, edge detection, referring detection, EPFL VILAB]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # With no custom vocabulary, detect decodes the checkpoint's COCO
        # label tokens into contiguous COCO-80 class ids.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Phrase grounding
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() switches detection to phrase grounding: each phrase
        # runs independently and returns through the same Boxes contract.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS()

        # One to three image-derived inputs (rgb, depth, normal, canny/edge),
        # plus optional auxiliary text, composed toward one target.
        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )
        normals = result.normal_map.data

        # Grounding through any2any() needs a text input naming the phrase.
        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )
        print(result.boxes.xyxy)
---

## Install

LibreMODUS needs its own extra, which pulls in `accelerate` for the big-model dispatch this checkpoint needs.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO does not redistribute or mirror MODUS weights. By default, loading a `LibreMODUS` model downloads the required files directly from `EPFL-VILAB/MODUS` at a pinned Hugging Face revision, and a fresh download always needs the user's own authenticated Hugging Face account, even if the upstream hosting gate is temporarily open. Review and accept the upstream terms, then authenticate:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

To avoid any network request, point at a snapshot you already have:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

That directory must contain `model.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt`. See Licensing below for what the checkpoint's terms permit.

## Predict

<code-tabs name="predict" />

The standard task API covers four tasks, each mapped to one MODUS target: `depth` to relative depth (`result.depth_map`), `normal` to surface normals (`result.normal_map`), `edge` to Canny-style edges (`result.edges`), and `detect` to COCO-80 boxes (`result.boxes`) unless `set_classes()` switches it to phrase grounding. `set_task()` switches between them on the same loaded model. The released recipe uses ten flow-sampling steps with text guidance 4.0 and image guidance 2.0; override them with `inference_steps=`, `inference_cfg=` and `inference_image_cfg=` at construction.

`any2any()` reaches the wider public analysis surface: one to three image-derived inputs (`rgb`, `depth`, `normal`, `canny`/`edge`), plus optional auxiliary text, composed toward any one of depth, normals, edges, SAM-derived edges, COCO detection or phrase grounding. All image-derived inputs must describe the same aligned canvas; LibreMODUS rejects mismatched widths and heights rather than resizing them independently. `chain=(...)` generates intermediate targets and feeds them back into the same context, within the checkpoint's three-condition training budget. `verify=N` (N >= 2) generates N candidates and keeps the one that scores highest on a constrained self-consistency check, exposed as `result.verification_score`.

`dtype="bf16"` (the default) matches the released checkpoint precision; `dtype="fp8"` stores eligible decoder-trunk linear weights as E4M3 with a per-output-channel scale, converts once into a local cache under `~/.cache/libreyolo/modus/fp8`, and dequantizes to the input dtype per matrix multiply, so it trades memory rather than trading accuracy at the activation level.

`train()`, `val()` and `export()` all raise: LibreMODUS is inference-only, dataset validation is not offered, and there is no ONNX, TensorRT or TFLite export path. Batched `predict()` and test-time augmentation are also not supported; each call handles one image.

## Licensing

<provenance-box>

LibreYOLO does not host or mirror the MODUS checkpoint anywhere, including on its own Hugging Face org: loading it always pulls the pinned revision directly from EPFL-VILAB/MODUS, or reads a snapshot already on disk at `checkpoint_path`.

</provenance-box>

## Citation

<citation-block />
