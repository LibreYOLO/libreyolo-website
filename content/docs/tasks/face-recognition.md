---
title: Face recognition
seo_title: "Face recognition in LibreYOLO"
description: "Detect, embed and identify faces in LibreYOLO. Enroll a gallery, compare two images and match by cosine similarity, from Python or the CLI."
lead: "Face recognition is the embed task applied to faces. A detector locates and aligns every face, a recognition head returns an L2-normalized vector per face, and identity is decided by cosine similarity against enrolled references rather than by a fixed class list."
keywords: [face recognition python, face embedding, face verification, face gallery, arcface onnx, libreyolo embed task, cosine similarity faces]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-* names route to the face-embedding family regardless
        # of file suffix, and download from the LibreYOLO Hugging Face org on
        # first use along with the default face detector.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # (N, 4) face boxes
        print(result.embeddings.data.shape)  # (N, D), one row per face
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Compare two images
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Runs detection and embedding on both images and compares their
        # most confident face. Cosine similarity is in [-1, 1].
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: Enroll a gallery and identify
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # name is None below the threshold
    - label: Enroll and identify from the CLI
      language: bash
      code: |
        libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=faces.npz
        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg gallery=faces.npz
    - label: Bring your own face boxes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes skips detection entirely; face_detector accepts a
        # callable, a LibreYOLO detection model, or a FaceDetector instance.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
---

## Definition

Face recognition returns a vector per face, not a label. Prediction runs two
stages: a face detector locates each face and its five landmarks, the crop is
warped to a canonical 112x112 alignment, and a recognition head emits an
L2-normalized embedding.

`result.embeddings` is an `Embeddings` payload of shape `(N, D)`, row-aligned
with `result.boxes`, so row `i` describes the face in box `i`. Because rows are
unit vectors, cosine similarity is a dot product, and `embeddings.similarity()`
computes it against another `Embeddings` or a whole matrix in one call.

Naming a face is a separate step. A `Gallery` holds named reference vectors;
passing `gallery=` to `predict()` attaches `result.identities`, row-aligned with
the embeddings, carrying a name and its best cosine score per face. A face below
the match threshold keeps `None` as its name, and the nearest below-threshold
name is never substituted.

The library's canonical task key is `embed`. `face-recognition`, `facial-recognition`,
`reid` and `face` all normalize to it, so `task="face-recognition"` and
`task="embed"` select the same thing.

## Models

[LibreFaceRec](/docs/models/librefacerec) is the family for this task. It is two
ONNX artifacts behind one call: `librefacerec-l.onnx`, an iResNet100 recognition
head producing 512-d embeddings, and `librefacerec-det.onnx`, the default face
detector with five landmarks, taken from the OpenCV zoo. Both download from the
LibreYOLO Hugging Face org on first use. Any other ArcFace-convention ONNX file
(aligned 112x112 in, `(N, D)` out) can replace the recognition head by passing
its path instead of a `librefacerec-*` name.

The `embed` task key is wider than faces. [CLIP](/docs/models/clip),
[SigLIP2](/docs/models/siglip2) and [DINOv2](/docs/models/dinov2) also support
`task="embed"` and return one whole-image vector, which is image retrieval rather
than face identity. They share the `Gallery` and `Embeddings` API, so the
enroll-and-match workflow below transfers, but they do not detect or align faces.

The recognition head runs through `onnxruntime`, which the base install does not
carry:

```bash
pip install "libreyolo[onnx]"
```

## Predict

<code-tabs name="predict" />

Left alone, `predict()` downloads and pairs the default detector. `face_detector`
overrides it with a callable, a LibreYOLO detection model, or a `FaceDetector`
instance, and can be set on the constructor or per call. `face_boxes` bypasses
detection with boxes you already hold. On the CLI, `face_detector=` accepts a
face-detector `.onnx` path or a LibreYOLO detector name.

`model.verify(image_a, image_b)` is the two-image shortcut: it embeds the most
confident face in each and returns `{"similarity", "same_person", "threshold"}`.
`model.embed(sources)` returns every face row across one or more images stacked
into a single `(N_total, D)` tensor. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Dataset format

Enrollment reads a folder per identity. The folder name becomes the identity,
and every image inside it contributes references for that name:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` walks that tree and writes a `.npz` gallery. An existing
gallery file is extended in place rather than replaced, so identities can be
added over time. Galleries are bound to the weights that produced them by
embedding dimension and a file fingerprint; matching with a different model
raises instead of comparing incompatible vector spaces.

By default each source image contributes one reference row, the most confident
face, so a portrait containing bystanders enrolls only its subject. Pass
`select="all"` to `Gallery.enroll` to store every returned row.

## Train

No family in this task trains inside LibreYOLO. `LibreFaceEmbedder.train()`
raises: train a recognition head upstream, export it to ONNX in the ArcFace
convention, and load the file by path.

## Validate

There is no dataset validator for this task, and `val()` raises rather than
pretending otherwise. Verification accuracy is measured on labeled image pairs
with `model.verify()`, sweeping `threshold` to pick the operating point you
want. Identification accuracy is measured by enrolling a gallery and reading
`result.identities.name` and `result.identities.score` on held-out images,
counting a `None` name as a rejection.

## Export

The recognition head is already an ONNX graph, so there is nothing to convert:
`LibreFaceEmbedder.export()` raises. Deploy the `.onnx` file directly, or point
LibreYOLO at it and let the family handle detection, alignment and
normalization.
