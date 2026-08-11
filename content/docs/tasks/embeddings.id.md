---
title: Embeddings
seo_title: Image and region embeddings in LibreYOLO
description: >-
  The embed task returns L2-normalized float32 vectors for a whole image, for
  each detected region, or for text. Enroll a gallery, match by cosine
  similarity, and search from Python or the CLI.
lead: >-
  One task covers every vector LibreYOLO produces. embed returns unit-length
  float32 rows whose dot product is a similarity score, whether the row
  describes a whole image, a single detected face, or a line of text, and the
  same Gallery matches all of them.
keywords:
  - image embeddings python
  - l2 normalized embedding
  - cosine similarity search
  - libreyolo embed task
  - image retrieval
  - gallery enroll
  - clip embeddings
  - dinov2 embeddings
  - reid embeddings
last_verified: 1.5.0
verification: >-
  Task key and aliases read from libreyolo/tasks.py. Result payloads from the
  Embeddings and Identities classes in libreyolo/utils/results.py. Gallery API
  from libreyolo/utils/gallery.py. embed and _postprocess_embeddings from
  libreyolo/models/base/model.py. Supported families located by searching
  libreyolo/models/**/model.py for embed in SUPPORTED_TASKS. CLI surface from
  libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py and
  libreyolo/cli/commands/predict.py. Design intent from
  docs/adr/0015-embed-generalization.md.
meta:
  - label: Task key
    value: embed
    mono: true
  - label: Aliases
    value: 'face-recognition, reid, face'
    mono: true
  - label: Result payloads
    value: 'Embeddings, Identities'
    mono: true
  - label: Row dtype
    value: 'float32, unit length'
snippets:
  predict:
    - label: Whole image
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # CLIP defaults to classify, so ask for the vector explicitly.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512), one row per image
        print(result.boxes)                  # None: nothing was localized
    - label: Per region
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # Row i describes the region in box i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Many images at once
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Every row from every result, concatenated into one tensor.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Text
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # Text is a method, never a prediction source. A string passed to
        # model(...) is still a path or a URL.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: Compare two sets of rows
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # Rows are unit length, so cosine similarity is a dot product.
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: Image against text
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Enroll and identify
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # name is None below the threshold
    - label: Top-k search
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(name, score), ...] for the first row
    - label: Enroll a vector you already hold
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # normalized on the way in
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Enroll a folder tree
      language: bash
      code: >
        # source/<identity>/*.jpg. An existing gallery is extended in place.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: Identify while predicting
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Compare two images
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify is the same command under a second name.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## Definition

`embed` turns an image, a region of an image, or a string into a float32 row of
fixed width whose length is one. Because every row is a unit vector, comparing
two of them is a dot product, and comparing two sets of them is a single matrix
multiplication. Nothing else in the task is model specific: retrieval,
duplicate detection, re-identification and face recognition are all the same
arithmetic over different rows.

The vector is the output. There is no class list, so a name is attached later by
comparing against references you supply rather than by anything the network was
trained to predict.

### Three shapes

| Shape | `Results.embeddings` | `Results.boxes` | Produced by |
|---|---|---|---|
| Whole image | `(1, D)` | `None` | Passing an image to a whole-image family |
| Region | `(N, D)` | `(N, 4)`, row-aligned | Families that localize first, such as face recognition |
| Text | not a `Results` at all | | `model.embed_text(texts)`, returning `(M, D)` |

A whole-image result stays two dimensional even for one image. `(D,)` is not a
permitted return shape, so a consumer never has to special-case the single-row
case. Text returns a plain tensor rather than a `Results`, because a string is
not an image source: passing one to `model(...)` still means a path or a URL,
and the library never guesses that a string is prose.

The canonical task key is `embed`. `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` and
`reid` all normalize to it, so `task="reid"` and `task="embed"` select exactly
the same thing.

## Models

Four families serve the task, and they split cleanly by whether they localize
anything first.

| Family | Shape | Dimension | Also supports |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Region, one row per detected face | 512 | Nothing; `embed` is its only task |
| [CLIP](/docs/models/clip) | Whole image, with a paired text tower | 512 for `b32` and `b16`, 768 for `l14` | `classify`, which stays its default |
| [SigLIP 2](/docs/models/siglip2) | Whole image, with a paired text tower | 768 for `b16`, 1152 for `so400m` | `classify`, which stays its default |
| [DINOv2](/docs/models/dinov2) | Whole image, image only | 384 | `semantic`, `classify` |

CLIP and SigLIP 2 keep `classify` as their default task, so `task="embed"` has
to be asked for. Their existing `-cls` checkpoint is the shared two-tower
artifact; no duplicate `-embed` checkpoint is published for identical weights.

`embed_text` exists only on CLIP and SigLIP 2, the two families with a text
tower. DINOv2 has none. DINOv2 embedding bypasses the semantic and
classification heads and reads the final normalized CLS token at 224 pixels; the
`n`, `s`, `m` and `l` variants all share the DINOv2-S encoder, so all four
return `D = 384`.

The classification-only backbones added in this release, [ViT](/docs/models/vit),
[Swin](/docs/models/swin) and [DeiT](/docs/models/deit), declare `classify` only
and do not serve this task.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` is the batch shortcut: it runs `predict` and
concatenates every row from every result into one `(N_total, D)` CPU float32
tensor, raising if the rows have mixed dimensions. A family without `embed` in
its supported tasks raises `NotImplementedError`.

## Result payloads

`result.embeddings` is an `Embeddings` payload. Its `data` is always `(N, D)`
float32, already L2-normalized by the inference path, and a non-two-dimensional
input raises rather than being reshaped silently.

| Member | Meaning |
|---|---|
| `.data` | The `(N, D)` matrix |
| `.dim` | `D` |
| `.normalized` | The same rows, defensively re-normalized |
| `.similarity(other)` | `(N, M)` against another set, or `(N,)` against a single `(D,)` vector |
| `.verify(i, j, threshold=0.4)` | Whether rows `i` and `j` are the same subject |

`result.identities` is an `Identities` payload, present only when a gallery was
passed. It is a plain container, not a tensor, so moving a `Results` between
devices leaves it alone.

| Member | Meaning |
|---|---|
| `.name` | List of names, `None` where nothing cleared the threshold |
| `.score` | `(N,)` float32 best cosine score, kept even when the name is `None` |
| `.data` | List of `(name, score)` tuples |

<code-tabs name="similarity" />

Vectors are left out of `summary()` and `to_json()` by default, since a 512-float
row is about two kilobytes per subject. Each row reports `embedding_dim`
instead, plus `identity` and `identity_score` when a gallery was used. Pass
`summary(embeddings=True)` to include the numbers.

## Galleries

A `Gallery` is a named set of reference rows. It stores each reference
separately rather than averaging them, so a name is scored by its single best
matching reference, and adding a bad photo cannot drag an identity's centroid
around.

<code-tabs name="gallery" />

`Gallery(model)` binds to the weights that will produce its vectors.
`enroll(name, sources, select="best")` runs prediction on each source and keeps
the highest-confidence row per result; `select="all"` keeps every row instead,
which is what you want when a reference image legitimately contains several
subjects. `enroll_embedding(name, vector)` skips inference and takes a vector
directly, normalizing it and rejecting an all-zero row.

`FaceGallery` is a permanent alias of the same class, and archives written by
earlier face-only releases still load.

### Matching and thresholds

Matching is a dense matrix multiplication against every stored reference,
reduced to one score per name by taking the maximum. There is no approximate
index, which keeps the numbers exact and puts a practical ceiling on gallery
size.

Two entry points differ in what they do below the threshold. `match()` returns
`[(name, score), ...]` per row with everything under the threshold dropped, so a
row with no match is an empty list. `identify()` returns an `Identities` payload
that always keeps the best score and sets the name to `None` when it is under
the threshold. Neither ever substitutes the nearest below-threshold name.

The default threshold is `0.4` throughout. It is a cosine value, not a
probability, and the right operating point is a property of your data and your
tolerance for false matches, so sweep it on labeled pairs rather than accepting
the default. `libreyolo enroll` and the `gallery=` prediction argument use the
same number.

### Persistence

`save(path)` writes a compressed `.npz` holding the vectors, the names and a
metadata block carrying the format version, the embedding dimension and a
fingerprint of the weights that produced the rows. `Gallery.load(path,
model=...)` checks both before comparing anything, so pointing a gallery at a
different model raises instead of silently scoring vectors from two unrelated
spaces against each other. Saving an empty gallery is refused.

## Command line

| Command | Purpose |
|---|---|
| `libreyolo enroll` | Walk a folder-per-identity tree and write or extend a `.npz` gallery |
| `libreyolo compare` | Embed the primary subject in two images and report cosine similarity |
| `libreyolo verify` | The same command under a second name |
| `libreyolo predict gallery=...` | Attach identities to an ordinary prediction run |

<code-tabs name="cli" />

Every LibreYOLO command accepts both `key=value` and `--key value`, so
`gallery=refs.npz` and `--gallery refs.npz` are the same argument.

`enroll` takes `model`, `source` and `gallery`, plus optional `face-detector`,
`device`, `--json` and `--quiet`. It reads one folder per identity, where the
folder name is the identity and every image inside contributes references:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

An image that yields nothing is skipped with a line on stderr rather than
aborting the run, and the summary reports how many references were stored for
each name. An existing gallery file is extended in place, so identities can be
added over time.

`compare` and `verify` are one function registered twice. They take `model`,
`source`, `source2` and an optional `threshold`, and print the cosine
similarity, the same-or-different verdict and the threshold that produced it.
`--json` prints the same three fields as an object.

On `predict`, `gallery` points at a saved `.npz` and `gallery_threshold`
overrides the `0.4` default. Passing a gallery to a model whose task is not
`embed` is an error rather than a silent no-op, and a missing gallery file
suggests the `libreyolo enroll` command that would create it.

## Faces

Face recognition is the region shape of this task, and it is the only shipped
implementation of that shape. It adds a detection and alignment stage in front
of the embedding head, plus a `verify()` method, a bring-your-own-boxes
argument, published accuracy numbers and calibration guidance for the threshold.
All of that lives on [face recognition](/docs/tasks/face-recognition), which is
the walkthrough to follow when the subject is faces. Everything on this page
applies to it unchanged.

## Train, validate and export

Nothing in this task trains inside LibreYOLO. The face embedding head is an
ONNX artifact whose `train()`, `val()` and `export()` all raise; train a head
upstream and load the file by path. CLIP, SigLIP 2 and DINOv2 train and export
through their classification and segmentation tasks, not through `embed`.

There is no retrieval validator. Measure verification accuracy on labeled pairs
by sweeping `threshold`, and identification accuracy by enrolling a gallery and
reading `identities.name` and `identities.score` on held-out images, counting a
`None` name as a rejection.


