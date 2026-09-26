---
title: LibreFaceRec
families: [facerec]
seo_title: "LibreFaceRec: face recognition and verification"
description: "Use LibreFaceRec in LibreYOLO for face detection, embedding and verification. Install and predict; the embedding weights are Apache-2.0."
lead: "LibreFaceRec is LibreYOLO's face-embedding task: a face detector locates and aligns faces, and a recognition head produces an L2-normalized identity embedding for verification or search."
keywords: [LibreFaceRec, face recognition, face embedding, face verification, ArcFace]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-* names route to this family regardless of file
        # suffix and download from the LibreYOLO Hugging Face org on first
        # use, along with the default face detector.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D), L2-normalized
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Verify
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Compares the most prominent face in each image via cosine
        # similarity of their L2-normalized embeddings.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: Gallery search
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # this image's faces
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # (query_faces, N_total) cosine similarities.
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
---

## Install

LibreFaceRec's recognition head runs through `onnxruntime`, which is not part
of the base install.

```bash
pip install "libreyolo[onnx]"
```

## Predict

<code-tabs name="predict" />

Detection and recognition are two separate ONNX graphs behind one call: a
face detector locates and aligns each face to a canonical crop, and the
recognition head returns an L2-normalized embedding per face. Left alone,
`predict()` downloads and pairs the bundled default detector automatically.
`face_detector` accepts a callable, a LibreYOLO detection model, or a
`FaceDetector` instance; `face_boxes` bypasses detection entirely with boxes
you already have. `result.embeddings` holds one row per detected face,
aligned with `result.boxes`; its `.similarity()` method computes cosine
similarity against another embedding or a whole gallery in one call. For
comparing two images directly rather than two already-computed embeddings,
`model.verify(image_a, image_b)` runs detection and embedding on both and
compares their most confident face. Any other ArcFace-convention ONNX
recognition model (aligned crop in, `(N, D)` embeddings out) can be
substituted by passing its file path instead of a `librefacerec-*` name. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Export

<export-matrix />

LibreFaceRec already wraps a pre-exported ONNX graph; re-exporting it to
another format is not implemented.

## Licensing

<provenance-box>

The bundled default face detector is a second artifact under a second
license: OpenCV Zoo's YuNet, MIT, copyright Shiqi Yu. No architecture code is
ported from either project; both graphs are consumed opaquely through
`onnxruntime`, so LibreYOLO's own wrapper carries no third-party code and is
MIT throughout.

</provenance-box>

## Citation

<citation-block />
