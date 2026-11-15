---
title: Qwen3-VL-Embedding
families: [qwen3vl_embedding]
seo_title: "Qwen3-VL-Embedding in Python: image and text retrieval vectors"
description: "Run Qwen3-VL-Embedding-2B in LibreYOLO: image and text embeddings in one 2048-dimensional space for search and retrieval. Apache-2.0."
lead: "Qwen3-VL-Embedding is Qwen's embedding model built on Qwen3-VL. It maps images and text into one vector space, so a text query can find the page or photo it describes. LibreYOLO runs the 2B size on the embed task, the first VLM-based embedder in the library."
keywords: [Qwen3-VL-Embedding, Qwen, multimodal embedding, image retrieval, text to image search, document retrieval, embeddings, LibreVLM]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-embedding-2b")   # weights download on first use
        result = model.predict(SAMPLE_IMAGE)
        print(result.embeddings.shape)              # (1, 2048), unit length
    - label: Retrieval
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("qwen3-vl-embedding-2b")
        pages = model.embed("pages/")               # (N, 2048), one row per image

        # Queries carry a task instruction; images keep the default one.
        q = model.embed_text(
            ["total due on the invoice"],
            prompt="Retrieve images or text relevant to the user's query.",
        )                                           # (1, 2048)
        scores = q @ pages.T                        # cosine similarity
---

## Install

Qwen3-VL-Embedding belongs to LibreYOLO's VLM tier and needs the `vlm` extra.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights download on first use from `LibreYOLO/LibreQwen3VLEmbedding2b`, a
mirror of `Qwen/Qwen3-VL-Embedding-2B` with unchanged parameters. The
architecture runs through the transformers Qwen3-VL implementation.

<code-tabs name="predict" />

The family loads through `LibreVLM()`, or the `LibreQwen3VLEmbedding` class.
It follows the same [embed](/docs/tasks/embeddings) contract as CLIP, SigLIP2 and
PE: `predict(image)` returns `Results.embeddings` of shape `(1, 2048)`,
`embed(source)` stacks one row per image, and `embed_text(texts, prompt=None)`
returns text rows in the same space. Rows are L2-normalized, so a dot product
is a cosine similarity.

Each input is a chat: a system instruction, then the image or the text. The
hidden state at the last token, normalized, is the vector. `prompt` is that
instruction. The default is upstream's `"Represent the user's input."`; set
`prompt=` at construction to change it for images and texts, or pass it to
`embed_text` per call. For retrieval, give queries a task instruction and keep
the default for documents.

Images are resized on a 32-pixel grid to between 4,096 and 1,843,200 pixels,
matching upstream's reference embedder. Against that embedder, in float32 on
CPU, text vectors differ by at most 7.4e-7 and image vectors have a cosine of
at least 0.99998.

## Variants

One size, `2b`. The 8B size is not wrapped. Not supported: combined
image-plus-text inputs, video clips (a video is embedded frame by frame), and
Matryoshka truncation as an option; slice the first `k` columns and
re-normalize instead.

## Export

Inference only: `chat()`, `set_classes()`, `train()`, `val()` and `export()`
raise.

## Licensing

<provenance-box>

Qwen3-VL-Embedding-2B is Apache-2.0, by the
[Qwen team at Alibaba](https://github.com/QwenLM/Qwen3-VL-Embedding).
Commercial use is allowed. The mirror is not gated; because the upstream
Hugging Face repository ships no LICENSE file, it carries the Apache-2.0 text
from the GitHub repository.

</provenance-box>

## Citation

<citation-block />
