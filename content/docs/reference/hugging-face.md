---
title: Hugging Face Hub
seo_title: Hugging Face Hub | LibreYOLO
description: >-
  Load LibreYOLO checkpoints from Hub repositories and publish trained
  checkpoints.
lead: >-
  Load LibreYOLO checkpoints from Hub repositories and publish trained
  checkpoints.
keywords:
  - Hugging Face Hub
  - LibreYOLO
last_verified: 1.6.0
snippets:
  load:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("hf://LibreYOLO/LibreYOLO9s/LibreYOLO9s.pt", device="cpu")
        print(model(SAMPLE_IMAGE).boxes)
---

## Install

```bash
pip install "libreyolo[hf]"
```

## Load

<code-tabs name="load" />

The accepted forms are `owner/repo`, `hf://owner/repo` and `hf://owner/repo@revision/filename`. A local path wins over a bare repository ID. Explicit `hf://` bypasses that precedence. Revisions must be slash-free branch names or commit hashes.

Without a filename, a repository must have one checkpoint. An ambiguous repository raises and lists the choices. Hub checkpoints require LibreYOLO metadata or a recognized upstream conversion. Downloads use the shared Hub cache.

## Publish

`model.push_to_hub(repo_id, private=False, license=..., metrics=...)` creates the repository if needed and uploads `model.pt` with a generated card. Authenticate with `hf auth login` or `HF_TOKEN`; publishing requires write access.

## Training logger

Pass `loggers="hf:owner/repo"` or a `HuggingFaceHubLogger` instance. The logger checks write access and creates a missing target repository at construction. It uploads the best checkpoint at train end, falling back to the last checkpoint. The logger defaults to a private repository; explicit `push_to_hub()` defaults to public. Existing repositories keep their visibility.
