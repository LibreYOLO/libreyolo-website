---
title: Checkpoints and weights
seo_title: "LibreYOLO checkpoints and weights"
description: "How LibreYOLO finds, downloads and verifies model weights, where they are hosted, how to run with no network, and what makes a checkpoint load safely."
lead: "A LibreYOLO checkpoint is a torch.save dictionary holding a state dict plus the metadata needed to identify it. This page covers where those files come from, where they land, and how they are loaded."
keywords: [libreyolo weights, libreyolo checkpoints, libreyolo download weights, libreyolo offline, libreyolo hugging face, checkpoint metadata]
last_verified: "1.5.0"
meta:
  - label: Hosted at
    value: "One Hugging Face repository per checkpoint:"
    links:
      - label: huggingface.co/LibreYOLO
        href: https://huggingface.co/LibreYOLO
  - label: Local cache
    value: weights/ under the working directory
    mono: true
  - label: Metadata schema
    value: v1.0
snippets:
  load:
    - label: Auto-download
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # A bare filename resolves to weights/LibreYOLO9t.pt and is
        # downloaded there if it is not already present.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Explicit path
      language: python
      code: |
        from libreyolo import LibreYOLO

        # A path with a directory component is used exactly as written and
        # is never fetched from the network.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Reads the metadata without constructing a model, and reports
        # whether it satisfies the schema.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: |
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )

        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")

        # Returns a list of problems. Empty means the file satisfies v1.0.
        print(validate_checkpoint_metadata(loaded))
        print(loaded["model_family"], loaded["size"], loaded["task"], loaded["nc"])
---

## Where a checkpoint is looked for

A model reference with no directory component, such as `LibreYOLO9t.pt`, is
resolved against `weights/` relative to the current working directory. If
`weights/LibreYOLO9t.pt` exists it is used; if a file of that name exists in the
working directory itself it is used instead; otherwise `weights/LibreYOLO9t.pt`
becomes the download target.

A reference that does contain a directory, absolute or relative, is taken
literally. That is the form to use when weights live somewhere central and
nothing should be fetched.

<code-tabs name="load" />

## Auto-download

When the resolved path does not exist, LibreYOLO parses the filename to recover
the family, the size and the task, and asks the matching family for a download
URL. Most families build it from the LibreYOLO organization on Hugging Face,
where each checkpoint has its own repository named after the file:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

A dataset-variant suffix stays part of the repository name, so a checkpoint
trained on something other than the family default resolves to its own
repository rather than overwriting the default one.

The transfer itself is defensive, because a truncated weight file fails later
with an unhelpful error. Downloads are streamed to a `.part` file and moved into
place atomically only when complete, so an interrupted process can never leave a
half-written checkpoint at the final path. An interrupted transfer resumes from
its byte offset using an HTTP validator, and restarts from zero if the server
indicates the object changed. Failures are retried three times with exponential
backoff. Concurrent processes targeting the same path take a lock file, so two
training runs starting together download once. Where a family fetches from a
third-party host rather than the LibreYOLO organization, it can pin a checksum
and refuse the file on mismatch.

If `HF_TOKEN` is set, or a token is cached at `~/.cache/huggingface/token`, it
is attached as a bearer token. It is attached only to `huggingface.co` URLs, so
a family that downloads from another host never receives it.

Not every family auto-downloads. Some deliberately return no URL because the
released weights may not be redistributed, and the error then explains what to
supply instead. Others print a license notice before the transfer starts, which
is the case for any weights whose terms are narrower than LibreYOLO's own
license. Read that notice: it is the only place some restrictions appear at
runtime.

## The Hugging Face organization

Published weights live at
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), one repository per
checkpoint. Each repository carries a license, and the license is not uniform
across a family: a family whose code is MIT can have some weights that are not.
The repository is authoritative. Every model page lists that family's published
checkpoints and their licenses under its Checkpoints and Licensing sections.

## Working offline

Nothing about the library requires network access once the files are local. Two
approaches work:

Pre-populate a `weights/` directory next to wherever the job runs. Fetching the
checkpoints once on a connected machine, then copying the directory, is enough;
the resolution step above finds them and never reaches the network.

Or pass an absolute path to a shared location. A reference with a directory
component is used as given, so a read-only mount of curated weights is a valid
setup. If the process cannot write next to a checkpoint it needs to convert,
conversion falls back to a private temporary directory instead of failing.

Datasets follow a separate rule: they resolve under `~/datasets`, or under the
directory named by `LIBREYOLO_DATASETS_DIR` when that variable is set.

## Loading safety

Checkpoints are pickles, and a pickle can execute arbitrary code when it is
opened. LibreYOLO treats every weight file as untrusted and loads it with
PyTorch's `weights_only=True` path, which restricts the unpickler to tensors and
a small set of safe types. This applies to the file you pass, not only to files
LibreYOLO downloaded. On a PyTorch build too old to support that argument, the
load is refused rather than performed unsafely.

Some upstream training checkpoints embed objects the restricted unpickler
rejects, such as a configuration object from the framework they were trained
with. Those objects are metadata that LibreYOLO does not need, so during
conversion each blocked class is replaced by an inert stand-in that satisfies
the unpickler without running anything, and only tensors survive into the
converted file. Sensitive module names are refused outright rather than stubbed,
and the retry loop is bounded so a file engineered to introduce an endless
series of blocked classes fails closed. See
[import existing weights](/docs/migrate) for the rest of that path.

## Checkpoint metadata

A LibreYOLO checkpoint is a dictionary whose `model` key holds the PyTorch state
dict. Nine keys are required by schema v1.0, and together they let the factory
identify a file without parsing its name or guessing from tensor shapes.

| Key | Meaning |
|---|---|
| `model` | The PyTorch state dict |
| `schema_version` | The metadata contract version. v1.0 uses the string `1.0` |
| `libreyolo_version` | The LibreYOLO version that produced the file |
| `model_family` | A registered family identifier, such as `yolo9` |
| `size` | The variant within that family, such as `t` or `r18` |
| `task` | One canonical task name |
| `nc` | A positive class count |
| `names` | A mapping of class index to label, covering `0` to `nc - 1` |
| `imgsz` | A positive input resolution |

Tasks with extra structure record it alongside those keys. Pose checkpoints add
`num_keypoints` and `keypoint_dim`, and may add per-keypoint OKS sigmas. OCR
checkpoints embed the full CTC charset so the file is self-contained. Restore
checkpoints may record the degradation type and an upscale factor. Trainer
checkpoints add resume state such as `epoch`, the optimizer state and the EMA
weights; published inference weights should not carry that.

A file that satisfies all nine keys loads through the metadata path. A file that
does not is either converted, if a family recognizes its layout, or loaded
through the compatibility path with a warning naming what is missing.

## Inspecting a checkpoint

<code-tabs name="inspect" />

`libreyolo metadata` never constructs a model, so it works on a file whose
family is not installed and on a file you are not sure about.
