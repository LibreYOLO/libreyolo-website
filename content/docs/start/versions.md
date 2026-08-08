---
title: Versions
seo_title: "LibreYOLO versions and older documentation"
description: "Which release these pages describe, where the documentation for 1.1.0 through 1.4.0 lives, and how to check the version you have installed."
lead: "These pages describe one release of LibreYOLO on unversioned URLs. Documentation for earlier releases stays online at its own versioned path."
keywords: [libreyolo versions, libreyolo docs archive, libreyolo 1.4.0 docs, check libreyolo version, pin libreyolo version]
last_verified: "1.5.0"
snippets:
  version:
    - label: CLI
      language: bash
      code: |
        libreyolo version
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
---

## The release these pages describe

These pages describe LibreYOLO 1.5.0. The version sits at the top of the docs
sidebar, and every page ends with a line naming the release it was verified
against.

These URLs carry no version number and do not move. `/docs/models/rf-detr` is
the current release's page for that family, today and after the next release.
Pages are revised in place when behavior changes, so the verified-against line
at the foot of a page is what tells you how current the text is.

## Check what you have installed

`libreyolo version` prints the version together with the Python, torch and CUDA
versions it is running against, which is usually what you want when something
behaves differently from the docs. `libreyolo --version` prints the bare number
and exits.

<code-tabs name="version" />

## Documentation for earlier releases

Each earlier release keeps the single-page documentation it shipped with, at
its own path.

| Release | Documentation |
| --- | --- |
| 1.4.0 | [/docs/v1.4.0](/docs/v1.4.0) |
| 1.3.1 | [/docs/v1.3.1](/docs/v1.3.1) |
| 1.3.0 | [/docs/v1.3.0](/docs/v1.3.0) |
| 1.2.0 | [/docs/v1.2.0](/docs/v1.2.0) |
| 1.1.0 | [/docs/v1.1.0](/docs/v1.1.0) |

Those pages are a record of what the library did at that release. They are not
revised, so an argument that has since been renamed or a default that has since
moved is still described there the old way. If you are pinned to one of those
versions, its page is the accurate reference and this tree is not.

## Pinning a release

`pip install libreyolo` installs the latest release published to PyPI. Pin the
version when a result has to be reproducible:

```bash
pip install "libreyolo==1.4.0"
```

Installing from source, a plain clone checks out `release`, the stable branch
whose code matches the published release. The `dev` branch carries work that
has not been released yet, including anything listed under
[changelog](/docs/changelog) as landing since the last version.

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
pip install -e .
```

## What changes between releases

Checkpoints move forward, not backward. A checkpoint written by a newer release
can fail to load in an older one: the 1.4.0 notes record that checkpoints using
the task strings introduced in that release, or carrying finalized quantization
state, are not loadable by 1.3.1. Keep the version that wrote a checkpoint
alongside the checkpoint itself.

Defaults move too, and the changelog records each one with its reason. In 1.4.0
PicoDet's `lr0` went from 0.1 to 0.01, because the old default destroyed
COCO-pretrained weights, and DEIM's went from 4e-4 to 1e-4. A training script
that leaned on a default therefore reproduces differently across versions. Pass
the values you care about explicitly and a run stays comparable.

[Changelog](/docs/changelog) summarizes what landed in recent releases, and
[`CHANGELOG.md`](https://github.com/LibreYOLO/libreyolo/blob/dev/CHANGELOG.md)
in the repository carries the full entries.
