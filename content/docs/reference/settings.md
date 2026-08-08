---
title: Settings
seo_title: "LibreYOLO environment variables and directories"
description: "Every environment variable LibreYOLO reads, the directories it writes to, the tokens it needs, and the toggles that change which code path runs."
lead: "LibreYOLO has no configuration file. Behavior that is not a function argument is controlled by environment variables and by a small number of conventional directories, all listed here."
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - libreyolo weights directory
  - libreyolo cache
last_verified: "1.5.0"
verification: "Variables located by searching libreyolo/**/*.py for os.environ and os.getenv at v1.5.0; semantics read at each use site. Directory conventions read from libreyolo/data/utils.py, libreyolo/utils/download.py, libreyolo/export/exporter.py, libreyolo/models/base/model.py and libreyolo/models/sam3dbody/mhr_body.py."
snippets:
  usage:
    - label: Point the dataset root somewhere else
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Read the resolved value from Python
      language: python
      code: |
        from libreyolo.data import DATASETS_DIR

        # Defaults to ~/datasets; LIBREYOLO_DATASETS_DIR overrides it at import time.
        print(DATASETS_DIR)
---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | Dataset root. Read once at import, into `libreyolo.data.DATASETS_DIR` |
| `LIBREYOLO_FASTER_COCO_EVAL` | unset | Overrides the `faster_coco_eval` validation flag. `1`, `true`, `yes` or `on` forces the faster backend on, any other value forces it off, unset defers to the config flag |
| `LIBREYOLO_KERNELS` | unset | Kernel selection. `off` or `reference` forces the reference implementations; any other value selects only implementations registered under that name |
| `LIBREYOLO_QUANT_KERNELS` | unset | Legacy alias for `LIBREYOLO_KERNELS`, read only when that one is unset |
| `LIBREYOLO_HUB_KERNELS` | unset | `0`, `false`, `off` or `no` disables Hugging Face Hub kernel loading. Any other value, including unset, leaves it enabled |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | Location of the MHR body model used by the `mesh` task |
| `LIBRELABEL_ENABLE_LOCATE` | unset | Must be exactly `1`, `true`, `yes` or `on` to expose the LocateAnything assistant in the labeling tool. Any other value keeps it off |
| `SAM_3D_BODY_PATH` | unset | Path to the SAM 3D Body package for the mesh family, when it is not passed to the constructor |
| `HF_TOKEN` | unset | Hugging Face access token, used for gated repositories |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` is read at import time, so setting it after importing
`libreyolo.data` has no effect on `DATASETS_DIR`.

Hub kernels are a two-part opt-in. The runtime fetch only happens when the
optional `kernels` package is installed, so installing
`libreyolo[hub-kernels]` is the opt-in and `LIBREYOLO_HUB_KERNELS=0` is the
opt-out. An installation without the extra is unaffected either way.

Kernel selection also short-circuits imports: when `LIBREYOLO_KERNELS` forces
`off` or `reference`, the in-tree accelerated providers are never imported at
all.

## Variables the library sets

These are written rather than read, so setting them by hand is not the
supported path.

| Variable | Set by |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | The DDP spawn helper, one value per worker process |
| `CUDA_VISIBLE_DEVICES` | Temporarily narrowed during distributed setup, then restored |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Set to `1` by the EC trainers, with `setdefault`, so an existing value wins |
| `MOMENTUM_ENABLED` | Set with `setdefault` by the mesh family loader |

`LOCAL_RANK` doubles as the distributed-mode signal: its presence in the
environment is how the training code detects that it is running under DDP.

## Logger variables

The optional training loggers fall back to environment defaults for the
project name.

| Variable | Default | Used by |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | The Weights and Biases logger, when no project is passed |
| `COMET_PROJECT_NAME` | `libreyolo` | The Comet logger, when no project is passed |

Authentication for those services follows their own tooling, not LibreYOLO.

## Tokens

`HF_TOKEN` is the Hugging Face access token. When it is unset, the token is
read from `~/.cache/huggingface/token`, which is where a Hugging Face CLI
login writes it. Either path works.

A token is needed only for gated repositories. SAM 3 is the shipped example:
its weights download from a gated repository under a custom license, so the
terms have to be accepted on the repository page and the session has to be
authenticated.

## Directories

| Path | Contents |
|---|---|
| `weights/` | Downloaded checkpoints, downloaded Hugging Face snapshots, and exported artifacts |
| `~/datasets` | Dataset root, unless `LIBREYOLO_DATASETS_DIR` says otherwise |
| `~/.cache/huggingface/token` | Hugging Face token, when not in `HF_TOKEN` |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | MHR body model, unless `LIBREYOLO_MHR_PATH` says otherwise |
| `runs/track/` | Default output for `model.track(save=True)` |

`weights/` is relative to the working directory. A bare filename resolves
through it, so `LibreYOLO("LibreYOLO9t.pt")` looks for `weights/LibreYOLO9t.pt`
and downloads there when it is absent. `model.export()` writes into the same
directory when `output_path` is not given. The sibling tiers download
multi-file snapshots into `weights/<Prefix><size>/`.

## Download behavior

Weight downloads are retried three times with backoff, resume from a partial
file, and are guarded by a lock file so two processes do not fetch the same
checkpoint at once. A family that fetches from a third-party host can pin a
checksum and fail closed on a mismatch.

Some downloads print a license notice before they start. Those notices are
part of the download path and are not suppressible through configuration.

## Validation backend

`model.val()` accepts `faster_coco_eval=True` by default and falls back to
pycocotools when the package is not installed, warning once. Setting
`LIBREYOLO_FASTER_COCO_EVAL` overrides the per-call flag, which is what a
benchmark harness that cannot touch per-run configs should use. The backend
that actually ran is reported on `model.last_eval_backend`.

## Dataset download scripts

A dataset YAML may carry a `download` field containing Python. It is not
executed unless `allow_download_scripts=True` is passed to the call that reads
it, which is a function argument on `val()` and `export()` rather than an
environment variable.
