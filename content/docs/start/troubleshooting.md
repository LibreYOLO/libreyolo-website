---
title: Troubleshooting
seo_title: Fix common LibreYOLO errors
description: >-
  The errors LibreYOLO raises most often, what each one means, and the fix.
  Includes two former silent failures, indexing and metrics, that 1.8.0 changed.
lead: >-
  Errors grouped by the message you see. Two entries at the end cover code that
  used to run and return the wrong thing, and what 1.8.0 changed.
keywords:
  - libreyolo error
  - modulenotfounderror libreyolo
  - libreyolo cuda out of memory
  - libreyolo notimplementederror
  - libreyolo troubleshooting
last_verified: 1.6.0
---

Errors are grouped by the text you see. If your message is not here, the
[FAQ](/docs/faq) answers the questions that are not failures, and
`libreyolo models` reports what your install can actually load.

## ModuleNotFoundError naming a package you never imported

Some families need an optional extra. The message names the missing package
rather than the extra, so the fix is not always obvious from the traceback.

Run `libreyolo models`. Any family whose dependency is missing is printed with
the exact pip command that enables it, so you do not have to map package back
to extra yourself. `libreyolo models --json` prints the same as an object.

RF-DETR and DINOv2 need `libreyolo[rfdetr]`, and their message names the extra.
Without it, `LibreYOLO("LibreRFDETRn.pt")`, the DINOv2 weight names and
`libreyolo predict model=rfdetr-n` stop with the install line, and the CLI
exits with `model_load_failed`:

```
ModuleNotFoundError: RF-DETR support requires extra dependencies.
Install with: pip install libreyolo[rfdetr]
```

The [install page](/docs/install) lists every extra and what it covers.

## ONNX inference requires onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

The base package does not depend on a runtime, because which one you want
depends on your hardware. Install `onnxruntime` for CPU or `onnxruntime-gpu`
for CUDA. Both provide the same `onnxruntime` module, so install one, not both.

## ONNX model not found

```
FileNotFoundError: ONNX model not found: <path>
```

The path is resolved relative to the working directory, not the script. This
also appears when an export silently wrote somewhere else: `export()` returns
the path it wrote, so capture the return value rather than assuming a name.

## NotImplementedError from train()

Not every family trains. Some are ported for prediction, validation and export
only, and their `train()` raises rather than pretending to run.

The [FAQ entry](/docs/faq) explains the reasoning. To check a specific family
before writing a training script, its model page states whether it trains.

## NotImplementedError from export()

A family can support a task and still not export it. EoMT is the case people
hit: `export()` accepts the semantic task and raises for `segment` and
`panoptic`, because the query-mask runtime contract those need is not defined.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

Every family's page carries an export matrix showing which task and format
combinations are validated.

## CUDA out of memory

Reduce `batch` first, then `imgsz`. Both change memory roughly with their size,
but batch is the one you can drop without changing what the model sees.

If it fails at validation rather than training, validation runs its own batch
size, so lower that too.

On Windows, a display GPU has a second failure mode that looks like a random
CUDA error rather than an out-of-memory: the driver resets a GPU that stops
responding for longer than the timeout, killing whatever was running. Long
kernels on the card driving your monitor can trip it.

## Weights will not download

Weights fetch from Hugging Face on first use and cache locally. The
[FAQ](/docs/faq) covers where the cache lives and how to run fully offline.

If a download 404s, check the filename you passed. The URL is derived from it,
including the task suffix, so a name that does not match a published checkpoint
produces a URL that does not exist. The checkpoint table on each model page
lists the exact published filenames.

## train() or val() was called at the top level of your script

```
train() is called at the top level of your script, and DataLoader workers re-run that script on this platform, so it runs with workers=0. Put the call under `if __name__ == "__main__":` to load data in parallel.
```

Windows has no `fork`, and Python on macOS does not use it by default, so
dataloader workers start by re-importing your script. The `spawn` and
`forkserver` start methods do the same on any platform. Without an
`if __name__ == "__main__":` guard, each worker would run your `train()` or
`val()` call again. Since 1.7.1, LibreYOLO detects this, runs the call with
`workers=0` and logs this line once. The script finishes, but data loading runs
in the main process, which is slower.

Put the call under the guard to get parallel data loading back:

```python
def main():
    ...  # build the model and call train()

if __name__ == "__main__":
    main()
```

Guarded scripts, notebooks, the CLI and Linux `fork` keep their workers.

## Two silent failures that 1.8.0 changed

The rest of this page is about errors. These two used to be worse: the code ran
and handed back something that looked right. Both behave the Ultralytics way
since 1.8.0, so check them when a script gives different numbers after an
upgrade.

### Indexing a single result

`predict()` returns one `Results` for one image, and a list for several. Since
1.8.0, an integer `0` or `-1` on a single-image `Results` returns the whole
result, so code written for the list form works on both:

```python
result = model.predict("image.jpg")   # a Results
result.boxes                          # every detection
result[0].boxes                       # every detection, the same Results
result[2:3].boxes                     # the third detection only
result.boxes[2]                       # the same detection
```

Other integers (`result[2]`) still select one detection, with a
`DeprecationWarning` naming `result[2:3]` or `result.boxes[2]`. Slices, boolean
masks and index arrays select detections, as before. Before 1.8.0,
`result[0]` silently returned the first detection only, so code that ran on
1.7 reports more boxes on 1.8.

### Reading metrics as attributes

`val()` returns a dictionary keyed by metric name, and since 1.8.0 its `box`
attribute also names the main detection metrics:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # the dictionary key
metrics.box.map               # the same value
metrics.box.map50, metrics.box.map75
metrics.box.mp, metrics.box.mr  # mean precision and recall
metrics.box.maps              # AP50-95 per class index
```

Segmentation adds `metrics.seg` with `map`, `map50`, `map75` and `maps` for the
masks, and detection results carry `metrics.confusion_matrix`. The dictionary
keys are namespaced with `metrics/` and `speed/`. Print the dictionary once to
see what your task produced, since the set differs by task.

## Checking a dataset before you train

Most training failures are dataset problems. `libreyolo doctor data.yaml` runs
health checks over a detection dataset and reports findings by severity, which
is faster than reading a traceback from epoch one.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

See the [doctor command](/docs/cli/doctor) for the check catalog.

## Persistent workers cannot observe dataset changes

Active `close_mosaic` or `set_epoch` hooks must reach the dataset copies used by workers. If a custom multi-worker loader keeps incompatible persistent copies, use `persistent_workers=False` or rebuild workers after the mutation. The default nonpersistent path is unaffected.
