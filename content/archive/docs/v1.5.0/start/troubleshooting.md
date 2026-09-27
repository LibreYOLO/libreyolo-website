---
title: Troubleshooting
seo_title: "Fix common LibreYOLO errors"
description: "The errors LibreYOLO raises most often, what each one means, and the fix. Includes two failures that produce wrong output instead of raising."
lead: "Errors grouped by the message you see. Two entries at the end cover the opposite problem: code that runs, returns something plausible, and is wrong."
keywords: [libreyolo error, modulenotfounderror libreyolo, libreyolo cuda out of memory, libreyolo notimplementederror, libreyolo troubleshooting]
last_verified: "1.5.0"
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

## Training hangs or restarts on Windows

Windows has no `fork`, so dataloader workers start by re-importing your script.
Without a `if __name__ == "__main__":` guard, each worker re-runs your training
call, which either deadlocks or spawns processes without end.

```python
def main():
    ...  # build the model and call train()

if __name__ == "__main__":
    main()
```

Setting `workers=0` also avoids it, at a throughput cost. The guard is the
better fix.

## Two failures that do not raise

The rest of this page is about errors. These two are worse, because the code
runs and hands back something that looks right.

### Indexing a single result

`predict()` returns one `Results` for one image, and a list for several.
Indexing the single-image return selects a *detection*, not an image:

```python
result = model.predict("image.jpg")   # a Results
result.boxes                          # every detection, correct
result[0].boxes                       # ONE detection, silently
```

Nothing raises, because indexing a `Results` is a valid operation that returns
a subset. Code written against the list form quietly reports one box per image.
Index only what you know is a list.

### Reading metrics as attributes

`val()` returns a plain dictionary keyed by metric name, not an object with
attribute access:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # correct
metrics.box.map               # AttributeError
```

The keys are namespaced with `metrics/` and `speed/`. Print the dictionary once
to see what your task produced, since the set differs by task.

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
