---
title: "LiteRT vs TensorFlow Lite: It Is a Rename, Not a New Format"
description: "Google renamed TensorFlow Lite to LiteRT in September 2024. Same runtime, same .tflite files, nothing breaks. Here is why they did it, what actually changed, and how to export a LibreYOLO model to it."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "Is TensorFlow Lite deprecated?"
    a: "The name is being retired, not the technology. The runtime lives on as LiteRT with the same APIs, and existing TFLite models and code keep working. New development, like the CompiledModel API and NPU acceleration, ships under the LiteRT name."
  - q: "Do I need to re-export my models for LiteRT?"
    a: "No. A .tflite file exported for TensorFlow Lite is a LiteRT model. Nothing about the file changed, so there is nothing to re-export or migrate."
---

**LiteRT is TensorFlow Lite with a new name. It is not a new runtime, and it is not a new file format. Models are still `.tflite` files, and every `.tflite` file that ran on TensorFlow Lite runs on LiteRT unchanged.**

If you searched "LiteRT vs TensorFlow Lite", "is TensorFlow Lite deprecated" or "what is LiteRT", that paragraph is the answer. The rest of this page is the short version of why the rename happened and how to export a model to it with LibreYOLO.

## Why Google renamed it

TensorFlow Lite launched in 2017 as the way to run TensorFlow models on phones and embedded boards. Over the years it grew far beyond that: Google built conversion paths from PyTorch, JAX and Keras, so by 2024 a large share of the models running on it had never touched TensorFlow at all.

At that point the name was actively misleading. PyTorch users skipped it because it sounded like a TensorFlow-only tool. So in September 2024 Google [renamed it LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), short for "Lite Runtime". That is the whole story. Same team, same code, framework-neutral name.

## What actually changed

Very little, and none of it breaks anything:

* The code moved to a new home: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* The docs moved to [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* The Python package for running models is now `ai-edge-litert`. The old `tflite-runtime` package is stale; the new one has the same `Interpreter` class.
* Since the rename, new features ship under the LiteRT name: a simpler `CompiledModel` API and GPU/NPU acceleration that Google declared [production-ready in January 2026](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

The classic Interpreter API also keeps working as-is. There is one genuinely new extension, `.litertlm`, but it is a packaging format for on-device LLMs only; for computer vision models it does not exist as far as you are concerned. So when one doc says "TFLite" and another says "LiteRT", they are talking about the same file.

## Exporting a LibreYOLO model to LiteRT

LibreYOLO added a TFLite/LiteRT export path in v1.3.0. It needs Python 3.12+ and one extra:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

Or from the CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

The validated paths today are YOLO9 detection and RF-DETR detection, segmentation and pose, all still marked experimental. The full support matrix is in the [docs](/docs/v1.3.0).

To run the exported file, use Google's `ai-edge-litert` package on the target device:

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, your preprocessed image here
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

Note the input is NHWC (channels last): the conversion from ONNX transposes the layout, which is normal for this format.

## What we changed in LibreYOLO

Not much, because not much needed to change. The docs now label the format "TFLite (LiteRT)" so both names are searchable, and the API keeps `format="tflite"` as the format name.

## FAQ

**Is TensorFlow Lite deprecated?**
The name is being retired, not the technology. The runtime lives on as LiteRT with the same APIs, and existing TFLite models and code keep working. New development, like the CompiledModel API and NPU acceleration, ships under the LiteRT name.

**Do I need to re-export my models for LiteRT?**
No. A `.tflite` file exported for TensorFlow Lite is a LiteRT model. Nothing about the file changed, so there is nothing to re-export or migrate.
