---
title: How to run Depth Anything v2 with LibreYOLO
description: Depth Anything V2 gives you state-of-the-art monocular depth. Here is how to run it in two lines with LibreYOLO.
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
---

![Guggenheim Bilbao next to its depth map](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2 produces some of the best monocular depth maps available right now.

If you come from the YOLO world, the official repo is not the easiest thing to pick up. Getting a single depth map out of it takes a handful of manual steps:

* download the right checkpoint by hand and drop it in the right folder,

* match the encoder to its config (`encoder="vitl"`, `features=256`, `out_channels=...`),

* write the normalize-and-colorize step yourself,

* handle device placement so it runs on a Mac or CPU, not just CUDA,

* and learn an inference API that looks nothing like the rest of your stack.

None of it is hard. It is just friction, and you can skip all of it.

LibreYOLO loads the same Depth Anything V2 weights and gives you the depth task as one `predict()` call. Name the model and it downloads on first use:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")  # auto-downloads on first run (Apache-2.0, commercial-safe)
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

That is the whole thing. If you have used the standard YOLO API, the shape is familiar: load a model by name, call `predict`, let `save=True` write the visualization. It is the exact same call you would use for object detection, except the result carries a depth map instead of boxes. The colormap, the normalization, the device placement are handled for you. It runs the same way on Linux with CUDA, on a Mac with Apple Silicon, or on plain CPU. No code change.

From there the same call does more: you can pull the raw depth values to work with directly. Training Depth Anything V2 itself stays in the original repo; LibreYOLO covers inference, video, and validation.

The exact same one-line call, on very different scenes:

![The parkour sample image next to its depth map: the jumpers in front stand out from the concrete walls behind.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![Aerial view of La Concha bay in Donostia next to its depth map: the boats and shoreline read as near, the open water recedes.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![The columned courtyard of the Casa de Juntas de Gernika next to its depth map, showing the receding architecture.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![A night festival crowd in Donostia's Plaza de la Constitucion next to its depth map, the near rows of people standing out from the lit facade behind.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

A note on weights: LibreYOLO hosts converted Depth Anything V2 checkpoints and pulls them on first use, so there is nothing to download by hand. The upstream license still applies: the Small encoder is Apache-2.0, while Base, Large, and Giant are CC-BY-NC-4.0 (non-commercial), so the strong checkpoints are for non-commercial use. Want to stay offline or convert your own? The one-time conversion script is still there:

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vits.pth weights/LibreDepthAnythingV2s-depth.pt
```

## Try it

```bash
pip install libreyolo
```

LibreYOLO is the most complete computer vision library you can pip install. One familiar API spans a growing roster of state-of-the-art models: object detection (RF-DETR, D-FINE, DEIM), segmentation, pose, oriented boxes, and now monocular depth with Depth Anything V2, plus training and validation for the tasks that support it. It runs on every major OS, on GPU or CPU, and it is fully MIT-licensed, so you can ship it commercially with no strings attached.

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Docs: [libreyolo.com/docs](https://libreyolo.com/docs)
