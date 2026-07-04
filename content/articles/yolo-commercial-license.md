---
title: Is YOLO Free for Commercial Use? YOLOv8, YOLO11 and YOLO26 Licenses
description: "YOLOv5, YOLOv8, YOLO11 and YOLO26 ship under AGPL-3.0, so they are not free for closed-source commercial use. Here is what the license actually requires, version by version, and the MIT-licensed alternative."
date: 2026-07-04
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, commercial-use, yolov8, yolo11, yolo26, mit-license]
faq:
  - q: "Is YOLOv8 free for commercial use?"
    a: "Not for closed-source products. YOLOv8 is AGPL-3.0, so you must either release your entire application's source under AGPL-3.0 or buy the vendor's paid Enterprise License. For unrestricted free commercial use, use an MIT-licensed library like LibreYOLO."
  - q: "What license is YOLO under?"
    a: "The mainstream releases, YOLOv5, YOLOv8, YOLO11 and YOLO26, are AGPL-3.0 by default. AGPL-3.0 is a strong copyleft license: it is open source, but it is not free to use inside a proprietary product."
  - q: "Can I use YOLO in a closed-source commercial product?"
    a: "Not under AGPL-3.0 without complying, which means open-sourcing your whole product under AGPL-3.0. To keep your code private you either buy the vendor's Enterprise License or switch to a permissively licensed framework such as LibreYOLO (MIT)."
  - q: "Is there a YOLO without AGPL?"
    a: "Yes. LibreYOLO is MIT-licensed, so there is no AGPL copyleft and no network-use clause. You can use it freely in commercial and closed-source software with no fee."
  - q: "How much does a YOLO commercial license cost?"
    a: "The vendor's Enterprise License is a paid, per-company agreement with non-public pricing. LibreYOLO costs nothing: the MIT license grants commercial use for free."
  - q: "Does running YOLO only on my own servers avoid AGPL?"
    a: "No. AGPL-3.0's Section 13 covers network interaction. If users reach your modified version over a network, you must offer them the corresponding source. Running it only on your own servers is not an exemption."
---

**No. The popular YOLO models, YOLOv5, YOLOv8, YOLO11 and the new YOLO26, are not free for closed-source commercial use.** They ship under **AGPL-3.0**, a strong copyleft license. If you build a product on one of them, you must open-source your *entire* application under AGPL-3.0, or buy the vendor's paid Enterprise License. If neither works for you, there is a permissively licensed alternative: **[LibreYOLO](/) is MIT-licensed and free for any commercial use, no strings attached.**

If you searched "is YOLOv8 free for commercial use," "YOLO license," or "YOLO without AGPL," this page answers it precisely, version by version.

## YOLO licenses at a glance

| Model | Default license | Free for **closed-source** commercial use? | What you have to do |
| --- | --- | --- | --- |
| YOLOv5 | AGPL-3.0 | No | Open-source your whole app, or buy an Enterprise License |
| YOLOv8 | AGPL-3.0 | No | Open-source your whole app, or buy an Enterprise License |
| YOLO11 | AGPL-3.0 | No | Open-source your whole app, or buy an Enterprise License |
| YOLO26 | AGPL-3.0 | No | Open-source your whole app, or buy an Enterprise License |
| **LibreYOLO** | **MIT** | **Yes** | **Nothing. Ship it in proprietary products, free.** |

The one-line version: the mainstream YOLO models are open source, but under **AGPL-3.0**, a copyleft license most companies cannot comply with in a proprietary product. LibreYOLO gives you the same YOLO workflow under the **MIT license**, which places no such obligation on you.

## What license is YOLO under?

The widely used YOLO models, **YOLOv5, YOLOv8, YOLO11, and the newest YOLO26**, are distributed by their vendor under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

AGPL-3.0 is *strong copyleft*. It is not the permissive "do what you want" kind of open source like MIT or Apache-2.0. It carries one specific, far-reaching obligation, and that obligation is exactly what trips up commercial users.

## Is YOLOv8 free for commercial use?

Not in the way most people mean. You *can* download YOLOv8 and use it commercially, but only if you comply with AGPL-3.0. In practice that means:

- If you distribute, ship, or host a product that includes YOLOv8, you must **release the complete source code of that product**, your app, your integration code, your scripts and config, under AGPL-3.0.
- That obligation is triggered not only by shipping software to users, but also by letting users interact with it **over a network**, AGPL's defining Section 13 clause. A SaaS or API backend does not escape it.

For a hobby project or academic research, that is usually fine. For a proprietary commercial product, publishing your entire codebase is almost always a non-starter. Everything here applies identically to **YOLO11** and **YOLO26**: same license, same obligation.

## What about YOLOv5, YOLO11 and YOLO26?

Same story. All of them default to **AGPL-3.0**. The vendor's Enterprise License explicitly spans "YOLO26, earlier YOLO versions, and any future YOLO models," which tells you the licensing is deliberate and consistent across the whole family. Picking a newer version does not get you a friendlier license.

For completeness: not *every* model with "YOLO" in the name is AGPL. Some community variants are GPL-3.0 or Apache-2.0, and licenses differ by author. But the releases people mean when they type "YOLOv8" or "YOLO11" are AGPL-3.0.

## What AGPL-3.0 actually requires, in plain English

AGPL-3.0 extends the GPL with one crucial addition: the **network-use clause**. The practical checklist:

1. **Distribute the source.** If you convey software that includes AGPL code, you must make the *complete corresponding source* of the whole combined work available, under AGPL-3.0.
2. **Network use counts.** If users interact with your modified version over a network, a web app, an API, a SaaS, you must offer them that same source. There is no "we only run it on our servers" loophole.
3. **Copyleft is viral.** The obligation reaches the *entire* derivative work, not just the YOLO files. Your business logic gets pulled into AGPL scope.

This is general information, not legal advice, so talk to a lawyer for your exact case. But the takeaway is simple: **AGPL-3.0 and closed-source commercial software do not mix.**

## The Enterprise license option, and its catch

The vendor offers a paid **Enterprise License** that removes the AGPL obligations, letting you embed YOLO in a proprietary product without open-sourcing anything. It is a legitimate path, and for some companies it is the right one.

The catches:

- **It costs money**, an ongoing commercial fee, negotiated per company, with non-public pricing.
- **It is a dependency on one vendor's terms**, which can change, and which cover only that vendor's code.
- **You are paying for permission** to use a model architecture that, at its core, is published research.

If a recurring fee and vendor lock-in are acceptable, the Enterprise License works. If you would rather not pay, or not carry the obligation at all, keep reading.

## The MIT alternative: LibreYOLO

Disclosure: LibreYOLO is ours. It is also the reason this page can end with a clean answer instead of a shrug.

**LibreYOLO is a YOLO framework released under the permissive MIT license.** That single fact changes everything about commercial use:

- **Free for commercial use**, including proprietary, closed-source products.
- **No copyleft, no network clause.** You never have to open-source your app.
- **No per-seat or enterprise fee.** MIT means MIT.
- **The same workflow you already know**, so migrating is straightforward.

LibreYOLO is a real, maintained framework, not a wrapper: object detection, segmentation, pose, classification, depth and more, under one familiar API, with training and ONNX/TensorRT/OpenVINO/NCNN export built in. The difference is not the workflow. The difference is the license.

If you are here from the broader "leaving the AGPL YOLO" question, we wrote a fuller comparison of the ecosystem in [Best Ultralytics Alternatives in 2026](/articles/best-ultralytics-alternatives).

### Why MIT matters for a business

The MIT license lets you use, modify, embed, and sell software built on LibreYOLO with **no obligation to disclose your source** and **no fee**. It is the license behind much of the modern software stack precisely because it is safe for commercial adoption. You own your product; you owe nothing.

## Try it

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

You keep the YOLO development experience: train on your data, predict, and export to ship. You drop the AGPL obligation and the enterprise invoice.

LibreYOLO is MIT-licensed, runs on Linux, Mac, and Windows, and works on GPU, Apple Silicon, and plain CPU with no code change. One API spans YOLO9, RF-DETR, RTMDet, YOLOX, D-FINE, the RT-DETR line, segmentation, pose, depth, and more, all permissively licensed.

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Docs: [libreyolo.com/docs](https://libreyolo.com/docs)
