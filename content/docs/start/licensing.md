---
title: Licensing
seo_title: "LibreYOLO licensing: code and weights"
description: "LibreYOLO's own code is MIT. Vendored upstream code and published checkpoints carry their own licenses, and several of those are non-commercial."
lead: "LibreYOLO carries three separately licensed things: its own code, upstream code vendored into a model family, and pretrained checkpoints. They are often not the same license."
keywords: [libreyolo license, mit computer vision library, non-commercial model weights, model checkpoint license, apache-2.0 object detection]
last_verified: "1.5.0"
---

## LibreYOLO's own code

The library is MIT. That covers the Python API, the CLI, the trainers,
validators and exporters, the dataset loaders, and the conversion scripts under
`weights/`. Use it in a commercial or closed-source product, keep the copyright
line and the license text with any copy you redistribute, and the obligation
ends there.

The grant stops at the code. The
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE) file
puts it plainly:

> Those licenses vary and are not all permissive: some published weights are
> non-commercial or otherwise restricted, and this MIT License does not extend
> to them. Choosing a model means choosing its license.

## Upstream code, per family

Most families are ports of published research, and several vendor upstream
source directly. A vendored file keeps its original copyright header and its
original license. MIT does not overwrite it, and LibreYOLO does not relicense
anyone's work. Apache-2.0 and BSD-3-Clause are the two that come up most often.

Apache-2.0 covers the DETR line and much of the transformer work: DETR from
Meta AI (FAIR), Deformable DETR from SenseTime, LW-DETR from Baidu, OV-DEIM by
Leilei Wang and coauthors, the SegFormer implementation LibreYOLO ports from
Hugging Face Transformers, PP-OCRv5 from the PaddlePaddle Authors, SwinIR from
the Computer Vision Lab at ETH Zurich, and Depth Anything 3 from ByteDance
Seed. It also covers the classifiers derived from timm by Ross Wightman and the
timm contributors, among them ResNet, DeiT, EfficientNetV2, MobileNetV4 and
Swin, whose module names mirror timm so that its ImageNet tensors load
unchanged.

BSD-3-Clause covers everything derived from torchvision: Faster R-CNN,
Mask R-CNN, FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN and DeepLabv3.

MIT covers a smaller group, including NAFNet from Megvii, CenterNet from Xingyi
Zhou, and YOLOv7 as re-released by its own authors, Kin-Yiu Wong and Hao-Tang
Tsui, at MultimediaTechLab. The YOLOv1 through YOLOv4 families reproduce
architectures from the Darknet project, by Joseph Redmon and, for YOLOv4, by
Alexey Bochkovskiy. Darknet is public domain, so those carry no obligation at
all.

One bundled subtree is not an open-source license. The DEIMv2 family ships
DINOv3 backbone code from Meta Platforms under the DINOv3 License Agreement, a
custom non-OSI license. Redistributing that code means shipping a copy of the
agreement with it, and the agreement forbids use for activities subject to
ITAR, military or warfare purposes, nuclear industries, espionage, and weapons
development. Those terms bind that subtree only.

Two files in the repository hold the full picture.
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE) lists
every bundled third-party subtree with its path, its license file and its
upstream source.
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
lists the upstream projects LibreYOLO derives from and reproduces each license
text in full.

## Weights, per checkpoint

No pretrained weight file ships inside the package. Published checkpoints live
on Hugging Face under the [LibreYOLO
organization](https://huggingface.co/LibreYOLO), and each repository carries its
own `LICENSE` and attribution reflecting the project the weights came from.

That repository is the authoritative source for the terms. Not this page, not
the model page, and not the summary in the source tree. See
[checkpoints and weights](/docs/weights) for how files are named and where they
are downloaded from.

Licenses differ between families, and they differ between files inside one
family. Two examples of the second:

- The YOLO9 COCO checkpoints are MIT. `LibreYOLO9P2s-visdrone.pt`, trained on
  VisDrone2019-DET, is CC BY-NC-SA 3.0, which is non-commercial.
- The RF-DETR COCO checkpoints are Apache-2.0. The oriented-box checkpoints
  trained on DOTA are CC BY 4.0.

Across families, the range runs wider, and several published checkpoints cannot
be used in a commercial product:

- SegFormer is the clearest split between the two layers. The implementation is
  an Apache-2.0 port of Hugging Face Transformers' code. The published ADE20K
  checkpoints are converted from NVIDIA's release under the NVIDIA Source Code
  License, which permits redistribution but limits use to non-commercial
  research or evaluation, and carries that limit forward into derivative works.
  Those checkpoints are not covered by LibreYOLO's permissive terms.
- OV-DEIM checkpoints are CC BY-NC 4.0, confirmed by the upstream author. Every
  prediction also loads Apple's MobileCLIP-B(LT) text tower, whose license
  restricts use to research, a stricter term than the checkpoint's own.
- SenseNova-Vision code is Apache-2.0 and its weights are CC BY-NC 4.0. The
  loader prints the non-commercial notice before every automatic download.

Some families have no checkpoint hosted by LibreYOLO at all, and their pages
say so in the Weights row. SAM 3 is gated on Hugging Face under Meta's custom
SAM License and is downloaded from Meta directly. MiDaS release assets are
fetched from the official URLs and hash-verified rather than rehosted. Dome-DETR
is linked upstream because its model card states no license in its metadata
while its prose claims Apache-2.0 and restricts use to academic research at the
same time, and those do not agree. The TEED and DexiNed architectures are MIT,
but the authors' released checkpoints were trained on BIPED, whose dataset terms
are non-commercial, so LibreYOLO neither bundles nor auto-downloads them.

Several torchvision checkpoints carry no license file of their own. LibreYOLO
mirrors them on the license the releasing project uses, states on each model
card that the basis is implied rather than granted per checkpoint, and repeats
torchvision's own warning that pretrained model terms may derive from the
training data.

## Finding the terms for one model

The model page carries a **Licenses** row in its header, in the form
`Code X, weights Y`, which links down to the page's Licensing section. That
section lists the original work and its authors, the upstream license, the
upstream source, the LibreYOLO code license, the weights, and an interpretation
of what the terms allow. The Checkpoints table on the same page has a
**Weights license** column, one row per published file, so a family with mixed
terms shows them file by file.

All of that renders from the same data the library is checked against, which is
why this page does not repeat it as a table. A hand-typed license matrix is
wrong within one release, and wrong here is expensive.

In the source tree, the equivalents are `NOTICE` for bundled code,
`THIRD_PARTY_NOTICES.txt` for upstream projects and their license texts, and
[`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)
for a per-family summary of the published checkpoints.

Then check the Hugging Face repository of the exact file you are about to
download. It is authoritative, and it can change without a docs page changing
with it.

## Commercial use

Code is rarely the problem. MIT, Apache-2.0 and BSD-3-Clause all permit
commercial and closed-source use. Each asks you to keep its license text and
attribution notices with copies you redistribute, Apache-2.0 also grants a
patent license, and none of them places conditions on your own application
code.

Checkpoints are where products get stuck. A non-commercial checkpoint stays
non-commercial however permissive the surrounding code is, and converting the
file does not change its applicable terms, which is what
`weights/LICENSE_NOTICE.txt` states directly. An ONNX or TensorRT artifact
built from a restricted checkpoint inherits the restriction.

Where a license carries its restriction into derivative works, as the NVIDIA
Source Code License does, fine-tuning does not escape it either. Training the
same architecture from scratch on data you have the right to use does: the code
is permissive, so a model you train yourself is yours, and the pretrained
checkpoint's terms never enter it. The SegFormer page spells that out for its
own weights; read the Interpretation row on the page of whichever family you
plan to ship.

Decide the license question when you pick the model rather than when you ship,
and read the terms on the file you actually downloaded, because a family with
one permissive checkpoint can have a restricted one beside it.

## Not legal advice

This page describes the licenses involved. It is a description, not legal
advice, and it does not create any warranty. If the answer matters
commercially, read the licenses yourself and take your own counsel.
