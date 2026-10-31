---
title: NAFNet
families: [nafnet]
seo_title: "NAFNet: denoise, train and export under MIT"
description: "Use NAFNet in LibreYOLO for image denoising and restoration. Install, predict, train, validate and export the SIDD checkpoint, MIT-licensed."
lead: "NAFNet is a convolutional network for image restoration that removes the nonlinear activation functions from a typical UNet block, replacing them with elementwise multiplication. LibreYOLO supports it for one task, restoration, with a published real-image denoising checkpoint trained on SIDD."
keywords: [NAFNet, image restoration, image denoising, image deblurring, nonlinear activation free network, SIDD]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg save=True
    - label: Save the restored image
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16, lr0=1e-3)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Checkpoint provenance
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation and dataset are recorded on the saved checkpoint; they
        # don't change what is trained.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() returns a plain dict, not an object
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx imgsz=256
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt imgsz=256 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model("noisy.jpg")

        result.restored.save("denoised.png")
---

## Install

NAFNet needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object carries one field for this family, `restored`, a
dense HWC uint8 RGB image on the original canvas; there are no boxes to
iterate. `save=True` writes that restored image straight to disk rather than
drawing an annotation over the input. `conf`, `iou` and `max_det` are accepted
for signature parity with every other family but have no effect, since
restoration produces no detections to filter. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Two widths share this architecture: `s` (width 32) and `l` (width 64), both
built around a 256 px training patch. Predict and validate run at native image
resolution regardless of size, padding only to the network's downsample
factor. Only the `l` width is currently published, as a real-image denoising
checkpoint trained on SIDD.

## Train

NAFNet fine-tunes on your own paired degraded/clean images: a dataset YAML
pointing at an `inputs/<split>/` folder of degraded images and a
`targets/<split>/` folder of clean targets, matched by filename stem.
`degradation` and `dataset` are optional strings recorded on the saved
checkpoint for provenance; they take no part in training.

<code-tabs name="train" />

Left alone, the trainer runs 100 epochs with AdamW at `lr0=1e-3`, a batch of
16, 256 px crops, and early stopping after 50 epochs without PSNR improvement.
There is no LoRA path for this family: `lora=True` raises an error rather than
running, since `NAFNetTrainer` never opts in to adapter fine-tuning.

During training the network runs with plain global-average pooling. NAFNet's
inference-only windowed local pooling (Test-time Local Converter) is detached
before the first epoch and reattached once training finishes, since
backpropagating through a fixed-window local pool would not match how the
checkpoint is used at inference.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary with `metrics/PSNR` and `metrics/SSIM`, computed
in RGB over the full valid canvas: SSIM uses an 11x11 Gaussian window with
sigma 1.5, and `fitness` for best-checkpoint selection is the PSNR value.
`data` points at the same paired-image dataset format used for training.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`, with `restored` carrying the output image. NAFNet exports at a
fixed spatial resolution: `imgsz` must be divisible by the network's
downsample factor (16 for both architecture widths), and only the batch dimension
is dynamic when `dynamic=True`; height and width are fixed at export time.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
