---
title: RF-DETR
families: [rfdetr]
seo_title: "RF-DETR: train, fine-tune and export under MIT"
description: "Use RF-DETR in LibreYOLO for detection, instance segmentation, pose and oriented boxes. Install, predict, train, validate and export, all MIT-licensed."
lead: "A detection transformer that predicts a fixed set of objects instead of a dense grid, so it needs no NMS at inference. LibreYOLO supports it for four tasks."
keywords: [RF-DETR, real-time detection transformer, DETR, object detection, instance segmentation, pose estimation, oriented bounding boxes]
last_verified: "1.5.0"
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: "LibreRFDETRs, detection on video at 512 px."
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # Any source the library accepts: file, folder, URL, webcam index,
        # RTSP stream, or a .streams list
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() returns a plain dict, not an object
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Against COCO
      language: bash
      code: |
        # The bundled COCO yaml carries an embedded download script, so it
        # needs explicit permission unless the dataset is already local.
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)

        # Arguments accepted for every format:
        #
        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"
        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"
        #             | "tflite" | "coreml" | "coreai".
        #             "engine" is an alias for tensorrt, "litert" for tflite.
        #   imgsz     int, or (height, width). Defaults to the checkpoint's
        #             native resolution.
        #   batch     int, default 1.
        #   half      bool, export in FP16. Default False.
        #   int8      bool, export in INT8. Default False. Needs `data`.
        #   data      path to a dataset YAML, used to calibrate int8.
        #   fraction  float, share of that calibration set to use. Default 1.0.
        #   dynamic   bool, dynamic axes. Default True.
        #   simplify  bool, run ONNX graph simplification. Default True.
        #   opset     int, ONNX opset. Chosen per family when not given.
        #   device    str, device to trace on. Defaults to the model's device.
        #   output_path  str, defaults to a name derived from the checkpoint.
        #   verbose   bool, default False.
        #   allow_download_scripts  bool, default False. Permits embedded
        #             Python in a dataset YAML that has to be downloaded.
        #
        # A few formats take extra arguments of their own, such as an RKNN
        # target platform. Those are documented on each format's page.
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512
        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreRFDETRs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
    - label: Without LibreYOLO
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # Running the graph directly means doing your own preprocessing and
        # postprocessing. Inspect the signature before wiring anything up.
        session = ort.InferenceSession("LibreRFDETRs.onnx")
        name = session.get_inputs()[0].name
        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512), dtype=np.float32)})

        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
---

## Install

RF-DETR needs its own extra, which pulls in `transformers` for the backbone.

```bash
pip install "libreyolo[rfdetr]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` and `max_det` filter the query
selection; there is no NMS step to tune. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Four sizes, and four tasks that share one architecture: segmentation, pose and
oriented boxes reuse the detection decoder with a different head, so they take
the same arguments. The sizes carry similar parameter counts and differ mainly
in input resolution.

<benchmark-table task="detect" />

<va-embed />

## Train

Training starts from a published checkpoint, for all four tasks. RF-DETR lists
`pretrained` among the arguments its native trainer ignores, so passing
`pretrained=False` does not give you a randomly initialized model here.

<code-tabs name="train" />

Two arguments matter more here than on a CNN detector. Keep `lr0` at or below
`1e-4`, since transformer detectors diverge at learning rates a YOLO model
tolerates. Leave `imgsz` at the checkpoint's native resolution unless you have a
reason to change it. The input must divide evenly by the backbone patch size
times the window count; LibreYOLO checks this before the run starts and names
the nearest valid sizes.

See [training](/docs/train) for datasets, augmentation, multi-GPU and loggers.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Running the graph in a bare runtime, with no LibreYOLO installed, is
also supported, but then preprocessing and postprocessing are yours to write.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />

