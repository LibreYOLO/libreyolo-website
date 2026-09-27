---
title: Paddle
seo_title: PaddlePaddle-Export aus LibreYOLO
description: >-
  Einen LibreYOLO-Detektor über X2Paddle in ein PaddlePaddle-Inferenzmodell
  konvertieren: die gepinnte Toolchain, statische FP32-Graphen mit Batch 1 und
  CPU-Inferenz.
lead: >-
  PaddlePaddle-Inferenzmodelle bestehen aus einem model.pdmodel-Graphen neben
  einer model.pdiparams-Gewichtsdatei. LibreYOLO exportiert einen statischen
  ONNX-Graphen mit Opset 15, konvertiert ihn mit X2Paddle und packt das Ergebnis
  mit einer metadata.yaml zusammen, damit es über dieselbe Factory lädt wie jede
  andere Runtime.
keywords:
  - yolo nach paddle exportieren
  - paddlepaddle inferenz
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="paddle")
    mono: true
  - label: Schreibt
    value: 'Ein Verzeichnis mit model.pdmodel, model.pdiparams und metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Lädt zurück
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Backend
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Shapes
    value: 'Statisch, Batch 1, Opset 15. Alle drei werden erzwungen.'
  - label: Präzision
    value: 'Nur FP32, nur CPU.'
  - label: Toolchain
    value: 'PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 oder älter, exakt geprüft'
verification: >-
  Gelesen aus libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md und
  pyproject.toml auf dem dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: >
        # Python 3.10 bis 3.12. Unter Windows ist WSL2 mit Ubuntu 22.04
        validiert.

        pip install "libreyolo[paddle]"
    - label: Die gepinnten Versionen prüfen
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Schreibt das Verzeichnis weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Argumente
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; quadratisches Canvas dieser Familie
            batch=1,          # jeder andere Wert löst ValueError aus
            dynamic=False,    # True löst ValueError aus
            simplify=True,    # False löst ValueError aus
            opset=15,         # jeder andere Wert löst ValueError aus
            output_path=None, # None schreibt weights/<stem>_paddle
        )
  run:
    - label: Über LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: Das Backend direkt
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # Was LibreYOLO() für ein Paddle-Verzeichnis baut. Gleiches Results-
        # Objekt, kein Factory-Routing dazwischen.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Reines Paddle
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Vor- und Nachverarbeitung liegen auf diesem Weg bei dir.
  support:
    - label: Eine Familie und Aufgabe vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Installation

<code-tabs name="install" />

Das Extra pinnt genau den Stack, den die Paritätsarbeit gemessen hat:
PaddlePaddle 2.6.2, X2Paddle 1.6.0 und ONNX 1.17 oder älter. Diese Pins werden
beim Export geprüft, nicht nur bei der Installation, und eine abweichende Version
löst einen `ImportError` aus, der die erwartete nennt. Neuere Paddle-Releases
lehnen Teile des statischen Codes ab, den X2Paddle 1.6.0 erzeugt, deshalb ist ein
früher Fehlschlag besser als ein Artefakt, das niemand validiert hat.

## Export

<code-tabs name="export" />

Vier Argumente sind festgelegt und nicht bloß vorbelegt. `dynamic` muss `False`
sein, `batch` muss 1 sein, `simplify` muss `True` sein, damit der
Konvertierungsgraph vollständig statisch ist, und `opset` muss 15 sein, die
Obergrenze, die X2Paddle 1.6.0 akzeptiert. Alles andere löst einen Fehler aus,
bevor das Tracing beginnt.

Auf dem Zwischengraphen läuft eine Normalisierung. ONNX definiert eine
weggelassene MaxPool-Dilation als eins, PyTorch schreibt das explizite Attribut
aus lauter Einsen, und X2Paddle 1.6.0 lehnt es ab, deshalb entfernt der Exporter
diesen redundanten Default und lässt die angegebene Operation unverändert.

Das Artefakt ist ein Verzeichnis: `model.pdmodel`, `model.pdiparams` und
`metadata.yaml`. Der Python-Code, den X2Paddle während der Konvertierung erzeugt,
gehört nicht dazu.

## Ausführung des Artefakts

<code-tabs name="run" />

`LibreYOLO()` erkennt jedes Verzeichnis, das sowohl `model.pdmodel` als auch
`model.pdiparams` enthält, liest `metadata.yaml` und liefert dasselbe
`Results`-Objekt wie der Checkpoint. Ein anderes Device als `auto` oder `cpu`
löst einen Fehler aus: Dieses Backend läuft nur auf der CPU.

Die Factory baut `PaddleBackend`, exportiert aus `libreyolo` und importierbar als
`libreyolo.backends.paddle.PaddleBackend`. Baue es selbst, wenn du das Backend
ohne das Suffix-Routing der Factory willst, etwa um `task=` explizit für ein
Verzeichnis zu setzen, dessen `metadata.yaml` nicht von dir stammt. Sein
`predict()` nimmt dieselben Quellen und liefert dieselben Ergebnisse.

Das Snippet mit der reinen Runtime spiegelt, was das Backend konfiguriert, und
die drei abgeschalteten Optionen sind Absicht. Die CPU-Fusion-Pipeline von Paddle
2.6 kann abstürzen, während sie die großen Gather- und Scatter-Graphen optimiert,
die für Deformable Attention entstehen, deshalb wurde die Parität gegen den
portablen, unfusionierten statischen Graphen gemessen. Vorverarbeitung, Decoding,
NMS und das Umrechnen der Koordinaten liegen auf diesem Weg bei dir.

## Einschränkungen

Keine dynamischen Shapes, kein FP16, kein INT8, kein eingebettetes NMS, keine
GPU-Runtime.

Validiert sind YOLO9-Erkennung, YOLO9-E2E- und YOLO9-P2-Erkennung, EC-Erkennung,
-Pose und -Segmentierung, RT-DETRv4-, D-FINE-, DEIM- und DEIMv2-Erkennung sowie
YOLO-NAS-Erkennung und -Pose. Jede Kombination ist durch Konvertierung, einen
Reload in der CPU-Runtime, Parität der Rohausgaben und übereinstimmende
öffentliche Ergebnisse abgedeckt.

Blockiert, mit dem Grund pro Kombination:

| Kombination | Grund |
|---|---|
| RF-DETR, alle Aufgaben | Braucht ONNX-Opset 17 und GridSample; X2Paddle 1.6.0 akzeptiert Opset 15 oder niedriger und hat keinen GridSample-Mapper |
| RT-DETR- und RT-DETRv2-Erkennung | Die trainierten Graphen brauchen GridSample ab Opset 16 |
| D-FINE-Segmentierung | Konvertiert und lädt wieder, aber der relative RMS-Fehler der Mask-Logits liegt bei 3.52 % und die minimale IoU der zugeordneten Masken bei 0.582 |
| YOLO9-Segmentierung | YOLO9 ist in LibreYOLO reine Erkennung |
| RTMDet-Ins-Segmentierung | Das Mask-Decoding mit dynamischen Kerneln hat keinen Vertrag für exportierte Runtimes |

Alles, was weder als validiert noch als blockiert aufgeführt ist, wird mit dem
Hinweis abgelehnt, dass es über den Konvertierungsweg von ONNX nach Paddle nicht
validiert wurde.

Das vollständige Raster aus Familien und Aufgaben findest du in
[der Export-Matrix](/docs/reference/export-matrix). Für eine einzelne Kombination:

<code-tabs name="support" />
