---
title: HRNet
families:
  - hrnet
seo_title: 'HRNet: Top-down-Pose-Schätzung in LibreYOLO'
description: >-
  Nutze HRNet in LibreYOLO für die Top-down-Pose-Schätzung mit COCO-17.
  Installiere, verwende, validiere und exportiere die W32- und W48-Checkpoints
  mit MIT-Lizenz.
lead: >-
  HRNet ist ein Convolutional Network, das durch wiederholte Multi-Scale-Fusion
  einen hochauflösenden Feature-Stream beibehält, anstatt die Auflösung nach dem
  Downsampling wiederherzustellen. LibreYOLO bindet die offizielle
  Top-down-Pose-Variante für Inferenz und Validierung ein.
keywords:
  - HRNet
  - menschliche pose schätzen
  - top-down pose
  - COCO-17 keypoints
  - hochauflösendes netzwerk
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Keine Personenquelle angegeben: HRNet koppelt sich automatisch mit
        # dem schlanken LibreYOLO9t-Detektor und protokolliert die Wahl einmal.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Personenquelle
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # Erkennung auslassen: das ganze Bild als eine Person behandeln.
        result = model(SAMPLE_IMAGE, cropped=True)

        # Oder HRNet Boxen eines bereits ausgeführten Detektors übergeben.
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # Oder mit einem bestimmten LibreYOLO-Detektor statt mit dem
        # standardmäßigen LibreYOLO9t koppeln.
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Exportierte Datei nutzen
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # Der exportierte Graph ist nur der Heatmap-Head mit fester Canvas:
        # Er erhält einen Batch fertig zugeschnittener, normalisierter
        # Personenausschnitte und gibt rohe Heatmaps zurück. Personenerkennung,
        # Zuschnittgeometrie, Heatmap-Decodierung und OKS-Unterdrückung gehören
        # nicht zu diesem Graphen. Außerhalb von LibreYOLO musst du diesen
        # Decodierungsschritt selbst neu implementieren.
        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")
        name = session.get_inputs()[0].name
        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## Installation

HRNet benötigt über das Basispaket hinaus keine weiteren Abhängigkeiten.

```bash
pip install libreyolo
```

Sein standardmäßiger Personendetektor, ein schlanker LibreYOLO9t-Checkpoint,
wird automatisch heruntergeladen, wenn HRNet zum ersten Mal mit ihm gekoppelt wird.

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen
und lokal zwischengespeichert.

<code-tabs name="predict" />

HRNet ist ein Top-down-Pose-Schätzer: Bevor der Pose-Head ausgeführt werden
kann, benötigt er eine Personenbox, daher ermittelt jeder Aufruf eine solche
Box. Ohne weitere Angaben koppelt er sich beim ersten Mal mit einem
LibreYOLO9t-Detektor und protokolliert diese Wahl. `cropped=True` überspringt
die Erkennung und behandelt das gesamte Bild als eine Person. `person_boxes`
akzeptiert Boxen eines bereits ausgeführten Detektors. `person_detector`
akzeptiert `"auto"`, `"rfdetr"`, ein beliebiges LibreYOLO-Erkennungsmodell
oder ein einfaches Callable. Mit `flip_test=True` wird das Modell zusätzlich
auf dem horizontal gespiegelten Ausschnitt ausgeführt und der Mittelwert der
beiden Heatmaps gebildet. Dies ist die eigene Test-Time-Augmentierung von
HRNet, die generische Option `augment=True` ist hier nicht definiert. Quellen
mit mehreren Bildern werden sequenziell verarbeitet: Der Detektor von HRNet
und die je Bild variable Personenanzahl unterstützen keine gestapelte
Vorhersage. Unter [Vorhersage](/docs/predict) findest du Informationen zu
Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Es gibt zwei Größen, `w32` und `w48`. Beide sagen den standardmäßigen
COCO-17-Keypoint-Satz aus einem Personenbild mit fester Auflösung voraus. `w48`
ist das breitere der beiden Backbones.

Der Model Zoo des Upstream-Projekts gibt die Pose-Accuracy für jede Größe mit
einem eigenen Personendetektor, einer eigenen Flip-Test-Konfiguration und dem
offiziellen COCO-Evaluierungsprotokoll an. Die Standardkopplung von LibreYOLO
verwendet einen anderen Detektor. Ein Validierungslauf misst hier daher diese
Kombination und nicht die des Upstream-Projekts. Um die Upstream-Werte zu
reproduzieren, benötigst du dieselben Personenboxen, Detektor-Scores und
Flip-Einstellungen wie bei der ursprünglichen Evaluierung.

## Validierung

`val()` führt Keypoint-OKS-AP im COCO-Stil aus und akzeptiert eine YOLO-Pose-
`data.yaml` oder eine COCO-Keypoints-JSON-Datei zusammen mit einem
Bilderverzeichnis. Das standardmäßige Metrik-Backend ist faster-coco-eval.
Wenn faster-coco-eval nicht installiert ist, wird automatisch `pycocotools`
verwendet. Mit `faster_coco_eval=False` erzwingst du den `pycocotools`-Pfad.

<code-tabs name="val" />

Die Validierung ruft intern die eigene `predict()`-Methode von HRNet auf und
verwendet daher den Personendetektor, mit dem das Modell erstellt oder
aufgerufen wurde. Erstelle das Modell mit einem expliziten `person_detector=`,
damit diese Quelle über mehrere Läufe hinweg gleich bleibt, anstatt bei jedem
Aufruf erneut den Standard ermitteln zu lassen.

## Export

<export-matrix />

Der Exportvertrag von HRNet umfasst ausschließlich ONNX, TorchScript, OpenVINO
und TensorRT. Bei jedem anderen Format wird vor Beginn des Tracings ein Fehler
ausgelöst. Jeder Export enthält nur den Heatmap-Head mit fester Canvas und
Batch-Größe eins in FP32. Er nimmt einen Personenausschnitt entgegen und gibt
rohe Heatmaps zurück. Die vorgelagerte affine Zuschnittgeometrie sowie die
nachgelagerte Heatmap-Decodierung, Wiederherstellung nach Spiegelung und
OKS-Unterdrückung verbleiben in Python. Eine vollständige Pipeline vom Bild zu
den Keypoints benötigt daher auf der anderen Seite weiterhin LibreYOLO.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />

