---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: 'Mask R-CNN in LibreYOLO: Vorhersage, Validierung und Export'
description: >-
  Führe Mask R-CNN in LibreYOLO für Objekterkennung und Instanzsegmentierung
  aus. Installiere, sage vorher, validiere und exportiere die
  torchvision-Portierung unter BSD-3-Clause.
lead: >-
  Mask R-CNN ergänzt Faster R-CNN um einen Maskenzweig pro Region und sagt neben
  jeder erkannten Box eine Segmentierungsmaske vorher. LibreYOLO portiert die
  torchvision-Implementierung für Objekterkennung und Instanzsegmentierung.
keywords:
  - mask r-cnn python
  - instanzsegmentierung
  - objekterkennung
  - faster r-cnn
  - torchvision
  - zweistufiger detektor
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Nur Boxen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" überspringt den Masken-Head und gibt Boxen aus demselben
        # Checkpoint zurück, ohne Masken im Ergebnis.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # Masken
        print(metrics["metrics/mAP50-95(B)"])   # Boxen
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreMaskRCNNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## Installation

Mask R-CNN benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. Wenn du den Checkpoint ohne Argument `task` lädst, werden Instanzmasken
zurückgegeben, weil die Segmentierung die Standardaufgabe dieser Familie ist.
`result.masks` enthält sie neben den Boxen. Mit `task="detect"` werden dieselben
Gewichte ohne Masken-Head geladen und nur Boxen zurückgegeben. `conf` und `iou`
legen die Schwellenwerte für Confidence und NMS fest. Im Gegensatz zu einem
Query-basierten Detektor behält Mask R-CNN seinen Upstream-NMS-Schritt bei.
Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und die
Verarbeitung von Ergebnissen.

## Varianten

Es gibt ein Backbone: ResNet-50 mit einer Feature Pyramid, aufgebaut mit dem
v2-Builder für Mask R-CNN von torchvision. Der veröffentlichte Checkpoint steht
unter BSD-3-Clause und dient beiden Aufgaben dieser Familie. Es gibt daher
keine Größen zur Auswahl.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln zurück. Bei der
standardmäßigen Segmentierungsaufgabe dieses Checkpoints enthält der einfache
Schlüssel `metrics/mAP50-95` den Masken-Score. Derselbe Lauf meldet Boxen mit
dem Suffix `(B)`, sodass beide in einem Durchlauf verfügbar sind.

<code-tabs name="val" />

## Export

<export-matrix />

Mask R-CNN lässt sich nur nach ONNX und mit Batch-Größe 1 exportieren. Der
exportierte Graph enthält weiterhin die Upstream-Schritte für Größenänderung
und Einfügen der Masken. LibreYOLO erzwingt daher unabhängig vom übergebenen
Wert `dynamic=True`, damit der Graph für nicht quadratische Quellen gültig
bleibt. Eine exportierte `.onnx`-Datei wird anhand ihrer Endung wieder über
`LibreYOLO()` geladen und gibt dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie. Der folgende einzelne
Checkpoint wird unter Erkennung aufgeführt, aber dieselbe Datei kann auch für
die Segmentierung geladen werden. Übergib kein Argument `task`, damit sie
standardmäßig Masken zurückgibt.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Mask R-CNN ist als Unterklasse des Faster-R-CNN-Wrappers von LibreYOLO
implementiert. Es verwendet dieselbe torchvision-Quelle und BSD-3-Clause-Lizenz
und ergänzt den Masken-Prädiktor und den Masken-RoI-Head aus demselben
portierten Commit.

</provenance-box>
