---
title: YOLO-NAS
families:
  - yolonas
seo_title: 'YOLO-NAS: Vorhersage, Training und Export in LibreYOLO'
description: >-
  Verwende YOLO-NAS in LibreYOLO für Objekterkennung und Posenschätzung. Die
  Gewichte von Deci.AI sind proprietär und nicht kommerziell nutzbar. LibreYOLO
  veröffentlicht keine davon.
lead: >-
  Ein konvolutionaler Detektor, dessen Backbone und Neck aus der
  Architektursuche von Deci.AI hervorgingen und auf quantisierungsbewussten
  RepVGG-Blöcken basieren. Die Gewichte stammen von Deci.AI und sind nur für die
  nicht kommerzielle Nutzung lizenziert. LibreYOLO veröffentlicht keine davon.
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - Objekterkennung
  - Posenschätzung
  - quantisierungsbewusster Detektor
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Ein Name, der noch nicht lokal vorliegt, wird von Decis CDN geladen.

        # Vor dem Download werden Decis Lizenzbedingungen angezeigt; mit dem
        Herunterladen akzeptierst du sie.

        model = LibreYOLO("LibreYOLONASs.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Posenschätzung
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Das Suffix -pose wählt den Pose-Kopf und dessen eigenen Gewichtssatz
        aus.

        model = LibreYOLO("LibreYOLONASs-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Von Grund auf
      language: python
      code: >
        from libreyolo import LibreYOLONAS


        # Es wird kein Deci-Checkpoint verwendet: Das Modell startet mit
        zufälligen Gewichten,

        # sodass das Ergebnis des Laufs ausschließlich aus deinen Daten
        hervorgeht.

        model = LibreYOLONAS(None, size="s")

        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Gegen COCO
      language: bash
      code: >
        # Die mitgelieferte COCO-YAML-Datei enthält ein Downloadskript und
        benötigt

        # daher eine ausdrückliche Berechtigung, sofern der Datensatz noch nicht
        lokal vorliegt.

        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreYOLONASs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## Installation

YOLO-NAS benötigt neben dem Basispaket kein zusätzliches Extra.

```bash
pip install libreyolo
```

## Vorhersage

Ein Checkpoint-Name, der noch nicht lokal vorliegt, wird von Decis öffentlichem CDN und nicht von der LibreYOLO-Organisation geladen, da diese keine solchen Gewichte hostet. Vor Beginn der Übertragung zeigt die Bibliothek Decis Lizenzbedingungen einmal pro Prozess an. Bevor die heruntergeladene Datei geöffnet wird, prüft sie deren SHA-256 gegen einen fest hinterlegten Wert. Welche Nutzung diese Bedingungen erlauben, steht unter [Lizenzierung](#licensing).

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem jeder anderen Familie, sodass der Wechsel zu einem anderen Detektor nur eine einzige Codezeile erfordert. `conf` legt den Konfidenzschwellenwert und `iou` den NMS-Schwellenwert fest. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Objekterkennung und Posenschätzung verwenden dieselbe Architektur mit unterschiedlichen Köpfen und akzeptieren dieselben Argumente. Die Größen in der folgenden Tabelle gehören zur Objekterkennung. Für die Posenschätzung ist zusätzlich eine kleinere Größe verfügbar. Der Pose-Kopf sagt den COCO-Keypoint-Satz vorher.

<benchmark-table task="detect" />

<va-embed />

## Training

<code-tabs name="train" />

Wenn du `epochs`, `lr0` und `amp` auslässt, werden sie aufgabenspezifisch bestimmt. Ein Lauf zur Posenschätzung beginnt daher mit anderen Standardwerten als ein Erkennungslauf. Als Optimierer wird standardmäßig AdamW verwendet. Die Klassenanzahl stammt aus der Datensatz-YAML-Datei und der Kopf wird vor der ersten Epoche entsprechend neu aufgebaut. Beim Pose-Kopf wird die Keypoint-Anzahl auf dieselbe Weise behandelt, sodass sich ein COCO-Pose-Checkpoint auf ein Skelett anderer Größe feinabstimmen lässt.

Das Fine-Tuning beginnt mit Decis Gewichten, die Decis Lizenz unterliegen. Beim Training eines zufällig initialisierten Modells wird dagegen keinerlei Deci-Checkpoint verwendet. Dies zeigt das dritte Snippet oben.

Unter [Training](/docs/train) findest du Informationen zu Datensätzen, Augmentation, Multi-GPU und Loggern.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall, mAP 50 und mAP 50-95 zurück, gemessen anhand jedes Datensatzes im Format, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Du kannst den Graphen auch in einer eigenständigen Laufzeitumgebung ohne installiertes LibreYOLO ausführen. Dann musst du Vor- und Nachverarbeitung selbst implementieren. Jedes Format installiert ein anderes Extra und besitzt eigene Argumente. Beides ist auf der Seite des jeweiligen Formats beschrieben.

Ein Export ist eine weitere Kopie derselben Gewichte in einem anderen Container. Der Export eines Deci-Checkpoints ändert weder die Herkunft der Gewichte noch die dafür geltende Lizenz.

<code-tabs name="export" />

## Checkpoints

Es gibt keine aufzulisten. Decis Lizenz verbietet die Weiterverteilung, daher veröffentlicht die LibreYOLO-Organisation keine YOLO-NAS-Gewichte und der Download wird an anderer Stelle aufgelöst. Ein Name der Form `LibreYOLONAS<size>.pt` oder für die Posenschätzung `LibreYOLONAS<size>-pose.pt` wird dem passenden Objekt auf Decis öffentlichem CDN zugeordnet.

Auf diese Weise lassen sich nur Checkpoints abrufen, deren SHA-256 die Bibliothek fest hinterlegt hat. Alle anderen werden sicher abgelehnt, statt ein ungeprüftes Pickle eines Drittanbieters zu öffnen. Du musst sie manuell herunterladen und als Pfad übergeben. Eine bereits lokal vorhandene Datei wird ohne Download und Prüfsummenprüfung von ihrem Pfad geladen. Das gilt auch für eine Deci-Datei mit der ursprünglichen Endung `.pth`, die der Loader erkennt.

## Lizenzierung

<provenance-box>

LibreYOLO hostet oder spiegelt diese Gewichte nicht. In der LibreYOLO-Organisation auf Hugging Face gibt es nichts für diese Familie. Jeder automatische Download verwendet stattdessen Decis öffentliches CDN, zeigt vor dem Start einmal pro Prozess Decis Bedingungen an und wird vor dem Öffnen anhand einer fest hinterlegten SHA-256 geprüft.

Eine Alternative ist das Training mit einem zufällig initialisierten Modell. Die Architektur steht im Upstream-Projekt unter Apache-2.0 und hier unter MIT. Ein auf diese Weise mit deinen eigenen Daten trainiertes Modell leitet sich daher von keinem Deci-Checkpoint ab.

</provenance-box>

## Zitieren

YOLO-NAS wurde ohne Paper veröffentlicht. Der folgende Eintrag entspricht der von den Autoren gewünschten Angabe und bezieht sich auf SuperGradients, die Bibliothek, mit der es veröffentlicht wurde.

<citation-block />
