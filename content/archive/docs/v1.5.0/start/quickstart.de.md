---
title: Schnellstart
seo_title: LibreYOLO-Schnellstart
description: >-
  Führe einen Detektor auf einem Bild aus, trainiere ihn mit einem kleinen
  Datensatz nach und exportiere ihn nach TorchScript oder ONNX. Alles läuft auf
  der CPU und benötigt etwa zehn Python-Zeilen.
lead: >-
  Der kürzeste Weg durch LibreYOLO: Sage zunächst für ein Bild voraus, trainiere
  dann mit einem kleinen Datensatz und exportiere schließlich das Ergebnis.
  Jeder Befehl auf dieser Seite läuft auf der CPU.
keywords:
  - libreyolo schnellstart
  - libreyolo tutorial deutsch
  - libreyolo vorhersage
  - libreyolo trainieren
  - libreyolo exportieren
  - yolo python beispiel
last_verified: 1.5.0
meta:
  - label: Installation
    value: pip install libreyolo
    mono: true
  - label: Checkpoint
    value: LibreYOLO9t.pt
    mono: true
  - label: Hardware
    value: Eine CPU genügt für alle Beispiele auf dieser Seite
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Lädt den Checkpoint bei der ersten Verwendung und speichert ihn in
        weights/.

        model = LibreYOLO("LibreYOLO9t.pt")


        # Ein einzelnes Bild gibt ein Results-Objekt zurück.

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video und Streams
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True liefert ein Results je Frame, statt eine Liste aufzubauen.

        # Ersetze den Pfad durch einen Webcam-Index, eine RTSP-URL oder einen
        Ordner.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco8 ist ein mitgelieferter Datensatz mit 8 Bildern. Er wird beim
        # ersten Aufruf von einer URL geladen, ohne ein Skript auszuführen.
        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )

        print(results["save_dir"])
        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Validierung
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() gibt ein einfaches Dictionary und kein Objekt zurück.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export() gibt den geschriebenen Pfad zurück.
        path = model.export(format="torchscript")
        print(path)

        # Die Factory routet nach Dateiendung. Das Artefakt wird daher wie ein
        # Checkpoint geladen und gibt dasselbe Results-Objekt zurück.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## Installation

```bash
pip install libreyolo
```

Das ist alles, was die nachfolgenden Abschnitte zu Vorhersage und Training
benötigen. Der Export nach ONNX ergänzt ein Extra. Die vollständige Liste
findest du unter [Installation](/docs/install).

## Vorhersage

<code-tabs name="predict" />

`LibreYOLO()` ist eine Factory. Sie liest die Datei, ermittelt die Familie der
Gewichte und gibt das Modell dieser Familie zurück. Der Wechsel zu einem
anderen Detektor erfordert daher nur eine geänderte Zeile. Wenn du
`LibreYOLO9t.pt` ohne Verzeichnis übergibst, wird relativ zum Arbeitsverzeichnis
nach `weights/LibreYOLO9t.pt` gesucht und die Datei dort heruntergeladen, falls
sie fehlt. Unter [Checkpoints und Gewichte](/docs/weights) findest du die
Downloadregeln und Informationen zum Offline-Betrieb.

`save=True` schreibt eine annotierte Kopie unter `runs/detect/` in ein
Verzeichnis `predict`, dessen Nummer sich mit jedem Lauf erhöht. Das
zurückgegebene `Results`-Objekt enthält `boxes`, und `names` ordnet einen
Klassenindex seinem Label zu. Ein einzelner Bildpfad gibt ein `Results`-Objekt
zurück. Ein Verzeichnis, eine Bilderliste oder `stream=True` gibt eine Liste
beziehungsweise einen Generator solcher Objekte zurück.

## Training

<code-tabs name="train" />

`data` ist eine Datensatz-YAML. `coco8.yaml` wird mit der Bibliothek
ausgeliefert, weshalb du das Snippet unverändert ausführen kannst. Ein nicht
mitgelieferter Name wird als Pfad gelesen. Datensätze werden unter `~/datasets`
oder, falls die Variable gesetzt ist, unter `LIBREYOLO_DATASETS_DIR` aufgelöst.

Ein Lauf schreibt in `project/name`, standardmäßig in ein Verzeichnis unter
`runs/train`. Darin liegen `weights/best.pt` und `weights/last.pt`. `train()`
gibt ein Dictionary zurück, das `save_dir`, `best_checkpoint`,
`last_checkpoint`, die Loss-Werte jeder Epoche und die Validierungsmetriken
jeder Epoche enthält. Der trainierte Checkpoint wird genau wie der
vortrainierte über `LibreYOLO()` geladen.

Nicht jede Familie kann trainiert werden. Wenn eine Familie nur Inferenz
unterstützt, löst `train()` einen `NotImplementedError` mit einer entsprechenden
Meldung aus. Unter [Grundkonzepte](/docs/concepts) erfährst du, welche
Support-Stufe welche Funktionen umfasst.

## Export

<code-tabs name="export" />

TorchScript benötigt über die Basisinstallation hinaus nichts. Jedes andere
Zielformat besitzt ein eigenes Extra. Die Unterstützung gilt pro Familie und
Aufgabe und ist nicht einheitlich. Siehe
[Export und Deployment](/docs/export).

Zu den von jedem Format akzeptierten Argumenten gehören `imgsz` (eine Ganzzahl
oder ein Paar aus Höhe und Breite), `batch` (Standardwert 1), `half`, `int8`
zusammen mit einer `data`-YAML zur Kalibrierung, `dynamic` (Standardwert True),
`simplify` (Standardwert True), `opset`, `device` und `output_path`. Wenn
`output_path` fehlt, wird die Datei unter `weights/` mit einem aus dem
Checkpoint abgeleiteten Namen geschrieben.

## Nächste Schritte

- [Grundkonzepte](/docs/concepts) erklärt Aufgaben, Familien, Größen und
  Checkpoint-Namen.
- [Checkpoints und Gewichte](/docs/weights) beschreibt automatische Downloads,
  Offline-Betrieb und sicheres Laden.
- Unter [Vorhandene Gewichte importieren](/docs/migrate) erfährst du, wie du
  einen Checkpoint aus einem Upstream-Projekt verwendest.
- [Alle Modelle](/docs/models) hilft dir, die passende Familie für dein Problem
  auszuwählen.
- [Training](/docs/train), [Vorhersage](/docs/predict) und
  [Export](/docs/export) beschreiben die vollständigen Workflows.

