---
title: Schlanke Installation
seo_title: LibreYOLO-ONNX-Inferenz ohne PyTorch ausführen
description: >-
  Installiere LibreYOLO mit --no-deps und führe ONNX-Erkennung ausschließlich
  mit numpy aus, ohne torch auf dem Datenträger. Erfahre mehr über Vorgehen,
  Grenzen und die genaue Paketliste.
lead: >-
  Der ONNX-Inferenzpfad von LibreYOLO verwendet von Anfang bis Ende numpy,
  einschließlich Decodierung und NMS. Zur Laufzeit benötigt er kein PyTorch.
  Eine Installation ohne Abhängigkeitsauflösung kann daher Erkennungen
  ausführen, obwohl torch auf dem Rechner fehlt.
keywords:
  - inferenz ohne torch
  - libreyolo ohne pytorch
  - onnx inference ohne torch
  - libreyolo schlank installieren
  - pip install no-deps
  - libreyolo speicherplatz
  - onnxruntime inference
last_verified: 1.5.0
meta:
  - label: Gilt für
    value: 'ONNX-Erkennung, sieben Modellfamilien'
  - label: Einstiegspunkt
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: Support-Stufe
    value: 'Best Effort, keine eigene Distribution'
snippets:
  install:
    - label: Schlank
      language: bash
      code: |
        # Paket ohne Abhängigkeitsliste installieren, danach die vier Pakete,
        # die der ONNX-Erkennungspfad tatsächlich importiert.
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: Nur-CPU-torch
      language: bash
      code: |
        # Probiere dies zuerst. Es behält alle Funktionen bei und vermeidet das
        # CUDA-Wheel, das den größten Teil des Speicherplatzes benötigt.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo.backends.onnx import OnnxBackend


        model = OnnxBackend("libreyolo9t.onnx")

        result =
        model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")


        # xyxy ist hier ein numpy-ndarray und kein torch-Tensor.

        print(result.boxes.xyxy)

        print(result.boxes.conf)

        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## Funktionsweise

`pip install --no-deps libreyolo` installiert das Paket und überspringt seine
Abhängigkeitsliste vollständig. Es wird nichts für dich aufgelöst, und du bist
selbst für die Installation der tatsächlich verwendeten Pakete verantwortlich.

Das ist nur dann sinnvoll, wenn der gewünschte Codepfad die übersprungenen
Abhängigkeiten wirklich nicht benötigt. Bei der ONNX-Erkennung ist dies der
Fall. Die Decodierung einschließlich Non-Maximum Suppression verwendet numpy.
Auch die Vorverarbeitungsrezepte verwenden numpy. PyTorch ist eine Abhängigkeit
für Training und Eager-Inferenz und wird in diesem Pfad nie aufgerufen.

Vor diesem Release scheiterte bereits der Import. Jeder Import unter
`libreyolo.models` erstellte alle Modellklassen, um das Register für die
automatische Checkpoint-Erkennung zu füllen. Diese Klassen sind Unterklassen
von `torch.nn.Module`. Die Vorverarbeitungsrezepte befinden sich jetzt in einem
eigenen Paket namens `libreyolo.preprocess`. Der torch-Import wird verzögert,
bis ein torch-Attribut verwendet wird. Dadurch lässt sich der ONNX-Pfad ohne
torch auf dem Rechner importieren. Das Paket enthält einen nativen
numpy-Preprocessor pro Familie: `yolo9`, `yolonas`, `yolox`, `ec`, `rtdetr`,
`rfdetr`, `dfine`, `deim` und `deimv2`. Das sind zwei mehr als die sieben
nachfolgend vollständig geprüften Familien. Jede Datei
`libreyolo/models/<family>/utils.py` exportiert die Funktionen erneut, damit
bestehende Importpfade weiter funktionieren.

## CPU-Wheel als erste Wahl

Die meisten Nutzer dieses Verfahrens möchten eine mehrere Gigabyte große
Installation vermeiden. Fast der gesamte Umfang entsteht an einer Stelle:
Das normale `torch`-Wheel enthält CUDA. Ein reiner CPU-Build ist deutlich
kleiner und benötigt keinen besonderen Installationspfad.

<code-tabs name="install" />

Die CPU-Option behält jede LibreYOLO-Funktion bei: Training, Validierung, alle
Aufgaben, alle Familien und die CLI. Verwende den schlanken Pfad, wenn auf dem
Rechner überhaupt kein torch vorhanden sein soll, und nicht nur eine kleinere
Variante.

## Umfang der schlanken Installation

| | |
|---|---|
| Aufgabe | Erkennung |
| Format | ONNX |
| Einstiegspunkt | `OnnxBackend` |
| Schnittstelle | Python-Bibliothek |

Sieben Familien wurden mit diesem Pfad geprüft:
[YOLOv9](/docs/models/yolov9), [YOLO-NAS](/docs/models/yolo-nas),
[EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr),
[RF-DETR](/docs/models/rf-detr), [D-FINE](/docs/models/d-fine) und
[DEIM](/docs/models/deim), jeweils einschließlich der Varianten einer Familie.

Dies ist der geprüfte Umfang und keine von der Bibliothek erzwungene Grenze.
Andere Aufgaben und Familien wurden lediglich nicht getestet. Einige laden
beim Aufruf torch, andere funktionieren möglicherweise. Betrachte alles
außerhalb dieser Liste als ungetestet und nicht als unterstützt oder defekt.

Innerhalb des Umfangs sind die Ergebnisse nicht nur ähnlich, sondern identisch
mit der normalen Installation. Jede Familie wurde nach ONNX exportiert und
zweimal ausgeführt, einmal normal und einmal mit blockiertem torch. Boxen,
Scores und Klassen stimmten exakt überein. Ein Paritätstest in der Testsuite
schützt diesen Vertrag vor unbeabsichtigten Änderungen.

## Fünf häufige Fallstricke

**Verwende `OnnxBackend` und nicht die Modellklassen.**
`LibreYOLO9("model.onnx")` benötigt weiterhin torch, weil `LibreYOLO9` selbst
eine Unterklasse von `nn.Module` ist. Dieser Fehler ist besonders naheliegend,
da jede andere Seite dieser Dokumentation ein Modell über seine Klasse oder
über `LibreYOLO()` lädt.

**Exportiere auf einem anderen Rechner.** Zum Erzeugen der `.onnx`-Datei wird
torch benötigt. Der schlanke Rechner kann sie daher nicht erstellen.
Exportiere das Artefakt auf einem Entwicklungs- oder CI-Rechner und übertrage
es anschließend auf das schlanke Zielsystem.

**Results enthalten numpy-Arrays.** `result.boxes.xyxy` ist hier ein `ndarray`.
Die Container akzeptieren beide Typen, sodass die Attributnamen gleich bleiben.
Code, der `.cpu()` oder `.numpy()` für ein Ergebnis aufruft, schlägt jedoch
fehl.

**Ein einzelnes Bild gibt ein einzelnes `Results` zurück.** `predict()` gibt
für ein Bild ein `Results`-Objekt und für mehrere Bilder eine Liste zurück. Die
Indizierung eines einzelnen Ergebnisses mit `[0]` wählt die erste Erkennung und
nicht das erste Bild. Dadurch entsteht unbemerkt ein Ergebnis mit einer Box,
statt dass ein Fehler ausgelöst wird.

**Die CLI funktioniert nicht.** `typer` und `click` gehören nicht zu den vier
Paketen. Der Befehl `libreyolo` ist daher nicht verfügbar. Dies ist eine
Bibliotheksinstallation.

## Vorhersage

<code-tabs name="predict" />

Ersetze `onnxruntime` durch `onnxruntime-gpu`, um CUDA zu verwenden. Die vier
Pakete sind genau diejenigen, die ein vollständiger torch-freier Aufruf von
`predict()` tatsächlich importiert. Sie wurden während des Aufrufs erfasst und
nicht nur theoretisch abgeleitet. `opencv-python-headless` ersetzt das
deklarierte `opencv-python`: dasselbe Modul, aber ohne GUI-Bibliotheken und mit
geringerem Speicherbedarf.

Von den übrigen deklarierten Abhängigkeiten wird `requests` nur zum Laden eines
Bilds von einer URL benötigt. `pycocotools` und `scipy` dienen der Validierung
und Evaluierung, `typer` und `click` der CLI.

## Absichtliche Veränderlichkeit der Liste

Die obige Paketliste gilt für das am Anfang dieser Seite genannte Release. Mit
`--no-deps` verzichtest du auf die Abhängigkeitsauflösung. Nichts prüft die
Liste für dich, und ein späteres Release kann ein hier nicht genanntes Paket
importieren.

Wenn ein `ModuleNotFoundError` auftritt, kennst du bereits das Verfahren:
Installiere das fehlende Paket. Dies ist das vorgesehene Wartungsmodell und
kein Fehlerbericht. Der Pfad wird nach Best Effort unterstützt und ist keine
eigene Distribution. Deshalb gibt es weder ein zweites schlankes Paket auf
PyPI noch Pläne dafür.

Prüfe explizit, ob deine Umgebung wirklich torch-frei ist und nicht unbemerkt
auf eine installierte Kopie zurückgreift:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

Diese Prüfung sollte im CI-System des schlanken Images erhalten bleiben. Ohne
sie besteht eine Umgebung mit zufällig vorhandenem torch jeden Test, ohne dir
etwas über den torch-freien Pfad zu sagen.

