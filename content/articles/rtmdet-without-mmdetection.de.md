---
title: "RTMDet ohne mmdetection installieren: schnelle Inferenz mit LibreYOLO"
description: "RTMDet gehört zu den schnellsten präzisen Detektoren. Die offizielle Installation funktioniert mit keiner PyTorch-Version der letzten zwei Jahre. Hier ist die Alternative."
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "Warum lässt sich mmcv mit aktuellen PyTorch-Versionen nicht installieren?"
    a: "Das neueste mmcv-Wheel, Version 2.2.0 vom April 2024, enthält vorgebaute Binärdateien nur bis torch 2.4 / CUDA 12.1. Bei jeder neueren PyTorch-Version muss mmcv seine C++/CUDA-Operatoren aus dem Quellcode kompilieren. Das dauert 10 bis 30 Minuten und erfordert ein passendes CUDA-Toolkit, nvcc und einen kompatiblen C++-Compiler. Daher stammen Fehler wie ein fehlendes mmcv._ext-Modul."
  - q: "Kann ich RTMDet ohne Installation von mmdetection ausführen?"
    a: "Ja. LibreYOLO lädt RTMDet-Gewichte, die aus den OpenMMLab-Checkpoints des Upstream-Projekts konvertiert wurden. Die Inferenz ist mit mmdetection auf demselben Checkpoint bitäquivalent. Lade LibreRTMDets.pt anhand des Namens, dann wird es automatisch heruntergeladen. Alle fünf Größen von Tiny bis X laufen mit 640 px."
  - q: "Ist RTMDet für die kommerzielle Nutzung kostenlos?"
    a: "Ja. MMDetection und RTMDet stehen unter Apache 2.0, für die konvertierten Gewichte gelten dieselben Apache-2.0-Bedingungen und der LibreYOLO-Code steht unter MIT. Es gibt keine kommerziellen Einschränkungen."
  - q: "Wann ist mmdetection weiterhin die richtige Wahl?"
    a: "Wenn du RTMDet mit der vollständigen MMDetection-Augmentierungspipeline trainieren oder feinabstimmen musst oder bereits tief im OpenMMLab-Ökosystem steckst und mmcv funktioniert. Für Inferenz und Deployment ist LibreYOLO der bessere Weg."
---

RTMDet ist ein Echtzeit-Detektor von OpenMMLab, der eine hohe COCO-Accuracy erreicht und zugleich schnell genug für das Deployment bleibt. Die Architektur ist übersichtlich. Die Installation ist es nicht.

Um RTMDet aus der offiziellen Quelle zum Laufen zu bringen, musst du einen vierteiligen OpenMMLab-Stack zusammensetzen:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

Die Versionsvorgaben sind eng. mmdet 3.3.0 verlangt `mmcv < 2.2.0`, aber `mim install mmcv>=2.0.0` installiert ohne Weiteres Version 2.2.0, die dann an einer Laufzeitassertion scheitert. Du musst die Version von Hand festlegen.

Dann kommt das größere Problem. Das neueste mmcv-Wheel ist Version 2.2.0, wurde im April 2024 veröffentlicht und seitdem nicht aktualisiert. Vorgebaute Binärdateien gibt es nur bis **torch 2.4 / CUDA 12.1**. Eine frische Installation mit `pip install torch` liefert dir heute PyTorch 2.12. Wegen dieser Lücke muss mmcv seine C++/CUDA-Operatoren aus dem Quellcode kompilieren: Das dauert 10 bis 30 Minuten und erfordert ein passendes CUDA-Toolkit, `nvcc` und einen kompatiblen C++-Compiler. Dabei häufen sich die dokumentierten Fehler:

* `ModuleNotFoundError: No module named 'mmcv._ext'`: Die kompilierten Operatoren wurden nicht erstellt oder passen nicht zur Installation.

* `nvcc fatal: Unsupported gpu architecture 'compute_86'` auf jeder RTX-30-Serie-Grafikkarte.

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`: Der Stack zwingt dich zurück zu Python 3.8.

* Segmentation Fault beim Import wegen einer inkompatiblen GCC-Version.

Die FAQ von OpenMMLab selbst empfiehlt, mmcv mit pip statt mit mim zu installieren, um die Versionsfalle zu vermeiden. Auf der Get-Started-Seite steht dagegen, dass du mim verwenden sollst. Beide Dokumente sind offiziell. Sie widersprechen einander.

Sobald der Stack läuft, musst du für die Inferenz immer noch eine Konfigurationsdatei auswählen, deren Name das Trainingsrezept codiert. Dazu kommt ein separat heruntergeladener Checkpoint mit einem Hash im Dateinamen, den du über eine Registry einbindest, die du erst kennenlernen musst.

LibreYOLO ersetzt das alles:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

Kein `mim`, keine Versionsmatrix mit vier Paketen, keine Kompilierung aus dem Quellcode, keine Konfigurationsdateien, keine Suche nach Checkpoints anhand ihres Hashes und keine Bindung an Python 3.8. Die Gewichte wurden aus den OpenMMLab-Checkpoints des Upstream-Projekts konvertiert. Die Inferenz ist mit mmdetection auf demselben Checkpoint bitäquivalent.

Es gibt fünf Größen: Tiny, Small, Medium, Large und X. Alle laufen mit 640 px.

```python
print(results[0].boxes.xyxy)  # xyxy coordinates
print(results[0].boxes.conf)  # confidence scores
```

## Wo das Original weiterhin die richtige Wahl ist

Wenn du RTMDet mit der vollständigen MMDetection-Augmentierungspipeline trainieren oder feinabstimmen musst oder bereits tief im OpenMMLab-Ökosystem steckst und mmcv funktioniert, bleib dabei. Für Inferenz und Deployment ist LibreYOLO der bessere Weg.

## Ein Hinweis zur Lizenz

MMDetection und RTMDet stehen unter Apache 2.0. Der LibreYOLO-Code steht unter MIT. Für die Gewichte gelten dieselben Apache-2.0-Bedingungen wie für das Upstream-Projekt. Es gibt keine kommerziellen Einschränkungen.

## Ausprobieren

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO steht unter MIT, läuft auf Linux, Mac und Windows und funktioniert ohne Codeänderungen auf GPU, Apple Silicon und einer gewöhnlichen CPU. Eine API deckt RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, Segmentierung, Pose, Tiefe und mehr ab.

Gib dem Projekt einen Stern auf GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Doku: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
