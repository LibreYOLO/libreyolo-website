---
title: "RT-DETR ohne Ultralytics: v1, v2 und v4 mit Apache-2.0"
description: "Führe RT-DETR, RT-DETRv2 und RT-DETRv4 mit Apache-2.0-Gewichten in einer MIT-Bibliothek aus und trainiere sie nach: eine API für predict, train, val und export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "Ist RT-DETR kostenlos kommerziell nutzbar?"
    a: "Ja, wenn du eine permissiv lizenzierte Implementierung verwendest. Der Originalcode von RT-DETR und RT-DETRv2 (lyuwenyu/RT-DETR) sowie RT-DETRv4 (RT-DETRs/RT-DETRv4) stehen unter Apache-2.0, und LibreYOLO führt alle drei Versionen mit Apache-2.0-Gewichten in einer MIT-Bibliothek aus. Die RT-DETR-Implementierung im ultralytics-Paket steht unter AGPL-3.0. Für Closed-Source-Nutzung ist die kostenpflichtige Enterprise License die Alternative. Das sind allgemeine Informationen, keine Rechtsberatung."
  - q: "Unter welcher Lizenz steht RT-DETRv4?"
    a: "Apache-2.0. Die LICENSE-Datei unter github.com/RT-DETRs/RT-DETRv4 weist Apache-2.0 aus. RT-DETRv4 verwendet während des Trainings einen DINOv3-Teacher. Die veröffentlichten Student-Gewichte enthalten keine DINOv3-Parameter, daher erstreckt sich Metas DINOv3-Lizenz nicht auf sie."
  - q: "Kann ich RT-DETR ohne Ultralytics nachtrainieren?"
    a: "Ja. LibreYOLO trainiert RT-DETR-, RT-DETRv2- und RT-DETRv4-Detektions-Checkpoints mit model.train() oder über die Befehlszeile mit libreyolo train, ausgehend von den veröffentlichten COCO-Gewichten. Die Modelle für orientierte Boxen bei RT-DETRv2 unterstützen nur die Inferenz."
  - q: "Welche RT-DETR-Gewichte liefert LibreYOLO mit?"
    a: "RT-DETR in r18, r34, r50, r50m, r101, l und x; RT-DETRv2 in r18, r34, r50, r50m und r101; RT-DETRv4 in s, m, l und x, alle für COCO-Detektion bei 640 px. Für RT-DETRv2 gibt es außerdem Modelle für orientierte Boxen in n, s, m, l und x, trainiert auf DOTA v1.0 bei 1024 px. Jede Datei steht unter Apache-2.0."
---

RT-DETR ist ein Echtzeit-Detection-Transformer von Baidu. Er dekodiert eine feste Menge von Queries statt eines dichten Rasters, daher gibt es keinen NMS-Schritt, den du abstimmen musst. Wenn du danach suchst, ist eine der ersten Implementierungen, die du findest, die im Python-Paket `ultralytics`. Das wirft eine Lizenzfrage auf, die das Modell selbst nie aufwarf.

## Die RT-DETR-Lizenz hängt vom Repository ab

Die Lizenz gilt für ein Repository, nicht für ein Modell. Für RT-DETR gibt es mehrere, und sie stimmen nicht überein:

| Repository | Deckt ab | Lizenz |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR und RT-DETRv2, PyTorch und Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR im `ultralytics`-Paket | AGPL-3.0 oder Enterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 und RT-DETRv4 | MIT-Code, Apache-2.0-Gewichte |

Die Ultralytics-Implementierung ist solide. Auf der [RT-DETR-Seite](https://docs.ultralytics.com/models/rtdetr/) sind die vortrainierten `rtdetr-l.pt` und `rtdetr-x.pt` mit Training, Validierung, Inferenz und Export aufgeführt. Sie wird in einem Paket mit AGPL-3.0-Lizenz ausgeliefert, und Ultralytics verkauft eine Enterprise License für Teams, die nicht ihr gesamtes Projekt als Open Source veröffentlichen möchten. Wenn beides nicht passt, ist die Architektur unter Apache-2.0 bei den ursprünglichen Autoren verfügbar. Der [Überblick zu YOLO-Lizenzen](/articles/yolo-licenses-explained) behandelt dasselbe Muster in der gesamten YOLO-Familie, und der [Leitfaden zu kommerziellen Lizenzen](/articles/yolo-commercial-license) erklärt, was AGPL-3.0 für ein Closed-Source-Produkt bedeutet.

## RT-DETR mit LibreYOLO ausführen

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

Derselbe Aufruf funktioniert über die Befehlszeile:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` und `max_det` filtern ein Top-k-Decoding über Queries und Klassen. `iou` wird akzeptiert, aber nicht verwendet, da es kein NMS gibt. Das zurückgegebene `Results`-Objekt ist dasselbe, das jedes LibreYOLO-Modell zurückgibt. Du kannst Detektoren also mit einer einzeiligen Änderung austauschen.

## RT-DETR, RT-DETRv2 und RT-DETRv4 mit einem Loader

Die Version ist Teil des Dateinamens, und `LibreYOLO()` wählt anhand des Checkpoints die passende Implementierung aus. Deshalb werden alle drei gleich geladen:

* **RT-DETR:** `LibreRTDETR` in r18, r34, r50, r50m, r101 (ResNet-Backbones) sowie l und x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` in r18, r34, r50, r50m und r101. Die Architektur aus Version 1 bleibt erhalten, geändert wird, wie Deformable Attention abtastet.
* **RT-DETRv4:** `LibreRTDETRv4` in s, m, l und x. Die Architektur von D-FINE wird wiederverwendet, und die Gewichte entstehen durch Destillation eines DINOv3-Teachers in einen HGNetv2-Student.

Alle Gewichte für die Detektion wurden auf COCO trainiert und laufen mit 640 px. Zum Vergleich: In den Upstream-READMEs werden 46.5 AP für RT-DETR-R18 und 53.0 AP für RT-DETR-L auf COCO sowie 49.8 AP für RT-DETRv4-S bis 57.0 AP für RT-DETRv4-X angegeben. Die [RT-DETR-Dokuseite](/docs/models/rt-detr) enthält LibreYOLOs eigene Benchmark-Tabelle für jeden Checkpoint.

RT-DETRv2 unterstützt auch orientierte Boxen. `LibreRTDETRv2n-obb.pt` bis `LibreRTDETRv2x-obb.pt` sind die offiziellen DOTA-v1.0-Checkpoints mit 15 Luftbildklassen bei 1024 px, konvertiert ins LibreYOLO-Format:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

[Orientierte Erkennung](/docs/tasks/oriented-detection) behandelt das Label-Format und die Metriken.

## RT-DETR mit eigenen Daten nachtrainieren

Das Training beginnt mit einem veröffentlichten Checkpoint:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Gib bei einem echten Lauf den YAML-Pfad deines eigenen Datensatzes an `data` weiter. Jede Version hat einen eigenen Standardwert für die Lernrate, und bei Version 1 und 2 ist die Backbone-Lernrate gemäß dem Originalrezept ein Zwanzigstel von `lr0`. Multi-GPU-Training gibst du auf der CLI als `device=0,1` an. Das Fine-Tuning mit [LoRA](/docs/train/lora)-Adaptern aktivierst du mit `lora=True`, wenn das `lora`-Extra installiert ist. Die Modelle für orientierte Boxen bei RT-DETRv2 unterstützen nur die Inferenz. Unter [Training](/docs/train) findest du Informationen zu Datensätzen, Augmentierung und Loggern.

## Validierung und Export

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` gibt ein gewöhnliches Dictionary zurück. Zu den Exportzielen zählen ONNX, TorchScript, ExecuTorch, TensorRT und OpenVINO. Die Exportmatrix auf der [Modellseite](/docs/models/rt-detr) zeigt, welche Ziele für jede Aufgabe validiert wurden. Eine exportierte `.onnx`- oder `.engine`-Datei lässt sich wieder mit `LibreYOLO()` laden und gibt dasselbe `Results`-Objekt zurück. Unter [Export](/docs/export) findest du die Anleitungen zu den einzelnen Formaten.

## Wann die Original-Repositories weiterhin die richtige Wahl sind

Wenn du ein Paper-Ergebnis mit den exakten Konfigurationen der Autoren reproduzieren möchtest, die Paddle-Version brauchst oder die DINOv3-Teacher-Destillation von RT-DETRv4 selbst ausführen willst, verwende die Upstream-Repositories. LibreYOLO trainiert den veröffentlichten v4-Student nach, führt aber den Teacher nicht aus. Wenn dein Projekt bereits unter AGPL-3.0 steht oder durch eine Ultralytics Enterprise License abgedeckt ist, gibt es keinen lizenzbedingten Grund für einen Wechsel.

## Hinweis zur Lizenz

Der LibreYOLO-Code steht unter MIT. Jede RT-DETR-Gewichtsdatei steht unter Apache-2.0, und jedes Hugging-Face-Gewichtsrepository enthält die Upstream-LICENSE und NOTICE, die Apache-2.0 bei der Weitergabe der Gewichte beizubehalten verlangt. Die Gewichte, die du mit deinen eigenen Daten trainierst, gehören dir. RT-DETRv4 verwendet DINOv3 nur während des Trainings als Teacher, und die veröffentlichten Student-Modelle enthalten keine DINOv3-Parameter. Das sind allgemeine Informationen, keine Rechtsberatung. Prüfe vor der Veröffentlichung die aktuellen LICENSE-Dateien.

## Ausprobieren

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO steht unter der MIT-Lizenz, läuft unter Linux, Mac und Windows und funktioniert ohne Codeänderung auf GPU, Apple Silicon und reiner CPU.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [RT-DETR-Doku](/docs/models/rt-detr)
