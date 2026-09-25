---
title: "LibreYOLO auf der CVPR 2026: Edge AI mit Hailo-8L und Snapdragon"
description: "Bei der CVPR 2026 in Denver nutzte ein Team von Jabra und der IT-Universität Kopenhagen LibreYOLOXs als Beispielmodell im Tutorial „Edge AI in Action“ und führte es auf einem Hailo-8L sowie auf Snapdragon aus."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "Wie schnell läuft LibreYOLOXs auf Edge-Hardware?"
    a: "Im CVPR-2026-Tutorial lief LibreYOLOXs mit 19.0 ms pro Frame (52.6 FPS) auf einem Raspberry Pi 5 mit Hailo-8L und nach der INT8-Quantisierung mit 6.69 ms auf dem HTP/DSP eines Qualcomm QCS6490."
  - q: "Wie bringe ich ein LibreYOLO-Modell auf einem Edge-Beschleuniger zum Einsatz?"
    a: "Exportiere das Modell mit dem Export-Aufruf von LibreYOLO nach ONNX und kompiliere es dann für das Zielgerät: Der Hailo Dataflow Compiler erzeugt eine HEF-Datei für Hailo-Chips, und der SNPE-/QAIRT-/AI-Hub-Stack deckt Qualcomm Snapdragon ab. Genau diese Pipeline führte das CVPR-Tutorial durchgängig vor."
---

![CVPR 2026, Denver, Colorado, June 3-7](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO war auf der CVPR 2026 dabei.

Die CVPR gilt weithin als die renommierteste Computervision-Konferenz der Welt. Dieses Jahr fand sie in Denver statt. Ein Team von Jabra und der IT-Universität Kopenhagen veranstaltete dort ein Tutorial mit dem Titel „Edge AI in Action: Mastering On-Device Inference“. Als Beispielmodell, um Objekterkennung auf Edge-Chips zu bringen, wählte das Team LibreYOLOXs.

## Was sie entwickelt haben

Das Tutorial führte die gesamte Edge-Pipeline von Anfang bis Ende vor. Zuerst ein LibreYOLOXs-Modell. Dann der Export nach ONNX. Anschließend wurde das ONNX-Modell für zwei sehr unterschiedliche Hardware-Plattformen kompiliert: einen Hailo-8L-Beschleuniger und Qualcomm Snapdragon.

## Echtzeit auf einem Hailo-8L

Das Team kompilierte das ONNX-Modell mit dem Hailo Dataflow Compiler zu einer HEF-Datei und führte es anschließend auf einem Raspberry Pi 5 mit dem Hailo-8L AI HAT aus.

![LibreYOLOXs erkennt auf einem Raspberry Pi 5 mit Hailo-8L Personen und Handtaschen auf einer belebten Straße](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Auf einem Raspberry Pi lief es mit 19.0 ms pro Frame und 52.6 FPS.

## Echtzeit auf Snapdragon

Auf der Qualcomm-Seite quantisierte das Team LibreYOLOXs auf INT8 und führte es über den SNPE-, QAIRT- und AI-Hub-Stack aus. Der Benchmark wurde auf einem QCS6490 durchgeführt.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs erkennt live in der EdgeVision-AI-App auf einem Snapdragon-Smartphone einen Fernseher, Laptop, eine Tastatur, eine Maus und ein Mobiltelefon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

Auf dem HTP/DSP erreichte es 6.69 ms.

## Danke

Vielen Dank an Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera und Shan Ahmed Shaffi, dass sie das Projekt vorgestellt haben.

- Tutorial und Folien: [Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Qualcomm-quantisierte LibreYOLOXs-Modelle: [auf Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Ausprobieren

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO steht unter der MIT-Lizenz, läuft unter Linux, Mac und Windows und funktioniert ohne Codeänderung auf GPU, Apple Silicon und reiner CPU. Eine API umfasst YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, Segmentierung, Pose, Tiefe und mehr.

Gib dem Projekt einen Stern auf GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Doku: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
