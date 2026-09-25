---
title: "LiteRT vs. TensorFlow Lite: Umbenennung, kein neues Format"
description: "Google hat TensorFlow Lite im September 2024 in LiteRT umbenannt. Gleiche Runtime, gleiche .tflite-Dateien, nichts geht kaputt. Hier erfährst du, warum Google das getan hat, was sich tatsächlich geändert hat und wie du ein LibreYOLO-Modell dafür exportierst."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "Ist TensorFlow Lite veraltet?"
    a: "Der Name wird eingestellt, nicht die Technologie. Die Runtime wird als LiteRT mit denselben APIs weitergeführt, und vorhandene TFLite-Modelle und Code funktionieren weiterhin. Neue Entwicklungen wie die CompiledModel-API und NPU-Beschleunigung erscheinen unter dem Namen LiteRT."
  - q: "Muss ich meine Modelle für LiteRT neu exportieren?"
    a: "Nein. Eine für TensorFlow Lite exportierte .tflite-Datei ist ein LiteRT-Modell. An der Datei hat sich nichts geändert, daher musst du nichts neu exportieren oder migrieren."
---

**LiteRT ist TensorFlow Lite mit einem neuen Namen. Es ist weder eine neue Runtime noch ein neues Dateiformat. Modelle sind weiterhin `.tflite`-Dateien, und jede `.tflite`-Datei, die mit TensorFlow Lite lief, läuft unverändert mit LiteRT.**

Wenn du nach „LiteRT vs TensorFlow Lite“, „Ist TensorFlow Lite veraltet?“ oder „Was ist LiteRT?“ gesucht hast, ist dieser Absatz die Antwort. Der Rest dieser Seite erklärt kurz, warum es zur Umbenennung kam und wie du mit LibreYOLO ein Modell dafür exportierst.

## Warum Google den Namen geändert hat

TensorFlow Lite startete 2017 als Möglichkeit, TensorFlow-Modelle auf Smartphones und Embedded-Boards auszuführen. Im Laufe der Jahre entwickelte es sich weit darüber hinaus: Google baute Konvertierungspfade für PyTorch, JAX und Keras, sodass bis 2024 ein großer Teil der darauf ausgeführten Modelle nie mit TensorFlow in Berührung gekommen war.

Zu diesem Zeitpunkt war der Name aktiv irreführend. PyTorch-Nutzer übersprangen das Tool, weil es nach einer ausschließlich für TensorFlow gedachten Lösung klang. Deshalb [benannte Google es im September 2024 in LiteRT um](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), kurz für „Lite Runtime“. Das ist die ganze Geschichte. Dasselbe Team, derselbe Code, ein Framework-neutraler Name.

## Was sich tatsächlich geändert hat

Sehr wenig, und nichts davon macht etwas kaputt:

* Der Code hat einen neuen Ort: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* Die Doku ist nach [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert) umgezogen.
* Das Python-Paket zum Ausführen von Modellen heißt jetzt `ai-edge-litert`. Das alte Paket `tflite-runtime` ist veraltet; das neue enthält dieselbe Klasse `Interpreter`.
* Seit der Umbenennung erscheinen neue Funktionen unter dem Namen LiteRT: eine einfachere `CompiledModel`-API und GPU/NPU-Beschleunigung, die Google im Januar 2026 für [produktionsreif erklärte](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

Auch die klassische Interpreter-API funktioniert weiterhin wie bisher. Es gibt eine tatsächlich neue Erweiterung, `.litertlm`, aber das ist ein Verpackungsformat ausschließlich für On-Device-LLMs. Für Computer-Vision-Modelle spielt sie keine Rolle. Wenn also in einer Doku „TFLite“ und in einer anderen „LiteRT“ steht, ist dieselbe Datei gemeint.

## Ein LibreYOLO-Modell nach LiteRT exportieren

LibreYOLO hat in v1.3.0 einen TFLite/LiteRT-Exportpfad ergänzt. Dafür brauchst du Python 3.12+ und ein zusätzliches Paket:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # wird beim ersten Aufruf automatisch heruntergeladen
model.export(format="tflite")        # schreibt eine .tflite-Datei
```

Oder über die CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

Die derzeit validierten Pfade sind YOLO9-Detektion sowie RF-DETR-Detektion, -Segmentierung und -Pose-Schätzung. Sie sind alle weiterhin als experimentell gekennzeichnet. Die vollständige Support-Matrix findest du in der [Doku](/docs/v1.3.0).

Um die exportierte Datei auszuführen, verwende auf dem Zielgerät Googles Paket `ai-edge-litert`:

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, hier kommt dein vorverarbeitetes Bild hin
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

Beachte, dass die Eingabe NHWC (Kanäle zuletzt) verwendet: Bei der Konvertierung von ONNX wird das Layout transponiert, was für dieses Format normal ist.

## Was wir in LibreYOLO geändert haben

Nicht viel, weil sich auch nicht viel ändern musste. In der Doku heißt das Format jetzt „TFLite (LiteRT)“, damit beide Namen auffindbar sind, und in der API bleibt `format="tflite"` der Formatname.

## FAQ

**Ist TensorFlow Lite veraltet?**
Der Name wird eingestellt, nicht die Technologie. Die Runtime wird als LiteRT mit denselben APIs weitergeführt, und vorhandene TFLite-Modelle und Code funktionieren weiterhin. Neue Entwicklungen wie die CompiledModel-API und NPU-Beschleunigung erscheinen unter dem Namen LiteRT.

**Muss ich meine Modelle für LiteRT neu exportieren?**
Nein. Eine für TensorFlow Lite exportierte `.tflite`-Datei ist ein LiteRT-Modell. An der Datei hat sich nichts geändert, daher musst du nichts neu exportieren oder migrieren.
