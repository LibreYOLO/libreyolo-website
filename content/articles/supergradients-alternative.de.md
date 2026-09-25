---
title: "SuperGradients-Alternative: YOLO-NAS weiter nutzen, trainieren und exportieren"
description: "SuperGradients wurde nach der Übernahme von Deci durch NVIDIA im Jahr 2024 nicht mehr weiterentwickelt. LibreYOLO ist eine gepflegte Alternative, die dieselben YOLO-NAS-Gewichte ausführt: vorhersagen, trainieren und exportieren mit einer MIT-lizenzierten API."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "Wird SuperGradients noch gepflegt?"
    a: "Nein. Das letzte Release war 3.7.1 im April 2024, kurz bevor NVIDIA Deci übernahm. Seitdem gab es keine Releases mehr, Fragen in Issues bleiben unbeantwortet und die Dokumentationswebsite ist offline. Das Repository wurde nicht formell archiviert, aber niemand kümmert sich mehr darum."
  - q: "Was ist die beste Alternative zu SuperGradients?"
    a: "Wenn du YOLO-NAS ausführen möchtest, lädt LibreYOLO dieselben Deci-Gewichte über eine gepflegte, MIT-lizenzierte API und ergänzt Training, Validierung und Export. Wenn du für andere Architekturen auf die Trainingsrezepte von SuperGradients angewiesen bist, läuft das alte Repository weiterhin, sobald du seine Abhängigkeiten festschreibst."
  - q: "Kann ich YOLO-NAS kommerziell nutzen?"
    a: "Die vortrainierten Gewichte von Deci sind nicht kommerziell nutzbar, unabhängig davon, welche Bibliothek sie lädt. Die Architektur selbst ist permissiv lizenziert. Wenn du YOLO-NAS von Grund auf neu mit deinen eigenen Daten trainierst, erhältst du ein Modell, in dessen Vorgeschichte sich kein Deci-Checkpoint befindet."
  - q: "Ersetzt LibreYOLO alles, was SuperGradients konnte?"
    a: "Nicht alles. LibreYOLO bietet keinen CoreML-Export für YOLO-NAS, und sein TensorRT-Pfad für diese Modellfamilie ist nicht getestet. Auch für die quantisierungsbewussten Trainingsrezepte von SuperGradients gibt es kein direktes Äquivalent. Für diese Fälle solltest du das alte Repository behalten."
---

SuperGradients war Decis quelloffene Trainingsbibliothek und die Heimat von YOLO-NAS, einem der genauesten Echtzeitdetektoren, die je veröffentlicht wurden. Im Mai 2024 übernahm NVIDIA Deci, und SuperGradients verstummte. Das letzte Release, 3.7.1, erschien am 8. April 2024. Die Dokumentation unter supergradients.com ging offline. Rund 120 Issues sind offen, und die Frage mit dem Titel ["Is this project dead?"](https://github.com/Deci-AI/super-gradients/issues/2062) blieb unbeantwortet von einem Maintainer. Auch das ist eine Antwort.

Das Repository ist nicht archiviert und wirkt auf den ersten Blick noch aktiv. In der Praxis merkst du die Aufgabe sofort bei der Installation: `super-gradients` pinnt `torchmetrics==0.8`, was mit aktuellen PyTorch-Stacks kollidiert, und zieht nebenbei hydra, omegaconf, boto3 und tensorboard mit hinein. Mit jedem Monat geraten die Versionsbindungen stärker mit dem Rest deiner Umgebung in Konflikt, und eine Lösung ist nicht in Sicht.

Das macht YOLO-NAS nicht zu einem schlechteren Modell. Die große Variante erreicht weiterhin 52.2 mAP auf COCO bei Echtzeitgeschwindigkeit. Das Modell ist in Ordnung, sein Zuhause ist abgebrannt.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS accuracy vs parameters - visionanalysis.org">
</iframe>

## Dieselben Gewichte in LibreYOLO ausführen

LibreYOLO lädt die YOLO-NAS-Checkpoints direkt von Decis CDN über dieselbe API wie alle anderen Modellfamilien. Du musst keine Hydra-Konfigurationen schreiben und keinen Prediction-Wrapper entfernen:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Die Detection-Varianten S, M und L funktionieren alle, ebenso Pose: Ersetze den Namen durch `LibreYOLONASs-pose.pt`, und du erhältst COCO-Keypoints zurück. Da jede Modellfamilie dasselbe `Results`-Objekt zurückgibt, ist der Vergleich von YOLO-NAS mit RF-DETR oder D-FINE auf deinen eigenen Daten eine Änderung in einer Zeile statt einer zweiten Codebasis.

## Training und Export, nicht nur Inferenz

SuperGradients war in erster Linie eine Trainingsbibliothek, daher muss eine echte Alternative auch trainieren können. LibreYOLO führt mit demselben Aufruf, den es überall sonst nutzt, Fine-Tuning von YOLO-NAS auf deinem Datensatz durch:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

Du kannst auch mit einem zufällig initialisierten Modell trainieren, was für die Lizenzierung wichtig ist (mehr dazu weiter unten). Die Validierung liefert mAP-Metriken für jeden Datensatz in deinem Format, und der Export unterstützt ONNX, TorchScript, OpenVINO, NCNN und TFLite. Das ist eine breitere Exportmatrix, als die ursprüngliche Bibliothek je für YOLO-NAS ausgeliefert hat. Alle Details findest du auf der [YOLO-NAS-Dokumentationsseite](https://www.libreyolo.com/docs/models/yolo-nas). Eine kürzere Notiz erklärt, dass [YOLO-NAS weiterhin gepflegt wird](/articles/yolo-nas-with-libreyolo).

## Wann das alte Repository weiterhin die richtige Wahl ist

Ein Abschnitt der Ehrlichkeit halber. In drei Fällen solltest du SuperGradients installiert lassen:

**CoreML.** LibreYOLO exportiert YOLO-NAS nicht nach CoreML. Wenn du auf Apple-Geräten auslieferst und ein `.mlpackage` brauchst, bietet SuperGradients weiterhin den funktionierenden Pfad.

**TensorRT.** SuperGradients dokumentierte und testete seinen TensorRT-Ablauf für YOLO-NAS, einschließlich aller Eigenheiten bei der Batch-Größe. Die TensorRT-Unterstützung von LibreYOLO für diese Modellfamilie ist nicht getestet.

**Quantisierungsbewusste Trainingsrezepte.** YOLO-NAS wurde für INT8 entwickelt, und SuperGradients lieferte die QAT-Rezepte, durch die das Modell dort glänzte. LibreYOLO exportiert mit INT8- und FP16-Quantisierung, hat aber keine gleichwertigen Trainingsrezepte.

Das Repository läuft weiterhin, wenn du eine Umgebung aus dem Jahr 2024 dafür festschreibst. Du solltest nur nichts Neues auf einem Fundament aufbauen, das niemand mehr pflegt.

## Hinweis zu den Gewichten

Die vortrainierten YOLO-NAS-Gewichte stammen von Deci und wurden unter einer nicht kommerziellen Lizenz veröffentlicht. Diese Lizenz gilt für die Gewichte auch dann, wenn eine andere Bibliothek sie lädt. LibreYOLO hostet und spiegelt sie nicht. Der Download kommt von Decis öffentlichem CDN, und Decis Bedingungen werden vor dem Start angezeigt. Für Forschung und nicht kommerzielle Arbeit kannst du beide Varianten nutzen.

Wenn du YOLO-NAS kommerziell nutzen musst, gibt es jetzt einen klaren Weg: Die Architektur selbst ist permissiv lizenziert. Wenn du sie von Grund auf neu mit deinen eigenen Daten trainierst, entsteht ein Modell, das von keinem Deci-Checkpoint abgeleitet ist. LibreYOLO unterstützt genau das mit `LibreYOLONAS(None, size="s")`.

## Ausprobieren

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO steht unter der MIT-Lizenz, läuft unter Linux, Mac und Windows und funktioniert ohne Codeänderung auf GPU, Apple Silicon und reiner CPU. Eine API deckt YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet und viele weitere Modelle ab, für Objekterkennung, Segmentierung, Pose, Klassifikation, Tiefe und Tracking.

Gib dem Projekt einen Stern auf GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Doku: [libreyolo.com/docs](https://www.libreyolo.com/docs)
