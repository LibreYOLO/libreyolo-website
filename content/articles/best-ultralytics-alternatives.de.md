---
title: Die besten Ultralytics-Alternativen 2026
description: "Ein praktischer Leitfaden zu den besten Open-Source-Alternativen zu Ultralytics YOLO im Jahr 2026: ihre Lizenzen, unterstützten Aufgaben und Grenzen beim Deployment sowie die Rolle von LibreYOLO, der umfassendsten YOLO-Bibliothek unter MIT-Lizenz."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Ist Ultralytics YOLO für die kommerzielle Nutzung kostenlos?"
    a: "Nur unter AGPL-3.0, die verlangt, den Quellcode der darauf aufbauenden Anwendung offenzulegen, auch bei über ein Netzwerk angebotenen Diensten. Für die kommerzielle Nutzung mit geschlossenem Quellcode brauchst du die kostenpflichtige kommerzielle Lizenz von Ultralytics. Permissive Alternativen (MIT, Apache-2.0) vermeiden diese Einschränkung."
  - q: "Welche Ultralytics-Alternative hat die wenigsten Lizenzauflagen?"
    a: "Für einen permissiven Stack, den du überall ausrollen kannst: LibreYOLO (MIT-Code), RF-DETR (Apache-2.0) und YOLOX (Apache-2.0). Sie alle vermeiden das Copyleft der AGPL."
  - q: "Was ersetzt YOLO im Jahr 2026?"
    a: "Zunehmend Echtzeit-Detektoren auf Transformer-Basis: RT-DETR, RF-DETR sowie die D-FINE- und DEIM-Familien. Auf leistungsfähiger Hardware erreichen oder übertreffen sie die Accuracy von YOLO bei ähnlicher Latenz. CNNs im YOLO-Stil sind auf kleinen Edge-Geräten weiterhin im Vorteil, auf denen diese Transformer schlecht laufen und kein ausgereifter NCNN- oder CPU-Pfad verfügbar ist."
  - q: "Laufen diese Tools auf einem Raspberry Pi oder einer NPU?"
    a: "Das hängt vom Export ab. CNN-Detektoren wie YOLOX und RTMDet lassen sich nach NCNN exportieren und laufen auf einem Pi; einige davon (YOLOX) können die Hailo-NPU nutzen. Transformer-Detektoren wie RF-DETR können das nicht, sie benötigen eine GPU oder eine leistungsstarke CPU."
---

Offenlegung: LibreYOLO ist unser Projekt und steht an erster Stelle dieser Liste. Alle anderen Tools hier setzen wir in der Praxis ein, und wir nennen auch die Bereiche, in denen sie LibreYOLO überlegen sind.

Die meisten Teams verlassen Ultralytics aus einem von drei Gründen:

- **Lizenz.** Die YOLO-Modelle von Ultralytics stehen unter AGPL-3.0 (YOLOv5 seit 2023 und ab v8). Das Muster ist bekannt: Du entwickelst ein Produkt und lieferst es aus, dann weist eine rechtliche Prüfung oder ein Kunde auf das Modell hin. Plötzlich musst du entweder den Quellcode deiner gesamten Anwendung veröffentlichen oder eine kommerzielle Lizenz kaufen. Das Copyleft der AGPL erstreckt sich auf deinen Code, sobald du ihn verbreitest oder als Dienst anbietest. Das ist ein häufiger Grund, sich nach Alternativen umzusehen.
- **Offenheit.** Ein Anbieter besitzt die Modelle, den Trainingscode und bestimmt die Bedingungen. Manche Teams wollen Gewichte und Code unter permissiven Lizenzen, auf denen sie ohne Rückfrage aufbauen können.
- **Das Modell.** YOLO ist nicht immer die beste Architektur für die jeweilige Aufgabe. Echtzeit-Detektoren auf Transformer-Basis wie RT-DETR, RF-DETR und D-FINE können bei gleicher Latenz genauer sein. Der Haken: Sie liegen in verstreuten Forschungs-Repositories. Wie du sehen wirst, führt LibreYOLO alle drei über eine einzige API aus. Das ist also ein Grund, die Bibliothek zu wechseln, aber nicht, sie zugunsten eines einzelnen Modells aufzugeben.

Jedes der folgenden Tools bewerten wir anhand von drei Kriterien: einer Lizenz, mit der du dein Produkt ausliefern kannst, der Installation und Ausführung mit einem aktuellen PyTorch sowie dem Export auf die Hardware, auf der du tatsächlich ausrollst.

Beim Lesen zeigt sich ein wiederkehrendes Muster: Jede Alternative ist für sich genommen gut, aber mühsam zu nutzen. YOLOX lässt sich kaum installieren, MMDetection ist eine Qual wegen festgelegter Wheel-Versionen, die RT-DETR-Familie besteht aus Forschungscode ohne Support-Kanal, und Detectron2 ist seit 2021 eingefroren. LibreYOLO steht an erster Stelle, weil es die meisten dieser Tools bereits einbindet, gepflegt, unter MIT-Lizenz, mit einem aktuellen Stack und hinter einer vertrauten API. Es ist weniger ein Eintrag in dieser Liste als die Ebene, die sie zusammenführt.

## Der schnelle Überblick

| Tool | Lizenz | Unterstützte Aufgaben | Am besten geeignet für | Edge-Export |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (Code) | Erkennung, Segmentierung, Pose, Klassifikation, Tiefe, Blickrichtung, Tracking | Der MIT-Einstiegspunkt zu über 20 Modellfamilien über eine API | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | Erkennung, Segmentierung, Pose (Vorschau) | Erkennung und Segmentierung in der Produktion | ONNX, TFLite, TensorRT; kein NCNN oder Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | Vortraining, Distillation | Nicht annotierte Daten, Distillation von DINOv2/v3 | entfällt (kein Detektor) |
| **YOLOX** | Apache-2.0 | Erkennung | Ankerfreie Echtzeit-Erkennung | Upstream schwer zu installieren, läuft über LibreYOLO |
| **MMDetection / OneDL-Fork** | Apache-2.0 | Alles (Forschung) | Breite an Architekturen, Reproduktion von Papers | per Export |
| **Detectron2** | Apache-2.0 | Erkennung, Segmentierung, Keypoints | Mask R-CNN und die R-CNN-Familie | manuell |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | Erkennung | Modernste Echtzeit-DETRs | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | Erkennung, Segmentierung, Pose | Kompakte Edge-ViTs, distilliert aus DINOv3 | Forschungsstand |
| **RT-DETR (v1-v4)** | Apache-2.0 | Erkennung | Fortschrittliche Echtzeit-Erkennung | ONNX, TensorRT |

## 1. LibreYOLO

Lizenz: MIT (Code). Am besten geeignet, wenn du die umfassendste YOLO-Bibliothek unter MIT-Lizenz und eine direkt einsetzbare Alternative zu Ultralytics suchst.

**LibreYOLO ist die YOLO-Bibliothek unter MIT-Lizenz, die jedes Modell über eine einzige API ausführt.** Es ist unser Projekt, und nach den übrigen Einträgen dürfte seine Rolle klar sein: Es ist der Hub, über den sich die Alternativen ausführen lassen. LibreYOLO bindet bereits fast alle der oben genannten, mühsamen Repositories ein: YOLOX, RTMDet, RF-DETR, D-FINE, DEIM und die RT-DETR-Familie. So erhältst du die Modelle über eine vertraute, gepflegte API, ohne dich durch Installationsarchäologie oder Lizenzfragen kämpfen zu müssen.

Der Grund für das Projekt geht über Bequemlichkeit hinaus. YOLO begann als offene Forschung: Joseph Redmons Darknet, von v1 bis v3, durfte jeder frei nutzen und darauf aufbauen. Die AGPL-Ära hat das abgeschlossen. LibreYOLO soll diese Arbeit wieder zugänglich machen, so wie die Entwickler es vorgesehen hatten: permissiv lizenziert, von der Community gepflegt und ohne ungefragt Daten nach Hause zu senden. Dazu gibt es zwei Aspekte.

**Erstens: LibreYOLO ist die YOLO-Alternative zu Ultralytics unter MIT-Lizenz.** Die API, die du bereits kennst, bleibt erhalten. Deine Datensätze im YOLO-Format und deine Skripte lassen sich mit wenigen Änderungen übernehmen, doch die Lizenz ist MIT statt AGPL: kein Copyleft, das auf deinen Code übergreift, keine kommerzielle Lizenz, die du kaufen musst, und keine Telemetrie, die wie bei Ultralytics standardmäßig Daten nach Hause sendet. Wenn du wegen der Lizenz hier bist, ist das der entscheidende Punkt. YOLO9 und RF-DETR sind die gründlich getesteten, produktionsreifen Standardmodelle. Der größere Modellkatalog ist neuer und klar als experimentell gekennzeichnet. Wir sagen das lieber offen, als so zu tun, als sei alles langjährig praxiserprobt.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**Zweitens: Der Umfang geht weit über Ultralytics hinaus.** Eine einzige API umfasst über 20 Modellfamilien: CNN-basierte YOLO-Modelle, Transformer-DETRs, ViT-Backbones, SAM und weitere, statt nur das Angebot eines einzelnen Anbieters. Außerdem gibt es mehr als Detektoren:

- **Aufgaben:** Instanz- und semantische Segmentierung, Pose, Klassifikation und orientierte Boxen. Dazu kommen zwei Aufgaben, die Ultralytics überhaupt nicht anbietet: monokulare Tiefenschätzung (Depth Anything V2) und Blickschätzung (L2CS).
- **Die praktische Ebene:** Multi-Objekt-Tracking mit persistenten IDs (ByteTrack und OC-SORT), Video-Inferenz mit Frame-Subsampling und Live-Vorschau, gekachelte Inferenz für Drohnen- und Satellitenbilder, eine Drag-and-drop-Browseroberfläche (`libreyolo ui`) sowie ein `doctor`-Befehl, der deinen Datensatz prüft, bevor du einen Trainingslauf verschwendest.
- **Umfangreiches Training:** Gradient Accumulation, Einfrieren von Schichten, Fortsetzen, Test-Time Augmentation, LoRA/DoRA-Fine-Tuning, Multi-GPU und Protokollierung mit TensorBoard/MLflow/Weights & Biases.
- **Export:** sieben Formate (ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite) mit INT8/FP16-Quantisierung, integriertem NMS und eingebetteten Modellmetadaten.
- **Läuft überall:** Akzeptiert Pfade, URLs, PIL, NumPy, Tensoren oder rohe Bytes und läuft ohne Codeänderungen auf CUDA, Apple Silicon (MPS) oder einer einfachen CPU.

Wenn das Repository eines starken Detektors aufgegeben wurde oder schwer zu installieren ist, führt LibreYOLO dessen Gewichte mit einem aktuellen, gepflegten Stack aus.

## 2. RF-DETR

Lizenz: Apache-2.0 (Nano, Small, Medium, Large). Am besten geeignet für Erkennung und Segmentierung in der Produktion.

RF-DETR von Roboflow, veröffentlicht auf der ICLR 2026, ist das produktionsreifste Modell dieser Liste. Es ist gut entwickelt, stammt von einem Team, das Produkte ausliefert, und ist eine starke erste Wahl, wenn du eine Erkennung oder Segmentierung brauchst, die funktioniert und der du in einem realen System vertrauen kannst. Das Paket `rfdetr` und die Gewichte von Nano bis Large stehen unter Apache-2.0. Für die größeren Gewichte XL und 2XL gilt die PML-Lizenz von Roboflow.

## 3. Lightly

Am besten geeignet für nicht annotierte Daten, selbstüberwachtes Vortraining und die Distillation eines DINO-Backbones.

Lightly ist kein Detektor. Es hilft dir, nicht annotierte Daten nutzbar zu machen, und besteht aus zwei Teilen mit unterschiedlichen Lizenzen.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** Ein Framework aus Komponenten für selbstüberwachtes Lernen: Losses, Heads, Augmentierungen, Memory Banks und Referenzimplementierungen von über zwanzig Methoden (SimCLR, MoCo, BYOL, DINO, DINOv2, MAE und weitere). Die Trainingsschleife schreibst du selbst. Lightly liefert dir alles, um ausgehend von nicht annotierten Bildern von Grund auf eine Repräsentation vorzutrainieren.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (mit kommerziellen und kostenlosen Community-Optionen).** Die schlüsselfertige Variante, die die meisten meinen. Gib ein Foundation Model wie DINOv2 oder DINOv3 und deine nicht annotierten Daten an. Mit wenigen Zeilen distilliert LightlyTrain dieses Backbone ohne Labels in ein kleineres Modell, das du ausrollen kannst (YOLO, RT-DETR, ViT oder ein eigenes Netz). Achte auf die Lizenz: Die AGPL-3.0 bringt dasselbe Copyleft mit sich, wegen dem Teams nach Ultralytics-Alternativen suchen. Lies daher die Bedingungen, wenn du ein Produkt mit geschlossenem Quellcode ausliefern willst, oder nimm die kommerzielle Lizenz.

Greif zu Lightly, wenn dir ein gutes Backbone fehlt, du aber keine Labels hast. Es ergänzt die Detektoren dieser Liste, statt mit ihnen zu konkurrieren. Auch die neuesten Modelle zeigen das: RT-DETRv4 integriert die Distillation eines Vision Foundation Models direkt ins Training, und DEIMv2 baut DINOv3-Backbones direkt in den Detektor ein.

## 4. YOLOX

Lizenz: Apache-2.0. Am besten geeignet für ankerfreie Echtzeit-Erkennung, sobald die Installation gelingt.

Das Upstream-Repository ist seit dem letzten Release v0.3.0 im April 2022 eingefroren, hat mehr als 700 offene Issues und lässt sich nur schwer auf einem Stack von 2026 installieren. Es gibt keine `pyproject.toml`, und in der `requirements.txt` ist `onnx-simplifier==0.4.10` festgelegt. Dafür gibt es nach Python 3.10 keine vorgefertigten Wheels mehr. Auf einem modernen Interpreter wird das Paket daher aus dem Quellcode gebaut und benötigt cmake sowie eine C++-Toolchain. Auch NumPy ist nicht festgelegt, was ABI-Konflikte zwischen NumPy 2.0 und älteren Versionen von pycocotools und torch begünstigt. Die eigentliche Erkennung läuft, sobald du die Abhängigkeitskette aufgelöst hast. Bis dahin zahlst du den Preis der Installation.

Das Modell selbst ist weiterhin gut: ein ankerfreier Echtzeit-Detektor von Megvii, dessen Code und Gewichte unter Apache-2.0 stehen, und eine sinnvolle Baseline, auch wenn neuere Detektoren es inzwischen überholt haben. Ein verlassenes, aber solides Repository wiederzubeleben ist heute eine Aufgabe, die ein Coding-Agent an einem Nachmittag erledigen kann. Der Installationszustand ist daher weniger abschreckend, als er zunächst wirkt. LibreYOLO führt die YOLOX-Gewichte außerdem direkt mit einem aktuellen Stack aus. In beiden Fällen lohnt sich die Architektur. Eine ausführlichere Anleitung findest du [hier](/articles/yolox-with-libreyolo).

## 5. Das MMDetection-Universum

Lizenz: überwiegend Apache-2.0. Am besten geeignet für eine große Architekturauswahl und die Reproduktion von Papers.

Jahrelang beherbergte MMDetection die offizielle Implementierung fast aller Papers zur Objekterkennung: Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN und Hunderte Konfigurationen. 2026 hat OpenMMLab die mm-Serie schrittweise eingestellt. Das letzte Release von [MMDetection](https://github.com/open-mmlab/mmdetection) war v3.3.0 im Januar 2024, Issues bleiben unbeantwortet, und der Stack hängt an der festgelegten `mmcv`-Version. Deren vorgefertigte Wheels hinken den aktuellen PyTorch- und CUDA-Versionen hinterher. Eine frische Installation geht daher oft schief, bis du alles zurückstufst. Wir haben das Problem [hier](/articles/rtmdet-without-mmdetection) beschrieben.

Ein niederländisches Unternehmen, VBTI, hält das Projekt am Leben. Der [OneDL-Fork](https://github.com/VBTI-development/onedl-mmdetection) veröffentlicht den gesamten Stack unter einem `onedl-`-Präfix neu (`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine` und weitere Pakete). Er wurde für aktuelle Versionen von PyTorch 2.x, CUDA und Python 3.10+ neu gebaut und löst damit das Versionsproblem. Er steht unter Apache-2.0 und wird aktiv gepflegt; v3.5.1 erschien im Mai 2026. Wenn du die Architekturauswahl von MMDetection ohne Installationsaufwand nutzen willst, nimm diesen Fork. Das Team ist klein, unterstütze es also mit Beiträgen, wenn du darauf angewiesen bist.

Zwei verwandte Tools:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0) wird nur noch gewartet. Seit v0.6 im Jahr 2021 gab es kein Release mit Tag, aber das Projekt ist zuverlässig und weiterhin die sauberste Quelle für Mask R-CNN und die R-CNN-Familie zur Instanz- und panoptischen Segmentierung.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, aktiv gepflegt) enthält Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN und Keypoint R-CNN. Es gibt kein modernes YOLO oder DETR, und du schreibst deine Trainingsschleife selbst. Dafür ist TorchVision eine Baseline ohne zusätzliche Abhängigkeiten.

Ein Hinweis zur kommerziellen Nutzung: **MMYOLO**, OpenMMLabs zentrale Sammlung von YOLO-Neuimplementierungen, steht unter GPL-3.0 statt Apache-2.0 und ist seit August 2023 eingefroren.

## 6. Die RT-DETR-Familie

Lizenz: durchgehend Apache-2.0. Am besten geeignet für hochmoderne Echtzeit-Erkennung mit Code auf Forschungsniveau.

Die aktuelle Spitze der Echtzeit-Erkennung besteht aus einer Gruppe verwandter Transformer-Detektoren statt aus einem YOLO-Modell. Die meisten stammen von RT-DETR ab, verwenden große Teile desselben Backbone- und Encoder-Codes und stehen fast alle unter Apache-2.0. Der Wechsel zwischen ihnen fällt daher leicht. Die meisten sind auf Erkennung beschränkt. Eine Ausnahme ist EdgeCrafter, das dieselben Distillation-Ideen auf Segmentierung und Pose erweitert. Gemeinsam ist ihnen der Aufwand durch ihre Forschungs-Repositories: Training über Konfigurationsdateien, eine andere Bedienung als bei Ultralytics und kein Support-Kanal. Die Modelle sind leistungsstark und der ONNX- und TensorRT-Export ist üblicherweise erstklassig.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, ICLR 2025 Spotlight). Formuliert die Box-Regression als Verfeinerung einer Verteilung mit Self-Distillation neu. Gute Accuracy im Verhältnis zur Latenz (D-FINE-X erreicht auf einer T4 etwa 55.8 AP bei ungefähr 13 ms), Größen von N bis X und die Integration in Hugging Face Transformers machen es am einfachsten, außerhalb des eigenen Repositories zu laden und per Fine-Tuning anzupassen.
- **[DEIM und DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025) ist ein Trainingsrezept (Dense O2O Matching), das sich mit D-FINE oder RT-DETRv2 kombinieren lässt und die Trainingszeit ungefähr halbiert. DEIMv2 ergänzt DINOv3-Features und reicht bis zu einer Atto-Klasse mit weniger als 1M Parametern für Mobilgeräte. DEIMv2-S ist das erste Modell unter 10M Parametern mit mehr als 50 AP. Es wird aktiv gepflegt und erhielt Updates bis 2026.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). Die neueste Arbeit desselben Labs und der einzige Eintrag hier, der über Erkennung hinausgeht: EdgeCrafter unterstützt Erkennung (ECDet), Instanzsegmentierung (ECSeg) und Pose (ECPose), jeweils in den Größen S/M/L/X und alles unter Apache-2.0. Ein mit DINOv3 vortrainierter großer ViT wird als aufgabenspezifischer Teacher angepasst und in kompakte Student-Backbones für Edge-Geräte distilliert. Die Werte sind gemessen an der Größe stark: ECDet-S erreicht auf COCO allein 51.7 AP mit weniger als 10M Parametern, ECPose-X kommt auf 74.8 AP (vor YOLO26Pose-X mit 71.6), und die Instanzsegmentierung ist mit der von RF-DETR vergleichbar. Das Modell ist brandneu, behandle es daher als Forschungs-Release. Es ist jedoch ein seltenes kompaktes Modell, das alle drei Aufgaben abdeckt.
- **[RT-DETR und RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). Das ursprüngliche Paper „DETRs beat YOLOs on real-time detection“ (CVPR 2024) legte den Grundstein für die Familie: ohne NMS, End-to-End und gut zu exportieren. v2 ist ein Bag-of-Freebies-Update von 2024 und ein direkt einsetzbares Upgrade bei gleicher Latenz. Das ursprüngliche Paddle-Modell und der kanonische PyTorch-Port liegen im selben Repository, außerdem ist das Modell in HF Transformers enthalten.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, WACV 2025 Oral) und **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Peking University und Tsinghua, Ende 2025). v3 unterstützt nur PaddlePaddle. v4 ist derzeit das beste Modell der Familie (X nahe 57 AP), basiert auf PyTorch und distilliert ein Vision Foundation Model in einen leichten Detektor, ohne zusätzliche Inferenzkosten. Das Repository reproduziert außerdem D-FINE, DEIM und RT-DETRv2 durch Änderungen an der Konfiguration.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). Ein schlichter ViT-DETR, der als Transformer-Alternative zu YOLO vorgestellt wird. Es gibt Größen von tiny bis xlarge sowie ONNX- und TensorRT-Export. Das Modell dient auch als Grundlage für das Open-Vocabulary-Modell OVLW-DETR. Die Aktivität hat zuletzt nachgelassen; betrachte es als stabiles Forschungs-Release.

LibreYOLO bindet bereits viele dieser Modelle über eine API ein (D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet und EdgeCrafter). Am gründlichsten getestet sind YOLO9 und RF-DETR. Für Referenzimplementierungen und die neuesten Forschungs-Checkpoints sind die oben verlinkten Quell-Repositories maßgeblich.

## FAQ

**Ist Ultralytics YOLO für die kommerzielle Nutzung kostenlos?**
Nur unter AGPL-3.0, die verlangt, den Quellcode der darauf aufbauenden Anwendung offenzulegen, auch bei über ein Netzwerk angebotenen Diensten. Für die kommerzielle Nutzung mit geschlossenem Quellcode brauchst du die kostenpflichtige kommerzielle Lizenz von Ultralytics. Permissive Alternativen (MIT, Apache-2.0) vermeiden diese Einschränkung.

**Welche Ultralytics-Alternative hat die wenigsten Lizenzauflagen?**
Für einen permissiven Stack, den du überall ausrollen kannst: LibreYOLO (MIT-Code), RF-DETR (Apache-2.0) und YOLOX (Apache-2.0). Sie alle vermeiden das Copyleft der AGPL.

**Was ersetzt YOLO im Jahr 2026?**
Zunehmend Echtzeit-Detektoren auf Transformer-Basis: RT-DETR, RF-DETR sowie die D-FINE- und DEIM-Familien. Auf leistungsfähiger Hardware erreichen oder übertreffen sie die Accuracy von YOLO bei ähnlicher Latenz. CNNs im YOLO-Stil sind auf kleinen Edge-Geräten weiterhin im Vorteil, auf denen diese Transformer schlecht laufen und kein ausgereifter NCNN- oder CPU-Pfad verfügbar ist.

**Laufen diese Tools auf einem Raspberry Pi oder einer NPU?**
Das hängt vom Export ab. CNN-Detektoren wie YOLOX und RTMDet lassen sich nach NCNN exportieren und laufen auf einem Pi; einige davon (YOLOX) können die Hailo-NPU nutzen. Transformer-Detektoren wie RF-DETR können das nicht, sie benötigen eine GPU oder eine leistungsstarke CPU.

## LibreYOLO ausprobieren

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO steht unter MIT-Lizenz, läuft auf Linux, Mac und Windows und funktioniert ohne Codeänderungen auf GPUs, Apple Silicon und einfachen CPUs. Eine einzige API umfasst RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter und weitere Modelle für Erkennung, Segmentierung, Pose, Klassifikation, Tiefe, Blickrichtung und Tracking.

Gib dem Projekt einen Stern auf GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentation: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
