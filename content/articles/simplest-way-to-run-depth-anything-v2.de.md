---
title: "Depth Anything V2 mit LibreYOLO ausführen: die einfache Anleitung"
description: Depth Anything V2 liefert hochmoderne monokulare Tiefenkarten. So führst du es mit LibreYOLO in zwei Zeilen aus.
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "Wie führe ich Depth Anything V2 am einfachsten aus?"
    a: "Mit LibreYOLO reichen zwei Zeilen: Lade LibreDepthAnythingV2l-depth.pt anhand des Namens und rufe predict mit save=True auf. Die Gewichte werden bei der ersten Verwendung automatisch heruntergeladen. Normalisierung, Colormap und Geräteauswahl werden für CUDA, Apple Silicon oder eine normale CPU für dich übernommen."
  - q: "Kann ich Depth Anything V2 kommerziell nutzen?"
    a: "Nur der Small-Encoder steht unter Apache 2.0. Base, Large und Giant stehen unter CC-BY-NC-4.0, die stärksten Checkpoints sind also für die nicht-kommerzielle Nutzung vorgesehen. Die Upstream-Lizenz gilt unabhängig davon, welche Bibliothek die Gewichte lädt."
  - q: "Unterstützt LibreYOLO das Training von Depth Anything V2?"
    a: "Nein. Das Training bleibt im ursprünglichen Repository. LibreYOLO deckt Inferenz, Video und Validierung für die Tiefenaufgabe ab."
---

![Guggenheim Bilbao neben seiner Tiefenkarte](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2 erzeugt derzeit einige der besten verfügbaren monokularen Tiefenkarten.

Wenn du aus der YOLO-Welt kommst, ist das offizielle Repository nicht besonders leicht zugänglich. Um eine einzelne Tiefenkarte zu erzeugen, sind dort einige manuelle Schritte nötig:

* Lade den passenden Checkpoint manuell herunter und lege ihn im richtigen Ordner ab,

* stimme den Encoder auf seine Konfiguration ab (`encoder="vitl"`, `features=256`, `out_channels=...`),

* schreibe den Schritt für Normalisierung und Einfärbung selbst,

* kümmere dich um die Geräteauswahl, damit es auf einem Mac oder einer CPU läuft und nicht nur auf CUDA,

* und lerne eine Inferenz-API kennen, die ganz anders aussieht als der Rest deines Stacks.

Nichts davon ist schwierig. Es sorgt nur für Reibung, die du dir sparen kannst.

LibreYOLO lädt dieselben Depth-Anything-V2-Gewichte und stellt dir die Tiefenaufgabe mit einem einzigen Aufruf von `predict()` bereit. Gib den Modellnamen an, und es wird bei der ersten Verwendung heruntergeladen:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # auto-downloads on first run
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

Das ist schon alles. Wenn du die standardmäßige YOLO-API kennst, kommt dir die Struktur vertraut vor: Lade ein Modell anhand seines Namens, rufe `predict` auf und lass `save=True` die Visualisierung schreiben. Es ist genau derselbe Aufruf wie bei der Objekterkennung, nur enthält das Ergebnis eine Tiefenkarte statt Bounding Boxes. Colormap, Normalisierung und Geräteauswahl werden für dich übernommen. Unter Linux mit CUDA, auf einem Mac mit Apple Silicon oder auf einer normalen CPU läuft es auf dieselbe Weise. Keine Codeänderung nötig.

Mit demselben Aufruf kannst du noch mehr tun: Du kannst die Roh-Tiefenwerte abrufen und direkt damit arbeiten. Das Training von Depth Anything V2 selbst bleibt im ursprünglichen Repository. LibreYOLO deckt Inferenz, Video und Validierung ab.

Derselbe einzeilige Aufruf bei sehr unterschiedlichen Szenen:

![Das Parkour-Beispielbild neben seiner Tiefenkarte: Die vorderen Springer heben sich von den Betonwänden im Hintergrund ab.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![Luftaufnahme der Bucht von La Concha in Donostia neben ihrer Tiefenkarte: Boote und Ufer erscheinen nah, das offene Wasser liegt weiter hinten.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![Der von Säulen gesäumte Innenhof der Casa de Juntas de Gernika neben seiner Tiefenkarte, die die zurückweichende Architektur zeigt.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![Eine Menschenmenge bei einem Nachtfestival auf der Plaza de la Constitucion in Donostia neben ihrer Tiefenkarte: Die nahen Reihen von Menschen heben sich von der beleuchteten Fassade im Hintergrund ab.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

Ein Hinweis zu den Gewichten: LibreYOLO hostet konvertierte Depth-Anything-V2-Checkpoints und lädt sie bei der ersten Verwendung herunter. Du musst also nichts manuell herunterladen. Die Upstream-Lizenz gilt weiterhin: Der Small-Encoder steht unter Apache-2.0, während Base, Large und Giant unter CC-BY-NC-4.0 (nicht-kommerziell) stehen. Die leistungsstarken Checkpoints sind daher für die nicht-kommerzielle Nutzung vorgesehen. Möchtest du offline bleiben oder selbst konvertieren? Das Skript für die einmalige Konvertierung gibt es weiterhin:

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## Ausprobieren

```bash
pip install libreyolo
```

LibreYOLO ist die umfassendste Computer-Vision-Bibliothek, die du mit pip installieren kannst. Eine vertraute API deckt ein wachsendes Angebot modernster Modelle ab: Objekterkennung (RF-DETR, D-FINE, DEIM), Segmentierung, Pose, orientierte Boxen und jetzt auch monokulare Tiefe mit Depth Anything V2, außerdem Training und Validierung für die unterstützten Aufgaben. Die Bibliothek läuft auf allen wichtigen Betriebssystemen, auf GPU oder CPU, und steht vollständig unter MIT-Lizenz. Damit kannst du sie ohne weitere Bedingungen kommerziell einsetzen.

Gib dem Projekt einen Stern auf GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Doku: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
