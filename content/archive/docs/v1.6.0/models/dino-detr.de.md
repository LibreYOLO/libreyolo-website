---
title: DINO-DETR
families:
  - dinodetr
seo_title: 'DINO-DETR: Vorhersage und Export unter Apache-2.0'
description: >-
  Führe DINO-DETR in LibreYOLO für die Objekterkennung aus. Installiere, sage
  vorher, validiere und exportiere drei Größen mit Denoising-Anchors, alle unter
  Apache-2.0.
lead: >-
  DINO-DETR wurde von IDEA Research als DINO veröffentlicht und kombiniert
  kontrastives Denoising-Training mit einer gemischten Query-Auswahl auf Basis
  der Sparse Attention von Deformable DETR. LibreYOLO bietet drei Größen für die
  reine Erkennungsinferenz.
keywords:
  - dino-detr objekterkennung
  - dino detection transformer
  - denoising anchor boxes
  - mixed query selection
  - object detection python
  - idea research dino
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() gibt ein einfaches dict zurück, kein Objekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreDINODETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## Installation

DINO-DETR benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten. Es verwendet denselben reinen PyTorch-Kern für
mehrskalige Deformable Attention wie die Deformable-DETR-Familie von LibreYOLO.

```bash
pip install libreyolo
```

Die Installation von `libreyolo[hub-kernels]` ist optional. Sobald das Paket
`kernels` vorhanden ist, lädt LibreYOLO zur Laufzeit einen kompilierten Kernel
für mehrskalige Deformable Attention vom Hugging Face Hub und verwendet ihn
anstelle des reinen PyTorch-Kerns. Mit `LIBREYOLO_HUB_KERNELS=0` schaltest du
ihn wieder ab.

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. `conf` und `max_det` filtern die Auswahl der Queries. `iou` wird aus
Gründen der API-Parität akzeptiert, hat aber keine Wirkung, weil der Decoder
Mengen vorhersagt und keinen NMS-Schritt verwendet. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

DINO-DETR unterstützt in LibreYOLO nur Inferenz. Upstream wird das Modell mit
kontrastivem Denoising und ungarischer Zuordnung trainiert. Dieses Rezept ist
hier nicht implementiert, daher löst `train()` den Fehler `NotImplementedError`
aus.

## Varianten

Es gibt drei Checkpoints mit derselben Eingabeauflösung. `r50` und `r50s5`
verwenden beide ein ResNet-50-Backbone und unterscheiden sich in der Anzahl
der Feature-Map-Skalen, die den Decoder speisen, nämlich vier gegenüber fünf.
`swinl` ersetzt das Backbone durch Swin-L und erfasst ebenfalls fünf Skalen.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95 zurück. Diese werden auf einem beliebigen Datensatz in
dem Format gemessen, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint
und gibt dasselbe `Results`-Objekt zurück. [Export](/docs/export) führt die
Argumente auf, die von den einzelnen Formaten akzeptiert werden.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Die drei offiziellen Checkpoints stammen aus dem Google-Drive-Release-Ordner
der Autoren und nicht aus einer Hugging-Face-Modellkarte. Das Upstream-Repository
weist auf Repository-Ebene Apache-2.0 aus, hängt den Checkpoints selbst aber
weder eine Lizenzdatei noch Lizenzmetadaten an. Die Grundlage für die
Weitergabe ist daher diese Erklärung auf Repository-Ebene und keine
Checkpoint-spezifische Genehmigung. Jeder LibreYOLO-Spiegel liefert den
unveränderten Apache-2.0-Lizenztext des Upstreams zusammen mit einem erklärenden
Hinweis aus.

</provenance-box>

## Zitieren

<citation-block />
