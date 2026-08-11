---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2: Oberflächennormalen vorhersagen, validieren und exportieren'
description: >-
  Nutze MoGe-2 in LibreYOLO für die dichte Vorhersage von Oberflächennormalen.
  Installiere, sage vorher, validiere und exportiere die offiziellen ViT-S-,
  ViT-B- und ViT-L-Checkpoints.
lead: >-
  MoGe-2 ist ein monokulares Geometriemodell mit einem einzelnen Vorwärtslauf,
  das aus einem RGB-Bild ein dichtes Feld von Oberflächennormalen vorhersagt.
  LibreYOLO unterstützt ausschließlich die Normalenschätzung mit den offiziellen
  Checkpoints ViT-S, ViT-B und ViT-L.
keywords:
  - moge-2
  - oberflächennormalen schätzen
  - monokulare geometrie
  - normalenkarte
  - dichte vorhersage
  - dinov2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # (H, W, 3) float32-Einheitsvektoren
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # Grad
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # Prozent der Pixel
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## Installation

MoGe-2 benötigt kein optionales Zusatzpaket. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung automatisch heruntergeladen.
LibreYOLO ruft die passende Größe direkt aus den offiziellen Checkpoints ab
und speichert sie lokal zwischen.

<code-tabs name="predict" />

MoGe-2 gibt ein dichtes Feld statt einer Menge von Erkennungen zurück.
`result.boxes` ist daher leer, und `conf`, `iou` sowie `max_det` haben keine
Wirkung. `result.normal_map` enthält das Ergebnis: ein Array der Form
`(H, W, 3)` mit Einheitsvektoren im OpenCV-Kamerakoordinatensystem. `+x` zeigt
nach rechts, `+y` nach unten und `+z` in die Szene. Eine zur Kamera gerichtete
Oberfläche hat den Wert `(0, 0, -1)`. Bei einer Bildliste wird pro Bild ein
Vorwärtslauf ausgeführt. Diese Familie besitzt keinen schnellen Pfad für
gestapelte Batches. Unter [Vorhersage](/docs/predict) findest du Quellen,
Streaming und die Verarbeitung von Ergebnissen.

## Varianten

Drei Encoder-Größen werden als getrennte Checkpoints angeboten: ViT-S, ViT-B
und ViT-L, alle mit derselben Eingabeauflösung. Der Benchmark-Testaufbau von
LibreYOLO hat diese Familie nicht gemessen. Es gibt daher keine veröffentlichten
Accuracy-Werte für einen Vergleich. Wähle die Größe passend zu deinem
Rechenbudget.

## Validierung

`val()` misst den Winkelfehler auf einem gepaarten Normalenkarten-Datensatz:
Bilder liegen neben gleichnamigen 16-Bit-Normalen-PNGs. Eine optionale
Gültigkeitsmaske verhindert, dass aufgefüllte und ungültige Pixel gezählt
werden. Die Methode gibt den mittleren und medianen Winkelfehler in Grad sowie
den Prozentsatz der Pixel innerhalb von 11.25, 22.5 und 30 Grad zurück.

<code-tabs name="val" />

## Export

<export-matrix />

Der Normalenexport nutzt einen Runtime-Vertrag mit fester Auflösung und Batch 1:
`dynamic` und ein anderer `batch`-Wert als 1 werden abgelehnt. `imgsz` muss
außerdem durch die Patch-Größe des ViT-Encoders teilbar sein. LibreYOLO prüft
das vor Beginn des Laufs. Ein exportiertes Artefakt wird anhand seiner
Dateiendung wieder über `LibreYOLO()` geladen. Eine `.onnx`-Datei verhält sich
daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

## Lizenzierung

<provenance-box>

LibreYOLO kopiert diese Checkpoints nicht in seine eigene Organisation.
`LibreYOLO("LibreMoGe2s-normal.pt")` lädt die passende Größe bei einer
festgeschriebenen Revision direkt aus den offiziellen Hugging-Face-Repositorys
und prüft die Datei vor der Verwendung anhand einer aufgezeichneten
SHA-256-Prüfsumme.

</provenance-box>

## Zitieren

<citation-block />
