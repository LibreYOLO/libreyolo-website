---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3: Box-basierte Edge-Segmentierung in LibreYOLO'
description: >-
  Nutze PicoSAM3 in LibreYOLO für Box-basierte Regionssegmentierung auf
  Edge-Sensoren. Installiere den Pico-Checkpoint unter Apache-2.0, sage vorher
  und exportiere.
lead: >-
  PicoSAM3 ist ein kompaktes CNN, das aus SAM 2.1 und SAM 3 destilliert wurde
  und für Box-basierte Region-of-Interest-Segmentierung auf Sensoren wie dem
  Sony IMX500 entwickelt wurde. LibreYOLO unterstützt es ausschließlich mit
  Box-Prompts über eine eigene LibreSAM-Factory, getrennt von der
  Detektor-Factory LibreYOLO().
keywords:
  - picosam3
  - segment anything edge
  - edge segmentierung
  - region of interest
  - box prompt
  - in-sensor inference
  - imx500
  - knowledge distillation
last_verified: 1.5.0
snippets:
  predict:
    - label: Box-Prompt
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # PicoSAM3 hat nur die Größe "pico", daher ist kein weiterer Alias
        nötig.

        model = LibreSAM("picosam3")


        # bboxes= ist der einzige unterstützte Prompt: [x1, y1, x2, y2] oder
        eine

        # Boxenliste, eine Maske pro Box. Jede Box wird um 10 % erweitert,
        quadratisch

        # gemacht, auf das Bild begrenzt und vor dem CNN auf 96x96 skaliert.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        print(result.masks.xy)      # Polygon pro Maske

        print(result.boxes.xyxy)    # aus der Maske abgeleitete enge Box
    - label: 'Einmal kodieren, mehrfach prompten'
      language: python
      code: >
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE


        model = LibrePicoSAM3()


        # set_image() speichert das Quellbild zwischen. PicoSAM3 führt pro Box
        einen

        # vollständigen CNN-Vorwärtslauf aus. Dies spart Laden und Dekodieren
        des Bildes,

        # nicht wie bei anderen SAM-Familien einen Encoder-Durchlauf.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(bboxes=[300, 200, 900, 700])

        b = model.predict(bboxes=[100, 100, 400, 400])

        model.reset_image()
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibrePicoSAM3


        model = LibrePicoSAM3()

        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")


        # opset (standardmäßig 13) und dynamic (standardmäßig True, nur
        Batch-Achse)

        # sind die einzigen von dieser Familie akzeptierten Exportargumente.
    - label: Exportierte Datei verwenden
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3 exportiert sein rohes 96x96-ROI-CNN: roi_image ->
        mask_logits.

        # Es gibt hier keine LibreYOLO-Vor-/Nachverarbeitung zur
        Wiederverwendung,

        # weil export() anders als bei einem Detektor-Checkpoint nicht über

        # LibreYOLO() zurückgeführt wird.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## Installation

PicoSAM3 benötigt das Zusatzpaket `sam`. Der eigene Gewichtsdownload von
LibreYOLO verwendet weiterhin die Hugging-Face-Werkzeuge aus `transformers`,
obwohl die Inferenz auf einem nativen CNN ohne `transformers` läuft.

```bash
pip install "libreyolo[sam]"
```

## Vorhersage

`LibreSAM(...)` (oder das familienspezifische `LibrePicoSAM3(...)`) ist ein
separater Einstiegspunkt neben `LibreYOLO(...)`. Er gibt einen Prompt-basierten
Segmentierer statt eines Detektors zurück, weil ein Vorwärtslauf ohne Prompt
hier keine Aussagekraft hat. Für diese Familie gibt es keinen CLI-Befehl
`libreyolo predict`. Verwende die Python-API.

<code-tabs name="predict" />

PicoSAM3 akzeptiert nur `bboxes=`. Die Übergabe von `points=`, `labels=`,
`masks=`, `text=`, `multimask=True` oder das Auslassen einer Box zum
Segmentieren des gesamten Bildes lösen jeweils einen eindeutigen `ValueError`
aus, weil diese Modi im Upstream-Modell nicht existieren. `conf` filtert
anhand der vorhergesagten Maskenqualität (IoU), nicht anhand einer
Erkennungs-Confidence, und muss zwischen `0.0` und `1.0` liegen. Jede Maske
trägt die Klassen-ID `0` mit dem Namen `"object"`. `train()`, `val()` und
`track()` lösen `NotImplementedError` aus. Verwende LibreSAM2 oder LibreSAM3
für Punkt-, Text-, Masken- oder Alles-segmentieren-Prompts. Unter
[Vorhersage](/docs/predict) findest du die Quelltypen.

## Varianten

Es gibt eine Größe, Pico, mit einer festen ROI-Eingabe von 96 px. PicoSAM3
führt pro Box einen vollständigen CNN-Vorwärtslauf aus, statt das gesamte Bild
einmal zu kodieren.

## Export

<export-matrix />

PicoSAM3 ist die einzige Familie im SAM-Tier mit Export: Sie exportiert ihr
rohes 96x96-ROI-CNN nach ONNX, `roi_image -> mask_logits`, ohne eingebaute NMS
oder Maskennachverarbeitung. Die anderen SAM-Familien lösen bei `export()`
`NotImplementedError` aus, weil für ihre Encoder-Decoder-Aufteilung noch kein
Runtime-Exportvertrag definiert ist. Ein exportierter PicoSAM3-Graph wird nicht
wieder über `LibreYOLO()` geladen. Führe ihn direkt mit einer Runtime wie
`onnxruntime` aus und verwende dieselbe quadratische ROI-Vorverarbeitung mit
10-%-Rand wie oben.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>

PicoSAM3 wurde aus SAM 2.1 und SAM 3 als Lehrermodelle destilliert. LibreYOLO
übernimmt oder verteilt in dieser Familie weder Code noch Gewichte dieser
Lehrer. Nur das kompakte Schüler-CNN und sein konvertierter Checkpoint werden
ausgeliefert.

</provenance-box>

## Zitieren

<citation-block />
