---
title: ONNX
seo_title: Export nach ONNX aus LibreYOLO
description: >-
  Exportiere ein LibreYOLO-Modell nach ONNX: das Opset, das LibreYOLO pro
  Familie wählt, dynamische Achsen, eingebettetes NMS, INT8 und wie der Graph
  zurückgeladen wird.
lead: >-
  ONNX ist ein portables Graphformat. LibreYOLO führt das Tracing des Modells
  mit torch.onnx.export durch, vereinfacht den Graphen auf Wunsch und schreibt
  Familie, Aufgabe, Klassennamen und Eingabegröße in die eigenen Metadaten der
  Datei, damit jedes LibreYOLO-Backend das Postprocessing wieder aufbauen kann.
keywords:
  - yolo nach onnx exportieren
  - onnxruntime
  - torch.onnx.export
  - onnx opset
  - dynamische achsen onnx
  - nms in onnx einbetten
  - onnx int8 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="onnx")
    mono: true
  - label: Schreibt
    value: 'Eine .onnx-Datei, die Metadaten im Graphen eingebettet'
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Zurückladen
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: Formen
    value: In Python standardmäßig dynamischer Batch; Ausnahmen pro Aufgabe unten
  - label: Präzision
    value: 'FP32, FP16 (half=True), INT8 (int8=True, YOLO9-Objekterkennung)'
verification: >-
  Gelesen aus libreyolo/export/onnx.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/onnx.py und
  libreyolo/cli/commands/export.py im dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Schreibt weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Argumente
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int, oder (Höhe, Breite)
            batch=1,
            dynamic=True,     # Python-Standard; die CLI nutzt False
            simplify=True,    # onnxsim über den Graphen laufen lassen
            opset=None,       # None wählt 13, oder 17 für DETR-Familien
            half=False,       # FP16-Gewichte und -Aktivierungen
            int8=False,       # QDQ-INT8, nur YOLO9-Objekterkennung
            data=None,        # Kalibrierungs-data.yaml, nur INT8
            device=None,      # Trace-Device; None nutzt das des Modells
            output_path=None, # None schreibt weights/<stem>.onnx
        )
  nms:
    - label: NMS in den Graphen einbetten
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Nur YOLO9-Objekterkennung, Batch 1. dynamic wird auf False gezwungen.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 mit Kalibrierungsdaten
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # ein paar hundert repräsentative Bilder
            fraction=1.0,
        )
  run:
    - label: Über LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Reine ONNX Runtime
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # Preprocessing und Postprocessing liegen auf diesem Weg bei dir.

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # Der Graph trägt Familie, Aufgabe, Klassennamen und Eingabegröße.

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Eine Familie und Aufgabe vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## Installation

<code-tabs name="install" />

Das Extra zieht `onnx`, `onnxsim` und `onnxruntime` nach. `onnx` allein genügt,
um die Datei zu schreiben; `onnxsim` führt den Vereinfachungsdurchlauf aus und
`onnxruntime` führt das Artefakt aus und übernimmt die INT8-Kalibrierung.

## Export

<code-tabs name="export" />

Ohne `output_path` landet die Datei unter dem Stamm des Checkpoints in
`weights/`, mit angehängtem `_fp16` oder `_int8`, wenn diese Präzision
angefordert wurde.

`dynamic` steht in Python standardmäßig auf `True` und auf der CLI auf `False`.
Ist es an, wird die Batch-Achse symbolisch, und einige Aufgaben öffnen sich noch
weiter: die semantische Segmentierung öffnet zusätzlich Höhe und Breite der
Maske, die Restauration mit Real-ESRGAN öffnet die räumlichen Achsen, und die
zweistufigen Detektoren halten Höhe und Breite der Quelle dynamisch, weil ihre
Skalierung innerhalb des Graphen passiert.

`opset` wird pro Familie gewählt, wenn es weggelassen wird. Die Familien im
DETR-Stil (`detr`, `deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`,
`ec`, `lwdetr`, `rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`) sowie `deit`,
`midas` und `moge2` bekommen Opset 17, denn dort wird
`aten::scaled_dot_product` abgesenkt. Alles andere bekommt 13. Matting wird
unabhängig davon auf 19 angehoben, weil der Decoder von BiRefNet den Operator
`DeformConv` braucht, den ONNX ab Opset 19 definiert.

`simplify=True` führt `onnxsim` aus und behält den ursprünglichen Graphen, wenn
der Durchlauf fehlschlägt, ein Vereinfachungsfehler ist also eine Warnung und
kein Fehlschlag des Exports. Unter macOS arm64 mit `onnx` 1.22 oder neuer und
`onnxsim` 0.6.5 oder älter wird der Durchlauf ganz übersprungen, weil diese
Paarung den Python-Prozess abbrechen kann.

### Eingebettetes NMS

<code-tabs name="nms" />

`nms=True` gilt nur für die YOLO9-Objekterkennung und setzt Batch 1 voraus;
zusammen mit `dynamic=True` angefordert, wird eine Warnung geloggt und dynamic
abgeschaltet. Der Graph hat dann zwei Ausgaben: `output` mit der Form
`(batch, max_det, 6)` und `raw`, den undekodierten Detektor-Tensor, den
LibreYOLOs eigenes Backend nutzt, damit das Postprocessing identisch zum
PyTorch-Weg bleibt.

### DeepStream

`deepstream=True` ist eine Option nur für ONNX. Sie exportiert den Graphen in
dem Layout, das der Parser von NVIDIA DeepStream erwartet, und schreibt zwei
Sidecar-Dateien daneben, `config_infer_primary_<stem>.txt` und
`<stem>_labels.txt`, damit sich das Artefakt ohne handgeschriebene Konfiguration
in eine Pipeline einfügt.

Sie schließt sich mit `nms=True` gegenseitig aus, und beides zusammen
anzufordern löst einen `ValueError` aus: DeepStream führt die Unterdrückung in
seiner eigenen Clustering-Stufe aus. Sie an ein anderes Format als ONNX zu
übergeben, löst ebenfalls einen Fehler aus. Das unterstützte Raster aus Familie
und Aufgabe und den Bau des Parsers zeigt
[DeepStream](/docs/export/deepstream).

### INT8

<code-tabs name="int8" />

`int8=True` führt die statische Quantisierung der ONNX Runtime aus und schreibt
einen QDQ-Graphen mit Float32-Eingaben und -Ausgaben. Quantisiert werden nur
`Conv`- und `Gemm`-Knoten. Das Decoding im Detektions-Head in Float32 zu
belassen, ist Absicht: diese Konkatenation mischt Box-Koordinaten im
Pixelmaßstab mit Klassen-Scores im Bereich 0 bis 1, und eine einzelne
Aktivierungsskala pro Tensor, dominiert von der Größenordnung der Box, würde
jeden Score auf null drücken.

Dieses Flag gilt derzeit nur für die YOLO9-Objekterkennung, und alles andere
löst im Preflight `NotImplementedError` aus. Wird `data` weggelassen, fällt der
Export mit einer Warnung auf `coco8.yaml` zurück; acht Bilder sind kein
repräsentatives Kalibrierungsset. Ein Modell, das bereits in PyTorch quantisiert
wurde, nimmt einen anderen Weg, beschrieben unter
[Quantisierung](/docs/export/quantization).

## Ausführung des Artefakts

<code-tabs name="run" />

`LibreYOLO()` unterscheidet anhand der Endung `.onnx` und liefert dasselbe
`Results`-Objekt wie ein `.pt`-Checkpoint, weil Klassennamen, Aufgabe,
Eingabegröße und Pose-Schema beim Export in die `metadata_props` des Graphen
geschrieben wurden. Mit `device="auto"` nimmt die Session den
`CUDAExecutionProvider`, wenn die ONNX Runtime ihn meldet, und fällt sonst auf
CPU zurück.

Das zweite Snippet ist für Lesende ohne installiertes LibreYOLO. Preprocessing,
Decoding, NMS und die Umrechnung der Koordinaten liegen auf diesem Weg alle bei
dir; der Metadatenblock ist weiterhin da und lässt sich auslesen.

## Einschränkungen

Die Namen der Ausgabetensoren stehen pro Aufgabe fest, und genau sie muss ein
Consumer ohne Metadaten treffen:

| Aufgabe | Ausgabenamen |
|---|---|
| Objekterkennung, Grid- und Anchor-Heads | `output` |
| Objekterkennung, DETR-Stil | `pred_logits`, `pred_boxes` |
| Objekterkennung, RF-DETR | `dets`, `labels` |
| Klassifikation | `output` |
| Semantische Segmentierung | `semantic_logits` |
| Tiefe | `depth` |
| Oberflächennormale | `normal` |
| Kanten | `edges` |
| Restauration | `restored` |
| Matting | `matte` |
| Gaze | `yaw_logits`, `pitch_logits` |

RF-DETR ist zugleich die eine Familie, deren Eingabetensor `input` statt
`images` heißt.

Mehrere Aufgaben tragen in dieser Version einen Runtime-Vertrag mit fester
Auflösung. Tiefe, Oberflächennormale und Kanten lehnen `batch != 1` ab und
erzwingen `dynamic=False`. Matting erzwingt das native Quadrat mit 1024, weil
die Tabellen für relative Positionen im Swin von BiRefNet an ihre Auflösung
gebunden sind. Die Restauration erzwingt eine feste Leinwand für jede Familie
außer Real-ESRGAN, dessen Generator vollständig faltend ist.

Ein rechteckiges `imgsz` funktioniert für die YOLO9-Familien, HRNet, NAFNet und
Real-ESRGAN. Familien mit festem quadratischem Vertrag (`clip`,
`deformable_detr`, `detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`,
`lwdetr`, `moge2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`,
`ssd`) lehnen es rundweg ab.

Zwei Kombinationen werden schon vor dem Tracing verweigert: die
YOLO9-Segmentierung, weil YOLO9 in LibreYOLO nur Objekterkennung kann, und die
RTMDet-Ins-Segmentierung, deren Masken-Decoding mit dynamischem Kernel keinen
Vertrag für die exportierte Runtime hat.

Das vollständige Raster aus Familien und Aufgaben zeigt
[die Export-Matrix](/docs/reference/export-matrix). Für eine einzelne
Kombination frag die Bibliothek direkt:

<code-tabs name="support" />
