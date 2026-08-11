---
title: Quantisierung
seo_title: Ein LibreYOLO-Modell in PyTorch quantisieren
description: >-
  Die PyTorch-Quantisierungs-API von LibreYOLO: neun Rezepte, Kalibrierung
  getrennt von den Trainingsdaten, QAT und QAD, und zwei Deployment-Artefakte.
lead: >-
  Die Quantisierung in LibreYOLO läuft vollständig in PyTorch: model.quantize()
  tauscht die Conv2d- und Linear-Module eines Modells gegen quantisierte
  Äquivalente aus und kalibriert sie. Das Ergebnis behält den normalen Vertrag
  aus predict, val, train und save, ein quantisiertes Modell wird also von
  denselben Validatoren bewertet wie ein Float-Modell.
keywords:
  - libreyolo quantisierung
  - int8 ptq pytorch
  - quantization aware training yolo
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - kalibrierungsdaten quantisierung
  - qdq onnx export
last_verified: 1.5.0
meta:
  - label: Aufruf
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: Befehl
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: Extra
    value: Keines. Die Quantisierung läuft in PyTorch.
  - label: Familien
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: Rezepte
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: Deployment-Artefakte
    value: >-
      export(format="pt") für einen gepackten Checkpoint, export(format="onnx")
      für einen QDQ-INT8-Graphen
    mono: true
verification: >-
  Gelesen aus libreyolo/quant/api.py, libreyolo/models/base/model.py,
  libreyolo/cli/commands/quantize.py und docs/quantization.md im dev-Branch. Die
  Angaben zur Checkpoint-Größe sind die in docs/quantization.md festgehaltenen
  Messwerte.
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Strukturtausch plus Kalibrierung. calib ist ein kleiner Bildsatz OHNE

        # LABELS, nur vorwärts gelesen, um Bereiche und Skalen abzuleiten.

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # dieselben Validatoren wie bei
        Float

        qmodel.save("LibreYOLO9s-int8.pt")     # Checkpoint trägt ein
        Quant-Manifest
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: Argumente
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # data.yaml-Pfad oder eingebauter Name; None überspringt sie
            samples=128,               # maximale Anzahl Kalibrierungsbilder
            batch=8,                   # Batch-Größe der Kalibrierung
            algorithm="auto",          # auto und minmax sind gleich; percentile ist die Alternative
            keep_high_precision=None,  # None nutzt die Regel der Familie
            verbose=True,
        )
  reload:
    - label: Ein quantisierter Checkpoint lädt als solcher zurück
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Das Quant-Manifest baut die quantisierte Struktur und die Skalen
        # wieder auf, bevor die Gewichte geladen werden.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: QAT ist einfach train() auf einem quantisierten Modell
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Ein Finetune, kein Lauf von Grund auf neu: Finetune-Lernraten nutzen.
        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: QAD ergänzt die vorhandenen Distillation-Argumente
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5
        --lr0 1e-4
  export:
    - label: Gepackter PyTorch-Checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")


        # Schreibt LibreYOLO9s-int8-final.pt: gepackte Low-Bit-Gewichte und
        Skalen,

        # fp32-Master entfernt, der nicht quantisierte Rest auf fp16 gecastet.

        qmodel.export(format="pt")


        # remainder="fp32" hält die nicht quantisierten Tensoren exakt.

        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # QuantizeLinear/DequantizeLinear-Paare im Graphen, mit den eigenen
        # kalibrierten oder QAT-trainierten Skalen des Modells.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: 'Zurück zu Float, mit erhaltenen QAT-trainierten Gewichten'
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        qmodel.dequantize()

        # Jeder Float-Exporter greift jetzt, in jeder Präzision, die er kann.
        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## Installation

Die Quantisierung braucht kein Extra. Der Modultausch, der Kalibrierungsdurchlauf
und die simulierte Arithmetik laufen alle in PyTorch, `pip install libreyolo` ist
also die gesamte Voraussetzung. Die Deployment-Artefakte brauchen das, was ihr
eigenes Format braucht, für den ONNX-Weg also `libreyolo[onnx]`.

## Quantisierung

<code-tabs name="quantize" />

`quantize()` verändert das geladene Modell an Ort und Stelle und gibt es zurück.
Gradienten sind nicht beteiligt: Der Tausch installiert quantisierte Module und
der Kalibrierungsdurchlauf läuft nur vorwärts.

Der entstehende Checkpoint ist ein normaler LibreYOLO-Checkpoint mit angehängtem
`quant`-Manifest, er lädt also mit intakter Struktur und intakten Skalen zurück:

<code-tabs name="reload" />

Die Trainer-Checkpoints, die während eines QAT-Laufs geschrieben werden, tragen
das Manifest ebenfalls, `best.pt` aus so einem Lauf ist also selbst ein
quantisierter Checkpoint.

## Rezepte

Unterstützt werden vier Familien: `yolo9`, `rfdetr`, `birefnet` und `feynobg`.

| Rezept | Was es tut | Familien | Kalibrierung |
|---|---|---|---|
| `fp16` | Cast auf halbe Präzision mit einem float32-Ein- und -Ausgabevertrag. Nur Inferenz. | alle vier | keine |
| `bf16` | Cast auf bfloat16, was den Exponentenbereich von float32 behält. Die Lösung, wenn fp16 auf einem Modell im DETR-Stil überläuft. Nur Inferenz. | alle vier | keine |
| `fp8` | E4M3-Gewichte und -Aktivierungen auf `Conv2d` und `Linear`: Gewichtsskalen pro Kanal, kalibrierte Aktivierungsskalen pro Tensor. | alle vier | erforderlich |
| `int8` | W8A8 auf `Conv2d` und `Linear`: symmetrische Gewichte pro Kanal, affine Aktivierungen pro Tensor. | alle vier | erforderlich, oder `calib=None` nur für die Gewichte |
| `w4a16` | Gruppierte symmetrische INT4-Gewichte, Gruppe 128 entlang `in_features`, Float-Aktivierungen, auf `Linear`. | rfdetr, birefnet, feynobg | nicht nötig |
| `w4a8` | Gruppierte INT4-Gewichte plus kalibrierte INT8-Aktivierungen, auf `Linear`. | rfdetr, birefnet, feynobg | erforderlich |
| `nvfp4` | W4A4-NVFP4 auf `Linear`: E2M1-Elemente, Blöcke zu 16 Elementen, FP8-E4M3-Blockskalen, FP32-Tensorskala. Dynamische Aktivierungsskalierung. | rfdetr, birefnet, feynobg | nicht nötig |
| `mxfp4` | OCP-MXFP4 auf `Linear`: E2M1-Elemente, Blöcke zu 32 Elementen, E8M0-Blockskalen als Zweierpotenzen. Dynamische Aktivierungsskalierung. | rfdetr, birefnet, feynobg | nicht nötig |
| `int2` | Nur für die Forschung: gruppierte 2-Bit-Gewichte, Gruppe 64, plus INT8-Aktivierungen, auf `Linear`. Rein nach dem Training ist es unbrauchbar, QAT oder QAD ist also erforderlich. | rfdetr | erforderlich |

Die Rezepte unterhalb von 8 Bit zielen auf `nn.Linear` und werden für `yolo9` mit
Absicht abgelehnt: Diese Beschleunigung läuft auf aktueller Hardware nur über
GEMM, die Faltungen blieben also in höherer Präzision. YOLO9 nutzt `int8` oder
`fp8`. `int2` wird für `birefnet` und `feynobg` abgelehnt, weil diese Familien nur
für die Inferenz gedacht sind, die heilende QAT, auf die das Rezept angewiesen
ist, dort also nicht zur Verfügung steht.

Die Standardwerte pro Familie halten die erste Schicht und die Heads in Float, und
die DFL-Faltung von YOLO9 wird nie quantisiert: Sie ist ein fester
Integral-Erwartungswert-Operator. Überschreibe das mit
`keep_high_precision=("head.",)`, wenn du einen Grund dazu hast.

## Kalibrierungsdaten sind keine Trainingsdaten

`calib=` nimmt ein paar hundert Bilder, liest keine Labels und läuft nur vorwärts,
um Aktivierungsbereiche zu schätzen. `data=` in `train()` und `val()` ist der
gelabelte Datensatz für Gradienten und Metriken. Das sind verschiedene Argumente
mit verschiedenen Zwecken, und der Standard für `calib` ist `coco128.yaml`.

`algorithm="minmax"` behält die absoluten Extremwerte über alle
Kalibrierungs-Batches hinweg und ist das, was `"auto"` wählt. `"percentile"` nutzt
den Mittelwert der Perzentile 0.1 und 99.9 pro Batch; gemessen bricht damit die
Accuracy der DETR-Familie ein, weil die Ausreißer in den Transformer-Aktivierungen
tragend sind. Was die INT8-Empfindlichkeit kleiner Modelle wirklich behebt, ist
die Kalibrierung auf genügend Batches: Mit dem Standard `coco128` landet YOLO9-t
etwa einen mAP-Punkt neben seinem Float-Wert. Der gewählte Algorithmus wird im
Manifest des Checkpoints festgehalten.

## Wiederherstellung der Accuracy

<code-tabs name="train" />

Quantisierte Module behalten fp32-Master-Gewichte und wenden Fake-Quantisierung
mit einem Straight-Through-Estimator an, die Gradienten erreichen also die Master
und die vorhandenen Trainer funktionieren unverändert: EMA, AMP, das Fortsetzen
aus einem Checkpoint und die Distillation-Argumente lassen sich alle kombinieren.

QAT ist ein Finetune eines bereits trainierten Modells. Nutze Finetune-Lernraten
statt der Standardwerte für einen Lauf von Grund auf neu, sonst zerstört schon ein
kurzer Lauf die vortrainierten Gewichte, ganz unabhängig von der Quantisierung.
Die Verfügbarkeit von QAD folgt der Distillation-Unterstützung der Familien, das
heißt heute `yolo9` und `rfdetr`.

Mit `fp16` und `bf16` quantisierte Modelle sind nur für die Inferenz, und der
Trainer lehnt sie mit einem Hinweis auf `amp=True` ab.

## Export

<code-tabs name="export" />

`format="pt"` lässt das Modell auskristallisieren. Gepackte Low-Bit-Gewichte und
Skalen ersetzen die Master, und der nicht quantisierte Rest wird auf fp16
gecastet, sofern nicht `remainder="fp32"` übergeben wird. Die Invariante des
Packens lautet, dass das Entpacken die Simulation Bit für Bit auf dem Gerät
reproduziert, auf dem du finalisiert hast, die finalisierte Datei erreicht also
genau den Wert, den du validiert hast. Gemessen: YOLO9-s int8 geht von 29.5 MB
auf 9.6 MB, RF-DETR-n nvfp4 von 122 MB auf 26 MB. Lädst du so eine Datei,
bekommst du ein einsatzbereites Modell für die Inferenz, und ein `train()` darauf
rekonstruiert die Master automatisch aus den gepackten Gewichten.

`format="onnx"` gilt für `int8`-Modelle und gibt einen QDQ-Graphen aus, der die
eigenen kalibrierten oder QAT-trainierten Skalen des Modells mitführt, den ONNX
Runtime und TensorRT mit echten INT8-Kernels ausführen. Das ist ein anderer Weg
als [`export(format="onnx", int8=True)`](/docs/export/onnx) auf einem
Float-Modell, wo ONNX Runtime die Skalen selbst ableitet.

Die Cast-Rezepte brauchen überhaupt keinen quantisierten Exporter:

<code-tabs name="dequantize" />

## Einschränkungen

Die quantisierte Arithmetik läuft in Simulation, also als Fake-Quantisierung, die
selbst unter AMP in float32-Inseln gerechnet wird. Die Simulation ist numerisch
treu, ein `val()`-Wert auf einem beliebigen Gerät ist also eine echte Aussage über
die quantisierte Arithmetik. Eine Aussage über Geschwindigkeit ist er nicht.

Zwei Ausnahmen laufen nativ. `fp16` und `bf16` sind gewöhnliche Casts.
Finalisierte `fp8`-Module rechnen ihr GEMM direkt auf gepackten E4M3-Gewichten
über `torch._scaled_mm` auf Hardware der Klassen Ada, Hopper und Blackwell, mit
denselben kalibrierten Aktivierungsskalen wie die Simulation; mit
`LIBREYOLO_KERNELS=off` kommt überall wieder genau der simulierte Weg zum Einsatz.

Die Deployment-Abdeckung ist schmaler als die Rezeptliste. Nur `int8` hat hier
eine ONNX-Form, die sich ausrollen lässt; `fp8` und die linearen Rezepte unterhalb
von 8 Bit laufen in PyTorch und kristallisieren über `format="pt"` aus. Ein
ONNX-Export aus ihnen wirft einen Fehler mit genau dieser Anweisung, und ebenso
jedes Nicht-ONNX-Format aus einem `int8`-Modell: Baue nachgelagerte Engines
stattdessen aus dem QDQ-Graphen.

Der Export eines `int8`-Modells, dessen Aktivierungen nie kalibriert wurden,
protokolliert eine Warnung und erzeugt einen Graphen, der nur die
Gewichtsquantisierung mitführt.
