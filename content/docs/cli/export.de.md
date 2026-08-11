---
title: libreyolo export
seo_title: Befehlsreferenz libreyolo export
description: >-
  Einen Checkpoint in ein Deployment-Format exportieren: jedes Argument mit
  seinem Standardwert, wo das Artefakt landet und welche Kombinationen der
  Befehl ablehnt.
lead: >-
  Konvertiert einen Checkpoint in ein Deployment-Format und schreibt das
  Artefakt nach weights/. Das Format entscheidet, welche der Argumente unten
  gelten.
keywords:
  - libreyolo export cli
  - libreyolo export befehl
  - yolo nach onnx exportieren
  - tensorrt export befehl
  - libreyolo export argumente
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo export
    mono: true
  - label: Erforderlich
    value: model
    mono: true
  - label: Ausgabe
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: Basis
      language: bash
      code: |
        # Schreibt weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS im Graphen
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Das Artefakt ausführen
      language: bash
      code: >
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640


        # Die Factory routet über das Dateisuffix, der Export lädt wie ein
        Checkpoint.

        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## Syntax

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Argumente sind `key=value`-Paare, und die POSIX-Form funktioniert ebenfalls,
`format=onnx` und `--format onnx` sind also dasselbe Argument.

## Argumente

| Argument | Standard | Bedeutung |
|---|---|---|
| `model` | | Modellgewichte `.pt`. Erforderlich |
| `format` | `onnx` | Export-Format: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | RKNN-Zielplattform, derzeit nur `rk3588`. Wird mit jedem anderen Format abgelehnt |
| `imgsz` | | Größe des Eingabebilds: `640` oder `480x640` (HxW). `480,640` wird ebenfalls akzeptiert. Ohne Angabe die eigene Größe des Modells |
| `batch` | `1` | Batch-Größe für den Export |
| `half` | `false` | FP16-Präzision |
| `int8` | `false` | INT8-Quantisierung |
| `dynamic` | `false` | Dynamische Eingabeformen (ONNX) |
| `simplify` | `true` | Vereinfachung des ONNX-Graphen |
| `nms` | `false` | NMS ins Modell einbetten. Nur ONNX und CoreML |
| `conf` | `0.25` | Confidence-Schwellenwert für eingebettetes NMS |
| `iou` | `0.45` | IoU-Schwellenwert für eingebettetes NMS |
| `max_det` | `300` | Maximale Anzahl Detektionen für eingebettetes NMS in ONNX |
| `opset` | | ONNX-Opset-Version. Ohne Angabe automatisch gewählt |
| `data` | | Kalibrierungsdaten für INT8 |
| `fraction` | `1.0` | Anteil der genutzten Kalibrierungsdaten |
| `device` | `auto` | Gerät für das Tracing |
| `allow_download_scripts` | `false` | Eingebettetes Python in den Download-Blöcken der Dataset-YAML erlauben |
| `json` | `false` | JSON-Ausgabe auf stdout |
| `quiet` | `false` | stderr unterdrücken |
| `verbose` | `false` | Ausführliche Export-Logs |
| `verify` | `false` | Den PC-Simulator von RKNN Toolkit2 ausführen und mit ONNX Runtime vergleichen. Nur RKNN |
| `help_json` | `false` | Befehlsschema als JSON ausgeben und beenden |

`engine` ist ein Alias für `tensorrt` und `litert` einer für `tflite`. Beide
werden auf den kanonischen Namen aufgelöst, bevor irgendetwas geschrieben wird,
deshalb melden die JSON-Ausgabe und die Log-Zeile immer `tensorrt` oder
`tflite`.

## Beispiele

<code-tabs name="examples" />

## Hinweise

### Wo die Datei landet

Der Befehl nimmt keinen Ausgabepfad entgegen. Das Artefakt wird nach `weights/`
geschrieben und nach dem Stamm des Quell-Checkpoints plus dem Suffix des Formats
benannt, mit eingefügtem `_fp16` oder `_int8`, wenn eine dieser Präzisionen
angefordert wurde. `LibreYOLO9s.pt`, nach ONNX in FP16 exportiert, wird zu
`weights/LibreYOLO9s_fp16.onnx`. Das JSON-Ergebnis enthält den aufgelösten
`output_path`, die Dateigröße in MB und die Eingabeform als
`[batch, 3, height, width]`.

### Abgelehnte Kombinationen

`nms=true` wird für ONNX und CoreML akzeptiert und für jedes andere Format mit
`nms_unsupported_format` abgelehnt. Bei ONNX schaltet es `dynamic` ab, weil der
eingebettete Graph auf Batch 1 festgelegt ist, und sagt das auf stderr. Bei
CoreML nimmt es `conf` und `iou`, aber nicht `max_det`, deshalb beendet sich ein
vom Standard abweichendes `max_det` zusammen mit `format=coreml nms=true` mit
`config_unsupported`.

`half=true` zusammen mit `int8=true` ist kein Fehler. INT8 gewinnt, `half` wird
verworfen, und eine Warnung geht auf stderr.

`name` und `verify` sind heute RKNN-Optionen. Wird eines von beiden mit einem
anderen Format übergeben, beendet sich der Befehl mit `config_unsupported`,
statt die Option zu ignorieren.

### Welche Formate eine Familie unterstützt

Die Unterstützung gilt pro Familie und pro Aufgabe, nicht global. `libreyolo
formats family=<family> task=<task>` gibt für diese Kombination die Stufe jedes
Formats aus, mit dem Grund und jeder angehängten Einschränkung. Die Argumente
stehen unter [`libreyolo formats`](/docs/cli/utilities).

Manche Formate brauchen eine optionale Installation und manche eine Toolchain.
Eine fehlende Python-Abhängigkeit beendet den Befehl mit `export_dep_missing`;
eine Präzision, die das Format nicht erzeugen kann, mit
`format_precision_unsupported`.

### Den Export ausführen

Exportierte Artefakte werden über dieselbe Modell-Factory geladen wie
Checkpoints, anhand des Dateisuffixes, deshalb funktioniert `libreyolo predict
model=weights/LibreYOLO9s.onnx` ohne weitere Konvertierung. Drei Optionen der
Vorhersage sind die Ausnahme und werden auf Runtime-Backends abgelehnt:
`tiling`, `overlap_ratio` und `output_file_format`.

Zwei Deployment-Ziele haben eigene Seiten:
[NVIDIA DeepStream](/docs/export/deepstream) und
[NVIDIA Jetson](/docs/export/jetson).

### Ausgabe und Exit-Codes

stdout trägt das Ergebnis; der Fortschritt geht auf stderr. Der Exit-Code ist
`0` bei Erfolg, `2` bei einem Nutzungs- oder Konfigurationsfehler, `4`, wenn das
Modell nicht geladen werden kann, `5` bei einem unbekannten Format, einer
fehlenden Export-Abhängigkeit, einer nicht unterstützten Präzision oder einer
abgelehnten Anfrage für eingebettetes NMS, und `1` bei sonstigen
Laufzeitfehlern.

Verwandt: [`libreyolo quantize`](/docs/cli/quantize), das in PyTorch bleibt und
einen Checkpoint statt eines Deployment-Artefakts schreibt.
