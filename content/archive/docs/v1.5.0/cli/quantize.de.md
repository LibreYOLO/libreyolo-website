---
title: libreyolo quantize
seo_title: Referenz zum Befehl libreyolo quantize
description: >-
  Quantisiere einen Checkpoint in PyTorch über die Kommandozeile: Rezepte,
  Kalibrierungsargumente, Standardwerte und die Familien, die jedes Rezept
  akzeptiert.
lead: >-
  Ersetzt die Float-Module eines Modells durch quantisierte, kalibriert sie auf
  Bildern ohne Labels, wenn das Rezept Statistiken braucht, und speichert das
  Ergebnis als PyTorch-Checkpoint.
keywords:
  - libreyolo quantize cli
  - int8 quantisierung kommandozeile
  - fp8 quantisierung
  - post training quantization yolo
  - libreyolo quantize argumente
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo quantize
    mono: true
  - label: Erforderlich
    value: model
    mono: true
  - label: Ausgabe
    value: 'Der Quellpfad mit -<recipe> vor der Endung, z. B. LibreYOLO9s-int8.pt'
    mono: true
snippets:
  examples:
    - label: Basis
      language: bash
      code: |
        # Kalibriert auf coco128 und schreibt LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: 'Nur Cast, keine Kalibrierung'
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: 'Breitere Kalibrierung, danach nachtrainieren'
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # Training des quantisierten Checkpoints holt die Accuracy zurück.

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## Synopsis

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

Argumente sind `key=value`-Paare, und die POSIX-Form funktioniert ebenfalls,
`recipe=int8` und `--recipe int8` sind also dasselbe Argument.

## Argumente

| Argument | Standard | Bedeutung |
|---|---|---|
| `model` | | Modellgewichte `.pt`. Erforderlich |
| `recipe` | `int8` | Quantisierungsrezept: `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | Kalibrierungsbilder: ein Daten-YAML oder der Name eines eingebauten Datensatzes. Ohne Labels, nur Forward-Pass. `none` überspringt die Kalibrierung |
| `samples` | `128` | Maximale Anzahl Kalibrierungsbilder |
| `batch` | `8` | Batch-Größe der Kalibrierung |
| `algorithm` | `auto` | Schätzung des Aktivierungsbereichs: `auto`, was minmax wählt, oder `minmax`, oder `percentile` |
| `out` | | Pfad des Ausgabe-Checkpoints. Standardmäßig der Quellpfad mit `-<recipe>` vor der Endung |
| `device` | `auto` | Gerät |
| `allow_download_scripts` | `false` | Eingebettetes Python in Download-Blöcken von Datensatz-YAMLs erlauben |
| `json` | `false` | JSON-Ausgabe auf stdout |
| `quiet` | `false` | stderr unterdrücken |
| `help_json` | `false` | Befehlsschema als JSON ausgeben und beenden |

## Beispiele

<code-tabs name="examples" />

## Hinweise

### Welche Familien es akzeptieren

Die Quantisierung deckt vier Familien ab: `yolo9`, `rfdetr`, `birefnet` und
`feynobg`. Jede andere Familie beendet sich mit `quantize_failed`, das die Liste
mitführt.

### Was jedes Rezept verändert

`fp16` und `bf16` sind Casts. Sie ändern nur den dtype, brauchen keine
Kalibrierung, und `calib=none` ist für sie die richtige Einstellung.

`int8` und `fp8` quantisieren `Conv2d`- und `Linear`-Module, weshalb sie zu den
konvolutionalen Familien passen.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4` und `int2` quantisieren nur `nn.Linear` und
zielen damit auf die Transformer-Familien. Eines davon auf `yolo9` anzufordern
wird mit einer Erklärung abgelehnt, statt still ein unquantisiertes Modell zu
erzeugen, denn die Beschleunigung unterhalb von 8 Bit läuft dort nur über GEMM
und die Faltungen blieben in höherer Präzision.

`int8`, `fp8`, `w4a8` und `int2` brauchen Kalibrierungsstatistiken für ihre
Aktivierungen. `int2` braucht danach zusätzlich Training zur Erholung, deshalb
wird es auf `birefnet` und `feynobg` abgelehnt, die keinen Trainer haben.

Jede Familie behält unabhängig vom Rezept einen Satz Module in Float: die ersten
Schichten, die Vorhersage-Heads und bei YOLOv9 die DFL-Faltung, ein fester
Integral-Erwartungswert-Operator, der nicht quantisiert werden darf.

### Kalibrierungsdaten sind keine Trainingsdaten

`calib` zeigt auf einen kleinen Satz Bilder ohne Labels, der nur im
Forward-Pass genutzt wird, um Aktivierungsbereiche abzuleiten. Es wird nicht
dagegen ausgewertet und seine Labels werden nie gelesen. Der Standard
`coco128.yaml` lädt beim ersten Aufruf von einer URL herunter und braucht daher
keine zusätzliche Erlaubnis; ein YAML mit eingebettetem Python-Download-Skript
braucht `allow_download_scripts=true`.

`algorithm=percentile` steht zur Verfügung und kann die Accuracy auf den
Transformer-Familien senken, weshalb `auto` minmax wählt.

### Wiederherstellung der Accuracy

Die Ausgabe ist ein normaler PyTorch-Checkpoint, deshalb nimmt
[`libreyolo train`](/docs/cli/train) sie direkt an. Das Training eines
quantisierten Checkpoints ist Quantization-aware Training; mit
`distill_model=<teacher>` wird daraus Quantization-aware Distillation.

### Ausgabe und Exit-Codes

Das Ergebnis gibt den gespeicherten Pfad aus, das Rezept, den Ausführungsmodus,
ob eine Kalibrierung lief, und die Anzahl der pro Art ausgetauschten Module. Der
Exit-Code ist `0` bei Erfolg, `4`, wenn das Modell nicht geladen werden kann,
`5`, wenn die Quantisierung oder das Speichern fehlschlägt, und `1` bei
sonstigen Laufzeitfehlern.

Verwandt: [`libreyolo export`](/docs/cli/export), das PyTorch verlässt und
stattdessen ein Deployment-Artefakt schreibt.
