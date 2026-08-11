---
title: libreyolo val
seo_title: Befehlsreferenz libreyolo val
description: >-
  Einen Checkpoint von der Kommandozeile aus auf einem Dataset-Split auswerten:
  jedes Argument mit seinem Standardwert und die Metrik-Keys, die jede Aufgabe
  zurückgibt.
lead: >-
  Wertet ein Modell gegen einen Dataset-Split aus und gibt die Metriken aus. Der
  Metriksatz hängt von der Aufgabe des Modells ab, und die Zahlen sind
  dieselben, aus denen eine Benchmark-Zeile entsteht.
keywords:
  - libreyolo val cli
  - libreyolo modell validieren
  - yolo modell auf datensatz evaluieren
  - mAP50-95 kommandozeile
  - libreyolo val argumente
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo val
    mono: true
  - label: Erforderlich
    value: 'model, data'
    mono: true
  - label: Ausgabe
    value: >-
      Metriken auf stdout. Plots und COCO-JSON unter runs/val/exp, wenn
      angefordert
snippets:
  examples:
    - label: Einfach
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Plots und COCO-JSON
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: Maschinenlesbar
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
source_hash: f6507840568c3725
---

## Synopsis

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

Argumente sind `key=value`-Paare, und die POSIX-Form funktioniert ebenfalls,
`batch=8` und `--batch 8` sind also dasselbe Argument.

## Argumente

| Argument | Standard | Bedeutung |
|---|---|---|
| `model` | | Pfad zur Gewichtsdatei oder CLI-Name. Erforderlich |
| `data` | | Pfad zum Dataset-YAML (YOLO-Format, z. B. `coco8.yaml`). Erforderlich |
| `data_dir` | | Direktes Dataset-Verzeichnis, umgeht den Pfad im YAML |
| `split` | `val` | Dataset-Split: `val`, `test`, `train` |
| `batch` | `16` | Batch-Größe |
| `imgsz` | | Bildgröße: `640` (quadratisch) oder `480x640` (Höhe x Breite). Ohne Angabe die eigene Eingabegröße des Modells |
| `conf` | `0.001` | Confidence-Schwellenwert |
| `iou` | `0.6` | NMS-IoU-Schwellenwert |
| `max_det` | `300` | Maximale Anzahl Vorhersagen pro Bild nach der NMS |
| `eval_max_det` | | Obergrenze des COCO-Evaluators. Ohne Angabe die AP@100-Konvention von pycocotools |
| `faster_coco_eval` | `true` | Nutzt das C++-Backend faster-coco-eval für die COCO-Metriken, wenn installiert; sonst fällt es auf pycocotools zurück |
| `half` | `false` | Inferenz in FP16 |
| `amp_dtype` | `float16` | Dtype des CUDA-Autocast bei `half=true`: `float16` oder `bfloat16` |
| `save_json` | `false` | Speichert die Ergebnisse als JSON im COCO-Format |
| `save_plots` | `false` | Speichert die Validierungs-Plots: Metriken, AP pro Klasse, Confusion Matrix, Beispiele |
| `workers` | `4` | Worker des Dataloaders |
| `device` | `auto` | Gerät |
| `project` | `runs/val` | Wurzel des Ausgabeverzeichnisses |
| `name` | `exp` | Name des Experiments |
| `exist_ok` | `false` | Verwendet das Ausgabeverzeichnis wieder |
| `allow_download_scripts` | `false` | Erlaubt eingebettetes Python in den Download-Blöcken des Dataset-YAML |
| `json` | `false` | JSON-Ausgabe auf stdout |
| `quiet` | `false` | Unterdrückt stderr |
| `verbose` | `true` | Ausführliche Ausgabe |
| `help_json` | `false` | Gibt das Schema des Befehls als JSON aus und beendet sich |

## Beispiele

<code-tabs name="examples" />

## Hinweise

### Was die Metriken sind

Der ausgegebene Satz richtet sich nach der Aufgabe des Modells, und die
JSON-Ausgabe nutzt dieselben Keys.

Erkennung, Segmentierung und orientierte Boxen melden `mAP50`, `mAP50_95`,
`precision` und `recall`. Wenn ein Modell mehr als eine Art von Ausgabe
vorhersagt, stehen die Gruppen pro Art daneben als `box_metrics`,
`mask_metrics` und `obb_metrics`, jede mit denselben vier Keys.

Die Klassifikation meldet `accuracy_top1` und `accuracy_top5`. Die
Punkterkennung meldet `precision`, `recall`, `f1`, `MLE`, `MAE`, `RMSE` und
`mAP_sweep`. Die Tiefenschätzung meldet `abs_rel`, `rmse`, `delta1`, `delta2`
und `delta3`. Die semantische Segmentierung meldet `mIoU` und
`pixel_accuracy`. Die Restaurierung meldet `PSNR` und `SSIM`.

Das JSON-Ergebnis führt außerdem `eval_backend` mit, das die
COCO-Evaluationsbibliothek und deren Version nennt, aus der die Zahlen stammen.
So lassen sich zwei Läufe im Wissen darüber vergleichen, ob dasselbe Backend
beide bewertet hat.

### Schwellenwerte

Die Standardwerte hier sind Werte für die Auswertung, nicht für die Vorhersage:
`conf` ist `0.001` und `iou` ist `0.6`, während
[`libreyolo predict`](/docs/cli/predict) `0.25` und `0.45` nutzt. `conf` auf
einen Anzeige-Schwellenwert anzuheben senkt den Recall und mit ihm die mAP, eine
so erzeugte Zahl ist also nicht mit einer veröffentlichten vergleichbar.

`imgsz` ist standardmäßig nicht gesetzt, was die eigene Eingabegröße des Modells
bedeutet. Setzt du es, wird bei der angegebenen Größe ausgewertet, und genau so
wird ein Checkpoint abseits seiner nativen Auflösung vermessen.

### Datensätze, die heruntergeladen werden

Ein Dataset-YAML, dessen Feld `download` eine URL ist, lädt beim ersten
Gebrauch ohne zusätzliche Erlaubnis herunter. Eines, das ein eingebettetes
Python-Download-Skript mitbringt, braucht `allow_download_scripts=true`, und der
Befehl warnt auf stderr, dass die Ausführung lokalen Codes aktiviert wurde. Die
mitgelieferten `coco8.yaml` und `coco128.yaml` sind URL-basiert und brauchen
daher nichts davon.

### Ausgabe und Exit-Codes

stdout trägt die Metriken, der Fortschritt geht auf stderr. `json=true` gibt ein
Objekt mit `schema_version` aus, und `quiet=true` legt stderr still.

Der Exit-Code ist `0` bei Erfolg, `2` bei einem Fehler in Aufruf oder
Konfiguration, `3`, wenn der Datensatz nicht gefunden wird, `4`, wenn das Modell
nicht geladen werden kann, und `1` bei anderen Laufzeitfehlern.

Verwandt: [`libreyolo train`](/docs/cli/train), das dieselbe Auswertung nach
eigenem Zeitplan über `eval_interval` ausführt.
