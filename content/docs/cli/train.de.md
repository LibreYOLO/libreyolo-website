---
title: libreyolo train
seo_title: Befehlsreferenz für libreyolo train
description: >-
  Ein Modell von der Kommandozeile aus trainieren: alle 59 Argumente mit ihren
  Defaults, wie die Defaults einer Modellfamilie sie überschreiben und welche
  Argumente eine Familie ignoriert.
lead: >-
  Trainiert ein Modell auf einem Datensatz und schreibt Checkpoints, Metriken
  und Logs in ein Run-Verzeichnis. Jedes Argument unten hat einen Default aus
  der Befehlsdefinition, den die eigene Trainingskonfiguration einer
  Modellfamilie ersetzen kann.
keywords:
  - libreyolo train cli
  - libreyolo training befehl
  - yolo über kommandozeile trainieren
  - libreyolo train argumente
  - libreyolo dry run
  - yolo layer einfrieren
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo train
    mono: true
  - label: Erforderlich
    value: data
    mono: true
  - label: Ausgabe
    value: 'Checkpoints, Metriken und Logs unter runs/train/exp'
snippets:
  examples:
    - label: Basis
      language: bash
      code: >
        # coco8.yaml liegt dem Paket bei und lädt seine 8 Bilder beim ersten
        Mal.

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: Zuerst die aufgelöste Konfiguration prüfen
      language: bash
      code: >
        # Gibt aus, was der Run nutzen würde, inklusive Familien-Defaults, und

        # beendet sich, ohne zu trainieren oder Daten zu laden.

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: Benannter Run mit explizitem Rezept
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## Synopsis

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

Argumente sind `key=value`-Paare, und die POSIX-Form funktioniert ebenfalls,
`epochs=50` und `--epochs 50` sind also dasselbe Argument. Booleans akzeptieren
`true` und `false`: `amp=false` wird zu `--no-amp`, wo das Flag eine negative
Form hat.

## Argumente

### Modell und Daten

| Argument | Default | Bedeutung |
|---|---|---|
| `data` | | Pfad zur Datensatz-YAML (YOLO-Format, etwa `coco8.yaml`). Erforderlich |
| `model` | `yolox-s` | Modellname oder Pfad zu Gewichten |
| `task` | | Setzt den Task explizit: `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | Vortrainierte Gewichte nutzen. `false` baut die Architektur auf und trainiert von Grund auf neu |
| `allow_download_scripts` | `false` | Erlaubt eingebettetes Python in den Download-Blöcken der Datensatz-YAML |

### Trainingsschleife

| Argument | Default | Bedeutung |
|---|---|---|
| `epochs` | `300` | Trainings-Epochen |
| `batch` | `16` | Batch-Größe pro Gerät |
| `imgsz` | `640` | Bildgröße beim Training: `640` (quadratisch) oder `480x640` (HxB) |
| `device` | `auto` | Gerät: `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | Worker für den Dataloader |
| `cache` | `false` | Bilder cachen, um das Laden zu beschleunigen: `ram`, `disk`, `true`, `false` |
| `seed` | `0` | Zufalls-Seed |
| `resume` | | Training fortsetzen: `true` oder ein Pfad zu einem Checkpoint |
| `amp` | `true` | Automatic Mixed Precision |
| `amp_dtype` | `float16` | CUDA-AMP-Dtype: `float16` oder `bfloat16` |
| `cuda_graph` | `false` | Nimmt Forward- und Backward-Pass des Trainings in CUDA-Graphs auf. Nur eine GPU, nur unterstützte Familien; der Rest läuft eager |
| `lora` | `false` | LoRA-Fine-Tuning, für die unter Hinweise gelisteten Transformer-Familien |
| `freeze` | | Schichten einfrieren: eine Anzahl als Integer, eine Liste von Indizes oder Modulnamen |

### Distillation

| Argument | Default | Bedeutung |
|---|---|---|
| `distill_model` | | Teacher: ein Detektor-Checkpoint oder die ID eines Foundation-Teachers wie `dinov2` für die Distillation von Backbone-Features |
| `dis` | | Gewicht des Distillation-Loss. Ohne Angabe der veröffentlichte Default für den Loss-Typ |
| `distill_loss_type` | `mgd` | Feature-Loss für Detektor-Teacher: `mgd`, `cwd`. Foundation-Teacher nutzen immer `feat_mse` |

### Optimizer

| Argument | Default | Bedeutung |
|---|---|---|
| `optimizer` | `sgd` | Optimizer: `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | Initiale Lernrate |
| `momentum` | `0.937` | SGD-Momentum, und der Koeffizient des ersten Moments für die Adam-Optimizer |
| `weight_decay` | `0.0005` | L2-Regularisierung |
| `nesterov` | `true` | Nesterov-Momentum |

### Scheduler

| Argument | Default | Bedeutung |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | Typ des LR-Schedules |
| `warmup_epochs` | `5` | Dauer des Warmups |
| `warmup_lr_start` | `0.0` | Initiale Warmup-LR |
| `min_lr_ratio` | `0.05` | Minimales LR-Verhältnis |
| `lr_drop` | `100` | Epoche des Step-LR-Drops bei RF-DETR |

### Augmentierung

| Argument | Default | Bedeutung |
|---|---|---|
| `mosaic` | `1.0` | Mosaic-Wahrscheinlichkeit |
| `mixup` | `1.0` | Mixup-Wahrscheinlichkeit |
| `hsv_prob` | `1.0` | Wahrscheinlichkeit für HSV-Jitter |
| `flip_prob` | `0.5` | Wahrscheinlichkeit für horizontales Spiegeln |
| `degrees` | `10.0` | Rotationsbereich, plus und minus, in Grad |
| `translate` | `0.1` | Anteil der Verschiebung |
| `shear` | `2.0` | Scherwinkel |
| `mosaic_scale` | `(0.1,2.0)` | Skalierungsbereich für Mosaic |
| `mixup_scale` | `(0.5,1.5)` | Skalierungsbereich für Mixup |
| `no_aug_epochs` | `15` | Augmentierung für die letzten N Epochen abschalten |

### EMA

| Argument | Default | Bedeutung |
|---|---|---|
| `ema` | `true` | Exponential Moving Average |
| `ema_decay` | `0.9998` | EMA-Decay-Faktor |

### Validierung während des Trainings

| Argument | Default | Bedeutung |
|---|---|---|
| `val` | `true` | Während des Trainings validieren |
| `eval_interval` | `10` | Alle N Epochen validieren |
| `max_det` | `300` | Maximale Anzahl Vorhersagen pro Bild nach der Validierungs-NMS |
| `eval_max_det` | | Obergrenze für den COCO-Evaluator. Ohne Angabe die AP@100-Konvention von pycocotools |
| `faster_coco_eval` | `true` | Nutzt das C++-Backend faster-coco-eval für COCO-Metriken, wenn es installiert ist; fällt sonst auf pycocotools zurück |
| `save_plots` | `false` | Finale Validierungs-Plots während des Trainings speichern |
| `patience` | `50` | Patience für das Early Stopping. `0` schaltet es ab |

### Ausgabe

| Argument | Default | Bedeutung |
|---|---|---|
| `project` | `runs/train` | Wurzelverzeichnis der Ausgabe |
| `name` | `exp` | Name des Experiments |
| `exist_ok` | `false` | Bestehendes Ausgabeverzeichnis wiederverwenden |
| `save_period` | `10` | Alle N Epochen einen Checkpoint speichern |
| `log_interval` | `10` | Alle N Batches den Loss loggen |

### Agent-Flags

| Argument | Default | Bedeutung |
|---|---|---|
| `json` | `false` | JSON-Ausgabe auf stdout |
| `quiet` | `false` | stderr unterdrücken |
| `dry_run` | `false` | Konfiguration auflösen und ausgeben, ohne sie auszuführen |
| `help_json` | `false` | Befehlsschema als JSON ausgeben und beenden |

## Beispiele

<code-tabs name="examples" />

## Hinweise

### Die Defaults oben sind nicht immer die genutzten Werte

Jede Modellfamilie bringt ihre eigene Trainingskonfiguration mit, und wo diese
von der Basiskonfiguration abweicht, ersetzt ihr Wert den Default des Befehls
für jedes Argument, das du nicht explizit gesetzt hast. Setzt du das Argument
selbst, gewinnt das immer. `libreyolo cfg` gibt die Basis-Defaults und die
Overrides je Familie aus, und das ist der Weg, um zu sehen, was eine bestimmte
Familie tatsächlich nutzt.

`imgsz` ist das Argument, bei dem das am meisten zählt. Der Default des Befehls
ist `640`, und das ist nicht bei jedem Checkpoint die native Eingabe: die
veröffentlichten RF-DETR-Größen für die Erkennung sind 384, 512, 576 und 704,
und die YOLOX-Checkpoints `n` und `t` liegen bei 416. RF-DETR und DEIMv2 werden
so behandelt, dass `imgsz` nur weitergereicht wird, wenn es explizit gesetzt
wurde, sonst bleibt ihre eigene Größe in Kraft. Andere Familien bekommen den
Wert so, wie er angegeben wurde, und trainieren damit. FOMO ist der strenge
Fall: jede Größe akzeptiert nur ihre native Eingabe (96, 192 und 224), eine
FOMO-Ausführung braucht also ein passend gesetztes `imgsz`, sonst bricht sie mit
einem Fehler ab. RF-DETR verlangt zusätzlich, dass der Wert durch seine
Patch-Größe mal seine Fensteranzahl teilbar ist, und meldet die beiden
nächstgelegenen zulässigen Größen, wenn er es nicht ist.

### Argumente, die eine Familie ignoriert

Nicht jede Familie liest jedes Argument, und bei den Augmentierungs-Argumenten
zeigt sich das am deutlichsten. RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETRv4 und
DINOv2 trainieren über Pass-through-Pipelines ohne Mosaic, ohne Mixup und ohne
affine Verzerrung, `mosaic`, `mixup`, `hsv_prob`, `degrees`, `translate`,
`shear`, `mosaic_scale` und `mixup_scale` laufen dort also ins Leere. EC teilt
sich diese Pipeline, liest aber `hsv_prob`, `degrees` und `translate`, wenn sein
Task Pose ist. Die Klassifikationsfamilien, SegFormer und NAFNet ignorieren
diesen ganzen Satz und `flip_prob` gleich mit, weil ihr Spiegeln mit einer
festen statt einer konfigurierbaren Wahrscheinlichkeit läuft. YOLO-NAS ignoriert
allein `mosaic`, da es stattdessen mit einer immer aktiven affinen
Transformation pro Sample augmentiert. RF-DETR ignoriert über diese Liste hinaus
drei weitere: `optimizer`, `momentum` und `nesterov`.

Eines davon zu setzen ist kein Fehler. Der Run schreibt eine Zeile auf stderr,
die die Familie und die Argumente nennt, die sie ignorieren wird, trainiert dann
und diese Zeile ist die verbindliche Liste für die installierte Version. Sie ist
außerdem das einzige Signal, ein geskripteter Run mit `quiet=true` unterdrückt
die Warnung also zusammen mit allem anderen auf stderr.

`val=false` ist ein verwandter Fall. Es setzt `eval_interval` bei den meisten
Familien auf `0`; RF-DETR kann die Validierung so nicht abschalten und
protokolliert, dass es die Anfrage ignoriert hat.

### Anderes Verhalten, das du kennen solltest

`lora=true` wird von RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 und v4, EC
und ConvNeXt akzeptiert. Jede andere Familie beendet sich mit
`config_unsupported`, statt ohne LoRA zu trainieren.

`pretrained=false` zusammen mit `resume` wird bei den Familien abgelehnt, die
Training von Grund auf neu unterstützen, denn die beiden verlangen
Gegensätzliches.

`mosaic` und `mixup` sind die Schreibweisen der Konfigurationsfelder
`mosaic_prob` und `mixup_prob` auf der Kommandozeile. Bei Familien, deren Mixup
nur auf Mosaic-Samples wirkt, greift `mixup` über null bei `mosaic` auf null
nie, und der Run sagt das auch.

`dry_run=true` löst die Modellreferenz auf, wendet die Defaults der Familie an
und gibt die Konfiguration aus, mit der trainiert würde. Der Datensatz wird
dabei nicht geladen, das ist also der billige Weg, um zu bestätigen, dass ein
Argument den Wert erreicht hat, den du erwartet hast.

stdout trägt das finale Ergebnisobjekt; Fortschritt und Warnungen gehen auf
stderr. Der Exit-Code ist `0` bei Erfolg, `2` bei einem Bedienungs- oder
Konfigurationsfehler, `3` wenn der Datensatz nicht gefunden oder gelesen werden
kann, `4` wenn das Modell nicht geladen werden kann, und `1` bei sonstigen
Laufzeitfehlern.

Verwandt: [`libreyolo doctor`](/docs/cli/doctor), um einen Datensatz zu prüfen,
bevor du dich auf einen Run festlegst, [`libreyolo monitor`](/docs/cli/monitor),
um einen Run im Browser zu verfolgen, [`libreyolo val`](/docs/cli/val), um das
Ergebnis zu messen.
