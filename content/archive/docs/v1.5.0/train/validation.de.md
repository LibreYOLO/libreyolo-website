---
title: Validierung und Metriken
seo_title: Validierung und Metriken in LibreYOLO
description: >-
  Führe val() mit jedem Modell aus, lies die von den einzelnen Aufgaben
  zurückgegebenen Metrikschlüssel, wähle ein Evaluierungs-Backend und aktiviere
  neben der Accuracy-Metrik einen Validierungs-Loss.
lead: >-
  Die Validierung führt ein Modell mit val() über einen Datensatz-Split aus und
  gibt ein flaches Dictionary aus Metrikschlüsseln und Fließkommawerten zurück.
  Die Schlüssel sind literale Strings. Welche du erhältst, hängt von der Aufgabe
  und nicht von der Familie ab.
keywords:
  - map50-95
  - coco evaluierung
  - validierungsmetriken
  - faster-coco-eval
  - pycocotools
  - validierungs-loss
  - miou
  - panoptic quality
  - top1 accuracy
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Mit einem anderen Split
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Vorhersagen im COCO-Format schreiben
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Ausführen einer Validierung

`val()` nimmt den Datensatz entgegen und gibt die Metriken zurück.

<code-tabs name="val" />

Der Rückgabewert ist ein einfaches `dict[str, float]`. Jeder Schlüssel ist
literal. Lies ihn daher über seinen Namen und nicht über seine Position aus.

Die wichtigsten Argumente sind `data`, `split`, `batch`, `imgsz`, `conf`,
`iou`, `workers`, `device`, `augment`, `save_json` und `verbose`. `conf` hat
den Standardwert `0.001`, `iou` den Standardwert `0.6`. Beide sind erheblich
lockerer als die Standardwerte für Vorhersagen, weil ein mAP-Sweep auch den
Ausläufer mit niedriger Confidence benötigt. Der Standardwert von `imgsz` ist
die eigene Eingabegröße des Modells und keine feste Zahl. `split` akzeptiert
ausschließlich `val`, `test` oder `train`.

Jedes weitere Feld der Validierungskonfiguration wird als Keyword-Argument
weitergereicht. Dazu gehören `save_dir`, `max_det`, `eval_max_det`, `half`,
`amp_dtype`, `cache` und `save_plots`.

## Metrikschlüssel pro Aufgabe

Die Erkennung gibt die COCO-Metrikfamilie zurück:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Zwei davon sind tückisch. `metrics/precision` und `metrics/recall` sind aus
Gründen der Abwärtskompatibilität erhaltene Aliasse. Sie enthalten die Werte
mAP 50-95 und AR@100 und kein Paar aus Precision und Recall. Verwende die
benannten Schlüssel.

Die Instanzsegmentierung gibt die oben genannten mAP- und AR-Werte als
Maskenmetriken unter den Schlüsseln ohne Suffix zurück. Die Boxenversionen
stehen unter einem `(B)`-Suffix und die Maskenversionen werden unter `(M)`
wiederholt. Precision und Recall sind für diese Aufgabe nur mit Suffix
vorhanden: `metrics/precision(B)`/`metrics/recall(B)` und
`metrics/precision(M)`/`metrics/recall(M)`. Beide Paare enthalten dieselben
Aliaswerte wie bei der Erkennung. Das `(B)`-Paar steht für Boxen-mAP50-95 und
Boxen-AR@100, das `(M)`-Paar für Masken-mAP50-95 und Masken-AR@100.

| Aufgabe | Schlüssel |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75` sowie die oben genannten Aufschlüsselungen nach Größe und Recall |
| segment | Maskenversionen der oben genannten Erkennungsschlüssel (Schlüssel ohne Suffix stehen für Masken); `precision`/`recall` gibt es nur als `(B)`/`(M)`, beide mit denselben Aliasregeln |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L` und die entsprechenden `keypoints_AR`-Schlüssel |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall` sowie Kopien mit dem Suffix `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE` sowie ein mAP-Sweep-Schlüssel |

Die Werte `metrics/precision` und `metrics/recall` von OBB sind keine Aliasse.
Sie sind die tatsächliche Precision und der tatsächliche Recall bei IoU 0.50,
gemessen am lockersten Betriebspunkt, also mit jeder Vorhersage, die `conf`
übersteht (Standardwert `0.001`). Die Kopien mit dem Suffix `(OBB)` wiederholen
dieselben vier Werte unter einem aufgabenspezifischen Namen, entsprechend der
obigen Konvention für `(B)` und `(M)`.

`accuracy_top5` bedeutet tatsächlich top-`min(5, num_classes)`. Bei einem
Datensatz mit drei Klassen ist dies also top-3, was jedes Sample erfüllt und
deshalb den Wert 1.0 ergibt.

Der Sweep-Schlüssel der Punktaufgabe wird aus den Distanzschwellenwerten
gebildet. Mit den Standardwerten lautet er `metrics/mAP@[0.01:0.10]`, während
der Schlüssel für einen einzelnen Schwellenwert `metrics/mAP@0.01` lautet.
Durch die Übergabe von `dist_thresholds` ändern sich beide Strings.

Die meisten Aufgaben geben außerdem einen `fitness`-Schlüssel zurück. Dieser
einzelne Wert wird standardmäßig zur Auswahl des besten Checkpoints verwendet.
Erkennung, Segmentierung und OBB besitzen ihn nicht. Ihre Familien werden
anhand von `metrics/mAP50-95` ausgewählt, der in ihren Dictionaries enthalten
ist. Pose gibt weder `fitness` noch `metrics/mAP50-95` zurück. Die zugehörigen
Trainer setzen `best_metric_key` stattdessen auf
`metrics/keypoints_mAP50-95`.

## Geschwindigkeitsschlüssel

Jeder Validator fügt Zeitmessungen hinzu:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Dies sind über den Lauf gemittelte Millisekunden pro Bild. Sie beschreiben den
verwendeten Rechner und die Einstellungen. Ein daraus entnommener Wert ist
daher nur zusammen mit Hardware, Batch-Größe und Präzision aussagekräftig.

## Evaluierungs-Backend

Metriken für Erkennung und Segmentierung werden mit einem COCO-Evaluator
berechnet. Der Standardwert `faster_coco_eval=True` wählt das C++-Backend aus,
wenn das Paket `faster-coco-eval` installiert ist. Andernfalls fällt der Lauf
auf pycocotools zurück und gibt pro Prozess eine Warnung aus:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Das tatsächlich verwendete Backend wird im Modell als `last_eval_backend`
gespeichert. Bei Erkennungsaufgaben meldet es auch die CLI in ihrer Ausgabe.
Setze `LIBREYOLO_FASTER_COCO_EVAL`, um den Konfigurationswert über die Umgebung
zu überschreiben.

`iou_thresholds` wird nur im OBB-Pfad berücksichtigt. Der COCO-Pfad evaluiert
mit einem eigenen festen Sweep von 0.50 bis 0.95 und ignoriert diesen Wert.

## Validierungs-Loss

Standardmäßig meldet die Validierung nur die Accuracy. Mit `val_loss=True`
wird zusätzlich die Trainingszielfunktion der Familie für die
Validierungs-Batches berechnet.

<code-tabs name="valloss" />

Die Ausgabe enthält `metrics/loss` sowie einen Schlüssel
`metrics/loss/<component>` pro Term. Alle werden genau wie im Training
gewichtet, sodass sich die Komponenten zum Gesamtwert addieren. In einem
Logger erscheinen sie als `val/loss` und `val/loss/<component>`, während
`libreyolo monitor` `metrics/loss` über `train/loss` legt.

Die Komponenten gehören jeweils zur Familie:

| Aufgabe | Familien | Komponenten |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

Die Option ist standardmäßig deaktiviert, weil die Target-Zuweisung die Dauer
und den Speicherbedarf der Validierung erhöht. Der Validator verwendet die
bereits für die Accuracy-Metrik erzeugte Modellausgabe, statt einen zweiten
Forward Pass auszuführen. Er läuft mit `no_grad` auf dem Evaluierungs- oder
EMA-Modell. Beim Multi-GPU-Training wird er lokal auf Rang 0 und ohne
Collectives berechnet. Die Auswahl des besten Checkpoints basiert weiterhin
auf der Accuracy-Metrik.

Drei Dinge werden bewusst nicht ausgeführt. Erstens werden keine Terme für
Contrastive Denoising einbezogen, da diese beim Forward Pass die Ground Truth
benötigen, während die Forward Passes der Validierung ohne sie laufen. Zweitens
wird das Modell im Evaluierungsmodus gemeldet. Wenn sich die Forward Passes
einer Familie im Trainings- und Evaluierungsmodus tatsächlich unterscheiden,
etwa durch BatchNorm-Statistiken oder Stochastic Depth, spiegelt der Wert den
Evaluierungsmodus wider. Genau dieser Vergleich ist beabsichtigt. Drittens
löst eine Aufgabe, für die eine Familie diese Funktion nicht implementiert
hat, bei der Einrichtung einen Konfigurationsfehler aus, statt stillschweigend
übersprungen zu werden:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO ist die Ausnahme, bei der sich nichts ändert: Sein Validator hat diesen
Loss schon immer berechnet. `val_loss=True` beeinflusst nur, unter welchen
Schlüsseln er veröffentlicht wird.

Augmentierte Validierung und Validierungs-Loss können nicht kombiniert werden.
Werden beide angefordert, wird ein Fehler ausgelöst.

## Von einer Validierung geschriebene Dateien

`val()` schreibt immer eine `config.yaml` in sein Ausgabeverzeichnis. Wenn
`save_dir` nicht angegeben ist, lautet dessen Standardwert
`runs/val/<model>_<size>_<timestamp>`.

<code-tabs name="json" />

Mit `save_json=True` wird für die Erkennung `predictions.json` geschrieben.
Für die Segmentierung entstehen `predictions_bbox.json` und
`predictions_masks.json`. OBB unterstützt dies nicht und meldet den Umstand.

`save_plots=True` schreibt in ein Unterverzeichnis namens `plots/`. Für die
Erkennung entstehen `box_metrics.png`, AP- und Recall-Diagramme pro Klasse,
Precision-Recall- und Confidence-Kurven, eine Konfusionsmatrix und bei
installiertem OpenCV annotierte Beispielbilder. Die Segmentierung ergänzt
davon jeweils maskenbezogene Kopien, während Pose einen eigenen Satz von
Metriken und Kurven erhält. Die anderen Validatoren implementieren keine
Diagramme. Klassifikation, semantische und panoptische Segmentierung, Tiefe,
Normalen, Kanten, Restaurierung, Matting, OCR, OBB und Punkte schreiben dort
nichts. Ein Fehler beim Plotten löst eine Warnung aus und bricht den Lauf nie
ab.

## Validierung während des Trainings

Das Training validiert alle `eval_interval` Epochen anhand des `val`-Splits
des Datensatzes. Die erzeugten Metriken steuern die Auswahl von `best.pt`, den
vorzeitigen Abbruch mit `patience` und die `val/`-Schlüssel in jedem Logger.
Bei aktiviertem EMA läuft die Validierung auf den EMA-Gewichten.

Unter [Hyperparameter](/docs/train/hyperparameters) findest du Informationen
zu `eval_interval`, `patience` und `save_plots`. Unter
[Experiment-Logger](/docs/train/loggers) erfährst du, wohin die Werte
geschrieben werden.

## Verwandte Themen

- [Datensätze](/docs/train/datasets) beschreibt die Split-Schlüssel und
  Formate, die Validatoren lesen.

