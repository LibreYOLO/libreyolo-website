---
title: Augmentationen
seo_title: Trainingsaugmentationen in LibreYOLO
description: >-
  Die Augmentationseinstellungen in TrainConfig, die vier zugrunde liegenden
  Pipeline-Formen und die familienbezogene Tabelle mit verwendeten, gekoppelten
  oder ignorierten Einstellungen.
lead: >-
  Augmentation wird über Einstellungen in TrainConfig konfiguriert. Jede
  Modellfamilie führt jedoch ihre eigene Trainingspipeline aus. Eine Pipeline
  ohne Mosaic-Zweig ignoriert mosaic_prob, statt es anzunähern.
keywords:
  - YOLO Datenaugmentation
  - Mosaic Augmentation
  - MixUp
  - HSV Jitter
  - zufällige affine Transformation
  - Copy Paste Augmentation
  - RandAugment
  - CutMix
  - no_aug_epochs
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # In der CLI heißt mosaic_prob mosaic und mixup_prob heißt mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  support:
    - label: Unterstützungstabelle einer Familie lesen
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Nur die ignorierten Einstellungen
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Klassifizierungspaket
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 47461cd13aab580c
---

## Festlegen der Einstellungen

Die Augmentationseinstellungen sind gewöhnliche Argumente von `train()`.

<code-tabs name="train" />

Zwei davon besitzen kürzere CLI-Schreibweisen: `mosaic` wird auf `mosaic_prob` und `mixup` auf `mixup_prob` abgebildet. Alle anderen Einstellungen heißen an beiden Stellen gleich.

## Drei statt zwei Zustände

Ob eine Einstellung eine Wirkung hat, hängt von der Familie ab. Die Bibliothek verwaltet dafür eine deklarative Tabelle. Jeder Eintrag besitzt einen von drei Zuständen.

`used` bedeutet, dass die Einstellung die Pipeline erreicht und Samples verändert. `ignored` bedeutet, dass sie die Pipeline nie erreicht und wirkungslos bleibt. `gated_by_mosaic` bedeutet, dass sie nur für Samples gilt, die den Mosaic-Zweig durchlaufen haben. Bei `mosaic_prob=0` wird sie daher trotz vorhandener Verdrahtung nie ausgelöst.

Dieser dritte Zustand überrascht häufig. In einer Pipeline im YOLOX-Stil wird die affine Transformation auf dem Mosaic-Canvas ausgeführt und MixUp mischt ein Mosaic-Sample ein. `mosaic_prob=0` deaktiviert deshalb gleichzeitig und ohne Fehler `degrees`, `translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` und `mixup_scale`. Der Trainer protokolliert speziell für MixUp eine Warnung:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

Die CLI warnt auch vor ignorierten Einstellungen und nennt nur die ausdrücklich angegebenen:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Vier Pipeline-Formen

Die Familien lassen sich vier Trainingspipelines zuordnen, die fast alle Antworten bestimmen.

Die Mosaic-Pipeline im YOLOX-Stil wendet HSV-Jitter und Spiegelungen auf jedes Sample an. Affine Transformation und MixUp werden anschließend innerhalb des Mosaic-Zweigs ausgeführt. Diese Pipeline gilt für YOLOX, YOLOv7, YOLOv9 und dessen E2E- und P2-Varianten, RTMDet, PicoDet, RT-DETR, RT-DETRv2 und FOMO.

Die durchleitende Pipeline im DETR-Stil besitzt weder Mosaic noch eine affine Transformation. Fotometrische Verzerrung, Zoom-out und IoU-Beschnitt sind feste Rezeptwerte und keine Konfigurationseinstellungen. Nur `flip_prob` und `no_aug_epochs` sind daher aktiv. Diese Pipeline gilt für D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4, EC und mit einer Änderung RF-DETR.

Die ImageFolder-Pipeline für Klassifizierung ignoriert jede Erkennungseinstellung. Ihre horizontale Spiegelung verwendet einen festen Wert von 0,5, den `flip_prob` nicht erreicht. Stattdessen besitzt sie ein eigenes, weiter unten beschriebenes Einstellungspaket.

YOLO-NAS bildet eine eigene Form: kein Mosaic, eine immer aktive affine Transformation je Sample und unabhängiges statt gekoppeltes MixUp. Der Wert `mosaic_scale` wird als affiner Skalierungsbereich wiederverwendet.

SegFormer und NAFNet führen jeweils eine aufgabenspezifische Pipeline aus, deren Zufälligkeit fest in der Familie definiert und nicht konfigurierbar ist. Bei SegFormer sind die Klassenattribute `semantic_scale_jitter` und `semantic_hsv_prob` aktiv, nicht `mosaic_scale` und `hsv_prob`. Beschnitt und Spiegelungen von NAFNet koppeln Eingabe- und Zieloperationen mit einer festen Wahrscheinlichkeit von 0,5.

## Berücksichtigte Einstellungen je Familie

Die folgende Tabelle fasst die ausgelieferte Spezifikation unter `libreyolo/data/augment/spec.py` zusammen. Bibliothekstests gleichen sie mit der tatsächlichen Pipeline-Verdrahtung ab. Lies sie dort, statt aus der Architektur zu folgern.

<code-tabs name="support" />

Zusammenfassung der Basiseinstellungen nach Pipeline:

| Einstellung | YOLOX-Stil | YOLO-NAS | DETR-Stil | Klassifizierung |
|---|---|---|---|---|
| `mosaic_prob` | verwendet | ignoriert | ignoriert | ignoriert |
| `mixup_prob` | an Mosaic gekoppelt | verwendet | ignoriert | ignoriert |
| `hsv_prob` | verwendet | verwendet | ignoriert | ignoriert |
| `flip_prob` | verwendet | verwendet | verwendet | ignoriert |
| `flipud` | verwendet | verwendet | ignoriert | ignoriert |
| `degrees` | an Mosaic gekoppelt | verwendet | ignoriert | ignoriert |
| `translate` | an Mosaic gekoppelt | verwendet | ignoriert | ignoriert |
| `shear` | an Mosaic gekoppelt | verwendet | ignoriert | ignoriert |
| `perspective` | an Mosaic gekoppelt | verwendet | ignoriert | ignoriert |
| `mosaic_scale` | an Mosaic gekoppelt | verwendet | ignoriert | ignoriert |
| `mixup_scale` | an Mosaic gekoppelt | verwendet | ignoriert | ignoriert |
| `no_aug_epochs` | verwendet | verwendet | verwendet | verwendet |

Innerhalb dieser Spalten gibt es ausschließlich einschränkende Ausnahmen:

- RTMDet, PicoDet, RT-DETR, RT-DETRv2 und FOMO besitzen keine vertikale Spiegelung, daher wird `flipud` ignoriert. Der Mosaic-Wrapper von FOMO ist außerdem ohne Perspektive aufgebaut.
- Die native Pipeline von RF-DETR besitzt keinen HSV-Jitter, weshalb zusätzlich zur Spalte im DETR-Stil `hsv_prob` ignoriert wird.
- EC berücksichtigt `hsv_prob`, `degrees` und `translate`, aber nur für `task="pose"`, dessen Keypoint-bewusste Transformation sie liest. Die detect- und segment-Pfade verwenden feste fotometrische Rezepte.
- DINOv2 folgt für detect und semantic der Spalte im DETR-Stil und ergänzt für `task="classify"` das Klassifizierungspaket.

`no_aug_epochs` wird überall verwendet, bedeutet jedoch nicht überall dasselbe. In Mosaic-Pipelines deaktiviert es Mosaic und MixUp für die letzten Epochen. In Pipelines im DETR-Stil beendet es fotometrische, Zoom-out- und Beschnitt-Augmentationen und formt das Ende des Schedules. In Klassifizierungs- und Semantikpipelines formt es nur das Ende des Schedules.

## Klassifizierungspaket

Vier Einstellungen steuern ausschließlich die Klassifizierungspipeline. Erkennungsfamilien ignorieren alle vier.

<code-tabs name="classify" />

`auto_augment` akzeptiert `"randaugment"`, `"autoaugment"`, `"augmix"` oder `None`. `erasing` ist die RandomErasing-Wahrscheinlichkeit. `mixup` und `cutmix` sind Wahrscheinlichkeiten je Batch, die weiche Labels erzeugen. Höchstens eine davon wird je Batch ausgeführt, wobei MixUp zuerst kommt. Die beiden Werte sind daher additiv und sollten höchstens 1 ergeben.

Alle vier sind standardmäßig deaktiviert. Das Klassifizierungstraining bleibt unverändert, sofern du sie nicht anforderst.

Ein Namenskonflikt ist wichtig: In der CLI ist `mixup` der Alias für das bei der Erkennung verwendete `mixup_prob`. Das Klassifizierungsfeld `mixup` besitzt keine eigene CLI-Schreibweise und ist nur über `model.train(mixup=...)` in Python erreichbar.

## Familienspezifische Einstellungen

Einige Einstellungen liegen in der Konfigurationsunterklasse einer Familie statt in der Basisklasse. Sie existieren daher nur für diese Familie und besitzen kein CLI-Flag.

| Familie | Einstellung | Wirkung |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Wahrscheinlichkeit der Copy-Paste-Instanzaugmentation, nur für `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` verwendet dasselbe Sample gespiegelt, `"mixup"` lädt ein zweites Sample |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Wahrscheinlichkeit einer zufälligen Drehung um 90 Grad |
| YOLOv9 | `max_labels` | Obergrenze der Ground-Truth-Labels je Bild in den Trainingstransformationen, Standard 100 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-Paste für `task="segment"`, nur Modus `"flip"` |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Wahrscheinlichkeit für zufälliges Beschneiden und Skalieren |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Jitter im Pose-Pfad und Wahrscheinlichkeit einer Keypoint-bewussten affinen Transformation |

`max_labels` ist die einzige Einstellung, durch die unbemerkt Daten verloren gehen. Boxen oberhalb der Grenze werden ohne Fehler verworfen. Für dichte Bilder wie Luftaufnahmen muss der Wert daher erhöht werden.

Mosaic und MixUp sind beim Training orientierter Boxen unabhängig von den Einstellungen deaktiviert, da eine eckenbewusste Augmentation für gedrehte Boxen nicht implementiert ist.

## Verwandte Themen

- Unter [Hyperparameter](/docs/train/hyperparameters) findest du `no_aug_epochs` als Schedule-Argument sowie die übrigen Argumente von `train()`.
- Unter [Datensätze](/docs/train/datasets) findest du die von diesen Transformationen verwendeten Labelformate.
