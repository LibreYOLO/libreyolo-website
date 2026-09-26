---
title: Augmentationsmatrix
seo_title: Welche LibreYOLO-Familie welche Augmentation berücksichtigt
description: >-
  Unterstützung der Augmentationseinstellungen je Familie: sechzehn
  TrainConfig-Einstellungen, drei Statuswerte, sechs Pipeline-Archetypen und von
  Familien unbemerkt ignorierte Einstellungen.
lead: >-
  Das Festlegen einer Augmentationseinstellung garantiert nicht, dass sie die
  Pipeline erreicht. Diese Seite dokumentiert anhand der deklarativen Tabelle
  der Bibliothek, wie jede trainierbare Familie die Einstellungen in TrainConfig
  behandelt. Sie ist die einzige maßgebliche Quelle.
keywords:
  - LibreYOLO Augmentation
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - Augmentation Unterstützungsmatrix
  - TrainConfig Einstellungen
last_verified: 1.6.0
verification: >-
  Liste der Einstellungen, Statuswerte, Archetypen, familienbezogene
  Abweichungen und Hilfsfunktionen aus libreyolo/data/augment/spec.py in v1.6.0
  gelesen. Die Zuordnung dieser Tabelle zu den tatsächlichen Pipelines wird
  durch tests/unit/test_augment_spec.py abgesichert.
snippets:
  usage:
    - label: Spezifikation direkt abfragen
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
source_hash: f3cba41ceadf131f
---

## Einstellungen

Dies sind Feldnamen von `TrainConfig`, nicht ihre Schreibweisen in der CLI. Die CLI bildet eigene Aliasse darauf ab. `--mosaic` setzt daher `mosaic_prob`.

| Einstellung | Bedeutung |
|---|---|
| `mosaic_prob` | Wahrscheinlichkeit für die Erstellung eines Mosaic-Samples aus 4 Bildern |
| `mixup_prob` | Wahrscheinlichkeit für die Einmischung eines zweiten Samples |
| `hsv_prob` | Wahrscheinlichkeit für HSV-Farb-Jitter |
| `flip_prob` | Wahrscheinlichkeit einer horizontalen Spiegelung |
| `degrees` | Bereich zufälliger Drehungen für die affine Transformation in Grad |
| `translate` | Anteil zufälliger Verschiebung für die affine Transformation |
| `mosaic_scale` | Bereich zufälliger Skalierung für die affine Transformation |
| `mixup_scale` | Bereich des Skalierungs-Jitters für das MixUp-Partnerbild |
| `shear` | Bereich zufälliger Scherung für die affine Transformation in Grad |
| `perspective` | Stärke der projektiven Transformation für die affine Transformation |
| `flipud` | Wahrscheinlichkeit einer vertikalen Spiegelung |
| `no_aug_epochs` | Abschließende Epochen mit deaktivierter starker Augmentation |
| `auto_augment` | AutoAugment-Richtlinie für Klassifizierung: randaugment, autoaugment oder augmix |
| `erasing` | RandomErasing-Wahrscheinlichkeit für Klassifizierung |
| `mixup` | Batch-MixUp-Wahrscheinlichkeit mit weichen Labels für Klassifizierung |
| `cutmix` | Batch-CutMix-Wahrscheinlichkeit mit weichen Labels für Klassifizierung |

Die letzten vier bilden die Klassifikationsparameter. Detektionsfamilien ignorieren sie. Die CLI leitet `mixup` bei Klassifikatoren an die Batch-Mischung und bei Detektoren an `mixup_prob` weiter.

<code-tabs name="usage" />

## Drei Statuswerte

| Status | Bedeutung |
|---|---|
| `used` | Die Einstellung erreicht die Trainingspipeline der Familie und verändert Samples |
| `gated_by_mosaic` | Die Einstellung gilt nur für Samples im Mosaic-Zweig und wird bei `mosaic_prob == 0` nie ausgelöst |
| `ignored` | Die Einstellung erreicht die Pipeline nie und bleibt wirkungslos |

Prüfe vor einem Lauf insbesondere `ignored`, da kein Fehler ausgelöst wird. Die CLI warnt, wenn ein ausdrücklich gesetzter Trainingsparameter von der gewählten Familie ignoriert wird. Der Trainer warnt außerdem, wenn `mixup_prob > 0` nicht ausgelöst werden kann, weil die Familie MixUp an Mosaic koppelt und `mosaic_prob` null ist.

## Pipeline-Archetypen

Jede erfasste Familie folgt einer von sechs Pipelines. Einige familienbezogene Abweichungen stehen weiter unten.

| Einstellung | YOLOX-Stil | YOLO-NAS | DETR-Stil | Klassifizierung | Semantik | Restauration |
|---|---|---|---|---|---|---|
| `mosaic_prob` | verwendet | ignoriert | ignoriert | ignoriert | ignoriert | ignoriert |
| `mixup_prob` | gekoppelt | verwendet | ignoriert | ignoriert | ignoriert | ignoriert |
| `hsv_prob` | verwendet | verwendet | ignoriert | ignoriert | ignoriert | ignoriert |
| `flip_prob` | verwendet | verwendet | verwendet | verwendet | ignoriert | ignoriert |
| `degrees` | gekoppelt | verwendet | ignoriert | ignoriert | ignoriert | ignoriert |
| `translate` | gekoppelt | verwendet | ignoriert | ignoriert | ignoriert | ignoriert |
| `mosaic_scale` | gekoppelt | verwendet | ignoriert | ignoriert | ignoriert | ignoriert |
| `mixup_scale` | gekoppelt | verwendet | ignoriert | ignoriert | ignoriert | ignoriert |
| `shear` | gekoppelt | verwendet | ignoriert | ignoriert | ignoriert | ignoriert |
| `perspective` | gekoppelt | verwendet | ignoriert | ignoriert | ignoriert | ignoriert |
| `flipud` | verwendet | verwendet | ignoriert | verwendet | ignoriert | ignoriert |
| `no_aug_epochs` | verwendet | verwendet | verwendet | verwendet | verwendet | verwendet |
| `auto_augment` | ignoriert | ignoriert | ignoriert | verwendet | ignoriert | ignoriert |
| `erasing` | ignoriert | ignoriert | ignoriert | verwendet | ignoriert | ignoriert |
| `mixup` | ignoriert | ignoriert | ignoriert | verwendet | ignoriert | ignoriert |
| `cutmix` | ignoriert | ignoriert | ignoriert | verwendet | ignoriert | ignoriert |

In der Pipeline im YOLOX-Stil wendet die Vorverarbeitung je Sample HSV-Jitter und Spiegelungen an. Die affine Transformation und MixUp laufen nur innerhalb des Mosaic-Zweigs. YOLO-NAS führt dagegen immer eine affine Transformation je Sample aus, ignoriert Mosaic und wendet MixUp unabhängig an. Dabei wird `mosaic_scale` als Skalierungsbereich der affinen Transformation wiederverwendet.

Die DETR-Pipeline verwendet eine Transformation ohne Mosaic. Photometrische Verzerrung, Zoom-out und IoU-Cropping sind Rezeptkonstanten statt konfigurierbarer Parameter; `hsv_prob` und die Geometrieparameter steuern sie daher nicht. Klassifikation verwendet `flip_prob` für horizontales und `flipud` für vertikales Spiegeln. Semantische Skalierungsvariation und HSV stammen aus Attributen der Modellklasse. Bei Restaurierung werden Eingabe und Ziel gemeinsam mit einer festen Wahrscheinlichkeit von 0.5 gespiegelt.

`no_aug_epochs` wird überall berücksichtigt, schaltet aber unterschiedliche Operationen ab: Mosaic und MixUp bei YOLOX-Pipelines, affine Transformation und MixUp bei YOLO-NAS, starke photometrische Augmentierung und Cropping sowie die Schlussphase des Lernratenplans bei DETR, automatische Augmentierung, Erasing, MixUp und CutMix bei Klassifikation. Cropping und Spiegeln bleiben bei Klassifikation aktiv.

## Familien nach Archetyp

| Archetyp | Familien |
|---|---|
| YOLOX-Stil | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| DETR-Stil | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| Klassifizierung | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| Semantik | `segformer` |
| Restauration | `nafnet` |

Fünfundzwanzig Familien werden erfasst. Für eine Familie außerhalb dieser Liste wird eine leere Menge ignorierter Einstellungen zurückgegeben, sodass keine Warnung erscheint.

## Abweichungen

| Familie | Unterschied zum Archetyp |
|---|---|
| `rtmdet` | `flipud` wird ignoriert: Seine Transformation besitzt keine vertikale Spiegelung |
| `picodet` | `flipud` wird ignoriert |
| `rtdetr` | `flipud` wird ignoriert |
| `rtdetrv2` | `flipud` wird ignoriert |
| `fomo` | `perspective` und `flipud` werden ignoriert |
| `ec` | `hsv_prob`, `degrees` und `translate` werden nur für `task="pose"` verwendet; detect und segment nutzen feste fotometrische Rezepte |
| `dinov2` | Das Klassifizierungspaket wird nur für `task="classify"` verwendet |

`ec` und `dinov2` sind Multi-Task-Familien. Eine Einstellung wird deshalb nur dann als ignoriert markiert, wenn jede trainierbare Aufgabe der Familie sie ignoriert. So kann die CLI-Warnung nie für eine Aufgabe falsch und für eine andere richtig sein.

Dome-DETR übernimmt die Transformationen von D-FINE unverändert. Nur Multi-Scale-Training wird nicht unterstützt. Dies wird von seiner Konfiguration statt der Augmentationsspezifikation deaktiviert.

## Familienspezifische Einstellungen

Einige Familien definieren Augmentationseinstellungen in ihrer eigenen `TrainConfig`-Unterklasse statt in der Basisklasse. Die CLI stellt sie nicht bereit. Lege sie über die Python-API fest.

| Familie | Einstellung | Bedeutung |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Wahrscheinlichkeit der Copy-Paste-Instanzaugmentation, nur für `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Copy-Paste-Quelle: `flip` spiegelt dasselbe Sample, `mixup` verwendet ein zweites Sample |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Wahrscheinlichkeit einer zufälligen Drehung um 90 Grad |
| `rfdetr` | `copy_paste` | Copy-Paste-Wahrscheinlichkeit für `task="segment"`, nur Modus `flip` |
| `rfdetr` | `copy_paste_mode` | Copy-Paste-Quellmodus für `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Wahrscheinlichkeit für zufälliges Beschneiden und Skalieren in der nativen Pipeline |
| `dfine` | `crop_resize_prob` | Wahrscheinlichkeit für zufälliges Beschneiden und Skalieren bei `task="segment"` |
| `ec` | `crop_resize_prob` | Wahrscheinlichkeit für zufälliges Beschneiden und Skalieren bei `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Wahrscheinlichkeit für Helligkeits- und Kontrast-Jitter bei `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Keypoint-bewusste Wahrscheinlichkeit einer affinen Transformation bei `task="pose"` |

`rot90` gilt bei `yolo9` für detect und OBB.

## Abfrage der Spezifikation

| Hilfsfunktion | Rückgabewert |
|---|---|
| `aug_support(family)` | Tabelle von Einstellung zu `Support` oder `None` bei einer unbekannten Familie |
| `ignored_aug_params(family)` | Menge der von der Familie ignorierten Einstellungsnamen; leer bei einer unbekannten Familie |
| `uses_mosaic_gating(family)` | Gibt an, ob MixUp der Familie nur bei Mosaic-Samples ausgelöst wird |
| `display_name(family)` | In Warnungen verwendeter Anzeigename der Familie |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Warntext, wenn MixUp nie ausgelöst werden kann, andernfalls `None` |

`Support` ist ein benanntes Tupel aus `status` und `note`. Die Anmerkung erklärt, warum eine Einstellung bei dieser Familie ignoriert oder gekoppelt wird.

## Mosaic-Kopplung

Bei einer Familie im YOLOX-Stil deaktiviert `mixup_prob=0.5` zusammen mit `mosaic_prob=0` MixUp vollständig, da MixUp nur auf Mosaic-Samples angewendet wird. Diese Kombination entsteht leicht, wenn Mosaic spät im Training deaktiviert wird. Der Trainer protokolliert eine Warnung mit dem Familiennamen. `mixup_gating_warning` ist die zugrunde liegende reine Funktion.
