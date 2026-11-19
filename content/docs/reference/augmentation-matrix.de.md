---
title: Augmentierungsmatrix
seo_title: Welche LibreYOLO-Familie welche Augmentierung berücksichtigt
description: >-
  Jede Augmentierungseinstellung mit Wertebereich und CLI-Schreibweise, der
  Zustand jeder Einstellung in jeder der 31 trainierbaren Familien und die
  Standardwerte, mit denen jede Familie ausgeliefert wird.
lead: >-
  Das Setzen einer Augmentierungseinstellung garantiert nicht, dass sie die
  Pipeline erreicht. Diese Seite hält fest, wie jede trainierbare Familie jede
  Einstellung in TrainConfig behandelt und mit welchem Wert jede Familie
  startet, generiert aus der Bibliothek im dokumentierten Release.
keywords:
  - libreyolo augmentierung
  - augmentierungsparameter
  - augmentierung standardwerte
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - augmentierung unterstützungsmatrix
  - TrainConfig einstellungen
last_verified: 1.8.0
verification: >-
  Generiert von scripts/build-augmentation/collect.py aus LibreYOLO 1.8.0: die
  Liste der Einstellungen, Zustände und Anmerkungen aus
  libreyolo/data/augment/spec.py, das die Tests der Bibliothek an die echten
  Pipelines binden, die Standardwerte aus der TrainConfig-Unterklasse jeder
  Familie und die CLI-Schreibweisen aus dem Befehl libreyolo train. Die
  Wertebereiche und die eine Stelle, an der die Spezifikation und eine Pipeline
  voneinander abweichen, prüft dasselbe Skript gegen die Bibliothek.
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
    - label: Standardwerte einer Familie
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## Die Einstellungen

Dies sind Feldnamen von `TrainConfig`. `train()` und die CLI akzeptieren außerdem die kurzen Schreibweisen aus der CLI-Spalte, sodass `mosaic` das Feld `mosaic_prob` setzt. In der CLI wird `mixup` nach Aufgabe weitergeleitet: an die Batch-Mischung der Klassifikation bei Klassifikatoren, an `mixup_prob` bei Detektoren. `perspective` hat kein CLI-Flag und wird aus Python gesetzt.

<aug-knobs />

Der Basis-Standardwert ist der Wert in `TrainConfig` selbst. Die meisten Familien überschreiben einige davon, und [Standardwerte je Familie](#defaults-by-family) nennt die Werte, mit denen ein Lauf tatsächlich startet. Bilder dazu, was jede Einstellung bewirkt, findest du unter [Augmentierungen](/docs/train/augmentations).

Wertebereiche in Klammern sind Intervalle, `[0, 1)` schließt also 0 ein und 1 aus. Die Klassifikationseinstellungen werden beim Aufbau ihrer Pipeline geprüft und lösen außerhalb ihres Bereichs `ValueError` aus. Die Einstellungen für die Objekterkennung prüft `TrainConfig` nicht, daher trainiert ein Wert außerhalb des Bereichs ohne jede Meldung. Ausnahmen sind eine gezogene Skalierung von 0 oder weniger und ein negatives `perspective`, die `ValueError` auslösen, sobald ein Sample transformiert wird.

<code-tabs name="usage" />

## Die drei Statuswerte

| Status | Bedeutung |
|---|---|
| `used` | Die Einstellung erreicht die Trainingspipeline der Familie und verändert Samples |
| `gated_by_mosaic` | Die Einstellung gilt nur für Samples, die den Mosaic-Zweig durchlaufen haben, und wird daher bei `mosaic_prob == 0` nie ausgelöst |
| `ignored` | Die Einstellung erreicht die Pipeline nie; das Setzen bewirkt nichts |

`ignored` ist der Status, den du vor einem Lauf prüfen solltest, weil nichts fehlschlägt. Die CLI warnt, wenn ein ausdrücklich gesetzter Trainingsparameter einer ist, den die gewählte Familie ignoriert, und der Trainer warnt, wenn `mixup_prob > 0` nicht ausgelöst werden kann, weil die Familie MixUp an Mosaic koppelt und `mosaic_prob` null ist.

## Berücksichtigte Einstellungen je Familie

<aug-matrix />

Jede erfasste Familie folgt einer von sechs Pipelines. In der Pipeline im YOLOX-Stil wendet die Vorverarbeitung je Sample HSV-Jitter und Spiegelungen an, während die affine Transformation und MixUp nur innerhalb des Mosaic-Zweigs laufen. YOLO-NAS führt stattdessen eine immer aktive affine Transformation je Sample aus, ignoriert Mosaic und wendet MixUp unabhängig an, wobei `mosaic_scale` als Skalierungsbereich der affinen Transformation wiederverwendet wird.

Die Pipeline im DETR-Stil ist eine durchreichende Transformation ohne Mosaic. Ihre fotometrische Verzerrung, ihr Zoom-out und ihr IoU-Zuschnitt sind Rezeptkonstanten statt konfigurierbarer Einstellungen, weshalb `hsv_prob` und die Geometrieeinstellungen sie nie erreichen. Die Klassifikation verwendet `flip_prob` für horizontale und `flipud` für vertikale Spiegelungen. Skalierungs-Jitter und HSV der semantischen Segmentierung stammen aus Klassenattributen der Familien statt aus Konfigurationseinstellungen, und die Spiegelungen der Restaurierung sind gekoppelte Operationen auf Eingabe und Ziel mit einer festen Wahrscheinlichkeit von 0.5.

`no_aug_epochs` wird überall berücksichtigt, doch was es abschaltet, unterscheidet sich: Mosaic und MixUp im YOLOX-Stil, die affine Transformation und MixUp bei YOLO-NAS, die starken fotometrischen und Zuschnitt-Augmentierungen sowie das Ende des Lernratenplans im DETR-Stil, und Auto-Augmentierung, Erasing, MixUp und CutMix bei der Klassifikation. Zuschnitt und Spiegelungen der Klassifikation bleiben aktiv.

Wo eine Familie vom Rest ihrer Pipeline abweicht, ist der Grund einer der folgenden:

| Familie | Unterschied |
|---|---|
| `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2` | `flipud` ignoriert: Ihre Transformationen haben keine vertikale Spiegelung |
| `fomo` | `perspective` und `flipud` ignoriert |
| `ppyoloe` | `flipud` ignoriert: Das Quellrezept verwendet stattdessen eine zufällige 90-Grad-Drehung |
| `yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolo7`, `rtdetr`, `rtdetrv2` | `mixup_scale` ignoriert: Ihr MixUp mischt ganze Bilder in einem zufälligen Verhältnis, ohne eines davon neu zu skalieren |
| `gtr` | `mosaic_prob` und `mixup_prob` verwendet, für die ersten `mosaic_epochs` Epochen; `degrees`, `translate` und `mosaic_scale` formen die affine Transformation, die auf das Mosaic folgt |
| `rfdetr` | `hsv_prob` ignoriert: Die native Pipeline hat keinen HSV-Jitter |
| `ec` | `hsv_prob`, `degrees` und `translate` verwendet, nur für `task="pose"`; detect und segment verwenden feste fotometrische Rezepte |
| `dinov2` | Das Klassifikationspaket wird verwendet, nur für `task="classify"` |

`ec` und `dinov2` sind Multi-Task-Familien, daher wird eine Einstellung nur dann als ignoriert markiert, wenn jede trainierbare Aufgabe der Familie sie ignoriert. So ist die CLI-Warnung nie für eine Aufgabe falsch und zugleich für eine andere richtig.

Dome-DETR übernimmt die Transformationen von D-FINE unverändert. Das Einzige, was es nicht verarbeiten kann, ist Multi-Scale-Training, das seine Konfiguration deaktiviert und nicht die Augmentierungsspezifikation.

Einunddreißig Familien sind erfasst. Für eine Familie außerhalb dieser Liste liefert `ignored_aug_params` eine leere Menge, sodass für sie keine Warnung ausgegeben wird.

## Standardwerte je Familie

Der Wert, mit dem die Konfiguration jeder Familie für Objekterkennung startet. Eine leere Zelle bedeutet, dass die Familie diese Einstellung ignoriert, ihr Standardwert also nichts ändert.

<aug-defaults />

Zwei Muster sind wichtiger als jeder einzelne Wert. RT-DETR, RT-DETRv2, PicoDet und FOMO starten mit `mosaic_prob=0`, sodass bei diesen Familien die in der Matrix als gekoppelt markierten Einstellungen nichts bewirken, bis Mosaic eingeschaltet wird. YOLOv9 startet mit abgeschalteter Drehung, Scherung und MixUp und mit `mosaic_scale=(0.5, 1.5)`, während der Basis-Standardwert `(0.1, 2.0)` ist.

EdgeCrafter liest `hsv_prob`, `degrees` und `translate` nur bei `task="pose"` aus, daher zeigt seine Zeile für diese drei die Standardwerte der Pose-Aufgabe.

Klassifikationsfamilien starten mit den Basis-Standardwerten: `flip_prob=0.5`, `flipud=0.0`, `scale=(0.5, 1.0)`, dem familieneigenen `crop_pct` und abgeschalteten `auto_augment`, `erasing`, `mixup` und `cutmix`.

## Familienspezifische Einstellungen

Einige Familien führen Augmentierungseinstellungen in ihrer eigenen `TrainConfig`-Unterklasse statt in der Basisklasse. Die CLI stellt sie nicht bereit; setze sie über die Python-API.

| Familie | Einstellung | Bedeutung |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Wahrscheinlichkeit der Copy-Paste-Instanzaugmentierung, nur bei `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Copy-Paste-Quelle: `flip` spiegelt dasselbe Sample, `mixup` verwendet ein zweites Sample |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Wahrscheinlichkeit einer zufälligen 90-Grad-Drehung |
| `rfdetr` | `copy_paste` | Copy-Paste-Wahrscheinlichkeit bei `task="segment"`, nur im Modus `flip` |
| `rfdetr` | `copy_paste_mode` | Copy-Paste-Quellmodus bei `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Wahrscheinlichkeit für zufälliges Zuschneiden und Skalieren in der nativen Pipeline |
| `dfine` | `crop_resize_prob` | Wahrscheinlichkeit für zufälliges Zuschneiden und Skalieren, `task="segment"` |
| `ec` | `crop_resize_prob` | Wahrscheinlichkeit für zufälliges Zuschneiden und Skalieren, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Wahrscheinlichkeit für Helligkeits- und Kontrast-Jitter, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Wahrscheinlichkeit einer Keypoint-bewussten affinen Transformation, `task="pose"` |

`rot90` gilt bei `yolo9` für detect und OBB.

## Abfrage der Spezifikation

| Hilfsfunktion | Rückgabewert |
|---|---|
| `aug_support(family)` | Die Tabelle von Einstellung zu `Support` oder `None` bei einer unbekannten Familie |
| `ignored_aug_params(family)` | Die Menge der Einstellungsnamen, die die Familie ignoriert; leer bei einer unbekannten Familie |
| `uses_mosaic_gating(family)` | Ob das MixUp der Familie nur bei Mosaic-Samples ausgelöst wird |
| `display_name(family)` | Der für Menschen gedachte Familienname, der in Warnungen verwendet wird |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Der Warntext, wenn MixUp nie ausgelöst werden kann, sonst `None` |

Ein `Support` ist ein benanntes Tupel aus `status` und `note`, wobei die Anmerkung erklärt, warum eine Einstellung bei dieser Familie ignoriert oder gekoppelt ist. In 1.8.0 führt die Spezifikation `mixup_scale` für die Familien, die über den Mosaic-Wrapper von YOLOv9 trainieren, noch als gekoppelt, daher meldet `aug_support` es dort als gekoppelt, obwohl die Matrix oben, die gegen die Pipeline geprüft ist, es als ignoriert markiert.

## Die Mosaic-Kopplung

Bei einer Familie im YOLOX-Stil deaktiviert `mixup_prob=0.5` zusammen mit `mosaic_prob=0` MixUp vollständig, weil MixUp nur auf Mosaic-Samples angewendet wird. Diese Kombination entsteht leicht, wenn Mosaic spät im Training abgeschaltet wird. Der Trainer protokolliert eine Warnung, die die Familie nennt, und `mixup_gating_warning` ist die reine Funktion dahinter.
