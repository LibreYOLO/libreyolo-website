---
title: Wissensdistillation
seo_title: Wissensdistillation in LibreYOLO
description: >-
  Trainiere einen kleinen Detektor mithilfe eines größeren Lehrermodells oder
  eines eingefrorenen DINOv2-Backbones. Erfahre mehr über MGD-, CWD- und
  Feature-MSE-Loss, Abgriffspunkte und Familienunterstützung.
lead: >-
  Die Distillation ergänzt einen zweiten Loss-Term, der die intermediären
  Feature Maps des Schülermodells an die eines eingefrorenen Lehrermodells
  annähert. LibreYOLO greift Features mit Forward Hooks ab, sodass der eigene
  Head und Loss des Lehrermodells nie beteiligt sind.
keywords:
  - wissensdistillation
  - masked generative distillation
  - channel-wise distillation
  - feature distillation
  - dinov2 lehrermodell
  - teacher student training
  - mgd loss
  - cwd loss
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Ein größerer Checkpoint derselben Familie leitet den kleinen an.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Ein eingefrorenes selbstüberwachtes ViT leitet eine Backbone-Stufe an.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Abstimmung des Loss
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # globales Distillationsgewicht
            distill_tau=1.0,   # CWD-Softmax-Temperatur
        )
source_hash: 7210031328f6826f
---

## Distillation aus einem größeren Checkpoint

Durch das Setzen von `distill_model` wird die Distillation aktiviert. Der Wert
ist der Checkpoint eines Lehrermodells und wird über dieselbe Factory wie jedes
andere Modell geladen.

<code-tabs name="detector" />

Der Forward Pass des Lehrermodells läuft unter `no_grad` und bei aktiviertem
AMP auch unter Autocast. Dadurch muss das eingefrorene Modell nicht bei jedem
Schritt die volle Rechenleistung für FP32 aufbringen. Forward Hooks erfassen
seine Feature Maps an benannten Abgriffspunkten. Der Loss vergleicht sie mit
denen des Schülermodells. Das Ergebnis wird zum Trainings-Loss addiert und als
Komponente namens `distill` gemeldet.

## Distillation aus einem eingefrorenen Foundation-Backbone

Alternativ kann ein selbstüberwachtes ViT eine einzelne Backbone-Stufe des
Schülermodells anleiten. Die Features des Lehrermodells stammen aus seinem
eigenen Feature Extractor und nicht aus Hooks. Der Loss gleicht den Unterschied
zwischen einem Patch-Raster und einem Convolutional Stride aus.

<code-tabs name="foundation" />

`distill_model` erkennt `dinov2`, womit DINOv2-base gemeint ist, sowie
`dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`,
`dinov2-base`, `dinov2-large` und jede rohe Hub-ID, die mit
`facebook/dinov2` beginnt. Jeder andere Wert wird als Pfad zu einem
Lehrermodell-Checkpoint behandelt.

Dieser Pfad verwendet unabhängig von `distill_loss_type` immer `feat_mse` und
setzt installiertes `transformers` voraus. Wenn beim Laden des Lehrermodells
Gewichtsschlüssel fehlen, wird der Vorgang abgebrochen, statt gegen ein
teilweise zufälliges Backbone zu distillieren.

## Unterstützte Familien

Die Distillationsunterstützung wird durch eine Methode des Schülermodells
bereitgestellt. Davon gibt es zwei.

`get_distill_config()` liefert die mehrskaligen Abgriffspunkte, die von einem
Detektor-Lehrermodell überwacht werden. YOLOv9, YOLOX und RF-DETR implementieren
diese Methode.

`get_backbone_distill_config()` liefert die einzelne Backbone-Stufe, die von
einem Foundation-Lehrermodell überwacht wird. Nur YOLOv9 implementiert diese
Methode.

Jeder andere Fall löst einen Fehler aus, statt ohne den Loss zu trainieren:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Abgriffspunkte

Die Abgriffspunkte sind für jede Familie und Rolle festgelegt. Lehrermodell und
Schülermodell müssen daher nicht dieselbe Architektur besitzen, benötigen aber
übereinstimmende Feature-Strides.

| Familie | Rolle | Abgriffspunkte | Strides |
|---|---|---|---|
| YOLOv9 | Lehrer oder Schüler | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | Foundation-Schüler | `backbone.elan3` | 16 |
| YOLOX | Lehrer oder Schüler | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | Lehrer oder Schüler | `model.backbone.0.projector.stages.0` | bei der Einrichtung ermittelt |

Nicht übereinstimmende Strides lösen vor Trainingsbeginn einen Fehler aus:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

Bei Foundation-Lehrermodellen wird diese Prüfung übersprungen, da gerade die
unterschiedlichen Raster ihr Anwendungsfall sind.

## Die drei Loss-Funktionen

`distill_loss_type` wählt den Feature-Loss für ein Detektor-Lehrermodell. Ein
Foundation-Lehrermodell verwendet immer `feat_mse`.

`mgd`, Masked Generative Distillation, maskiert einen Anteil der räumlichen
Positionen des Schülermodells. Ein kleiner Generator aus zwei Convolutions wird
darauf trainiert, die vollständige Feature Map des Lehrermodells aus den
verbleibenden Positionen zu rekonstruieren. `distill_mask_ratio` legt den
maskierten Anteil fest. Der Standardwert ist 0.65.

`cwd`, Channel-Wise Distillation, wandelt die räumlichen Aktivierungen jedes
Kanals in eine Wahrscheinlichkeitsverteilung um und minimiert kanalweise die
KL-Divergenz. `distill_tau` ist die Softmax-Temperatur mit dem Standardwert 1.0.

`feat_mse` gleicht die Kanäle des Schülermodells mit einer 1x1-Convolution an
die des Lehrermodells an, skaliert das Raster des Lehrermodells bilinear auf
das Raster des Schülermodells und berechnet den mittleren quadratischen Fehler.
Mit `distill_normalize=True` werden beide Feature Maps zunächst über die
Kanaldimension L2-normalisiert. Dadurch berücksichtigt der Abgleich nur den
Winkel und ist skaleninvariant. Der Standardwert ist `False`.

`dis` ist das anschließend angewendete globale Gewicht. Ohne Angabe verwendet
jeder Loss seinen eigenen veröffentlichten Standardwert: 2e-5 für MGD, 1.0 für
CWD und 1.0 für Feature MSE. Diese Werte unterscheiden sich um fünf
Größenordnungen. Ein für einen Loss-Typ abgestimmtes Gewicht ist daher für
einen anderen bedeutungslos.

<code-tabs name="tuned" />

Für `distill_mask_ratio`, `distill_tau` und `distill_normalize` gibt es keine
CLI-Flags. Sie werden als Python-Argumente oder Schlüssel einer `cfg=`-YAML
angegeben. Auch die Distillation mit RF-DETR ist insgesamt nur über Python
verfügbar, da seine CLI-Argumentzuordnung keine Distillationsschlüssel enthält.

## Adapter, Checkpoints und Multi-GPU

Jeder Loss erstellt kleine trainierbare Module außerhalb des Schülermodells:
die 1x1-Kanaladapter und den Generator von MGD. Sie erhalten eine eigene
Optimizer-Parametergruppe mit der effektiven Lernrate des Laufs.

Diese Module werden unter einem `distiller`-Schlüssel in den Checkpoint
geschrieben und beim Fortsetzen wiederhergestellt. Ein fortgesetzter Lauf
beginnt daher nicht mit neu initialisierten Projectors.

Unter DDP liegen die Adapter außerhalb des umschlossenen Schülermodells. Der
DDP-Reducer sieht ihre Gradienten deshalb nicht. Der Trainer führt für sie bei
jedem Schritt explizit ein All-Reduce aus, damit alle Ränge dieselben Adapter
trainieren.

Die CUDA-Graph-Erfassung ist während eines Distillationslaufs nicht verfügbar.
`cuda_graph=True` protokolliert eine Zeile und trainiert im Eager-Modus. Siehe
[Trainingsperformance](/docs/train/performance).

## Verwandte Themen

- [Einfrieren von Schichten](/docs/train/layer-freezing) und
  [LoRA-Fine-Tuning](/docs/train/lora) lassen sich beide mit der Distillation
  kombinieren.
- Unter [Hyperparameter](/docs/train/hyperparameters) findest du die übrigen
  Optionen von `train()`.

