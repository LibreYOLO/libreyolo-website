---
title: Augmentierungen
seo_title: 'Trainingsaugmentierungen in LibreYOLO, mit Beispielen'
description: >-
  Was jede Augmentierungseinstellung von LibreYOLO bewirkt, gezeigt an einem
  echten Bild durch den Trainingscode der Bibliothek selbst, mit Wertebereich,
  Standardwert und den Familien, die sie auslesen.
lead: >-
  Trainingsaugmentierungen sind zufällige Transformationen, die auf jedes
  Trainingssample angewendet werden. Sie werden über Einstellungen in
  TrainConfig konfiguriert, und jede Modellfamilie führt ihre eigene Pipeline
  aus, sodass eine Einstellung Samples nur dort verändert, wo diese Pipeline sie
  ausliest.
keywords:
  - yolo datenaugmentierung
  - mosaic augmentierung
  - mixup
  - hsv augmentierung
  - zufällige affine transformation
  - random erasing
  - erkennung kleiner objekte
  - augmentierung beispiele
  - copy paste augmentierung
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
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
  small:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # YOLOX startet mit mosaic_scale=(0.1, 2.0), was ein Objekt auf ein

        # Zehntel seiner Größe schrumpfen kann. (0.5, 1.5) stoppt bei der
        Hälfte.

        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # In der CLI ist ein Paar zwei durch Komma getrennte Zahlen.
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
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
    - label: Klassifikationspaket
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
source_hash: 72d22a38d647de0a
---

## Festlegen der Einstellungen

Die Augmentierungseinstellungen sind gewöhnliche Argumente von `train()`.

<code-tabs name="train" />

Drei davon haben kürzere Schreibweisen, die `train()` und die CLI gleichermaßen akzeptieren: `mosaic` wird auf `mosaic_prob` abgebildet, `fliplr` auf `flip_prob` und bei der Objekterkennung `mixup` auf `mixup_prob`. Übergibst du eine kurze Schreibweise und ihr Feld mit unterschiedlichen Werten, wird ein Fehler ausgelöst. `perspective` hat in 1.8.0 kein CLI-Flag und wird daher aus Python gesetzt.

Die Klassifikationseinstellungen werden beim Aufbau ihrer Pipeline validiert, und die CLI prüft zusätzlich `flip_prob`, `flipud`, `cutmix` und `erasing`. `TrainConfig` prüft die Wertebereiche der Einstellungen für die Objekterkennung nicht, daher trainiert `degrees=100` mit Drehungen von bis zu 100 Grad. Die [Augmentierungsmatrix](/docs/reference/augmentation-matrix) nennt den Wertebereich jeder Einstellung.

## Drei statt zwei Zustände

Ob eine Einstellung etwas bewirkt, hängt von der Familie ab. Die Bibliothek führt dazu eine deklarative Tabelle, und jeder Eintrag hat einen von drei Zuständen.

`used` bedeutet, dass die Einstellung die Pipeline erreicht und Samples verändert. `ignored` bedeutet, dass sie die Pipeline nie erreicht, das Setzen also nichts bewirkt. `gated_by_mosaic` bedeutet, dass sie nur für Samples gilt, die den Mosaic-Zweig durchlaufen haben, sodass sie bei `mosaic_prob=0` nie ausgelöst wird, obwohl sie verdrahtet ist.

Dieser dritte Zustand ist der, der überrascht. In einer Pipeline im YOLOX-Stil läuft die affine Transformation auf dem Mosaic-Canvas, und MixUp mischt ein Mosaic-Sample, daher deaktiviert `mosaic_prob=0` stillschweigend `degrees`, `translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` und `mixup_scale` auf einen Schlag. Der Trainer protokolliert speziell für den MixUp-Fall eine Warnung:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

Auch die CLI warnt vor ignorierten Einstellungen und nennt nur die, die du tatsächlich eingegeben hast:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Beim Aufruf aus Python warnt `train()` nicht vor ignorierten Einstellungen. Die [Augmentierungsmatrix](/docs/reference/augmentation-matrix) enthält den Zustand jeder Einstellung für jede Familie und den Wert, mit dem jede Familie startet. Die Tabelle ist die ausgelieferte Spezifikation unter `libreyolo/data/augment/spec.py`, die die Tests der Bibliothek selbst gegen die Pipelines prüfen, und sie lässt sich direkt auslesen:

<code-tabs name="support" />

## Wirkung der einzelnen Einstellungen

Die folgenden Abbildungen erzeugen die Trainingstransformationen von LibreYOLO 1.8.0 selbst, angewendet auf das Beispielbild über einen Datensatz aus einem einzigen Bild, sodass jede Mosaic-Kachel und jeder MixUp-Partner dasselbe Foto ist. Die Box ist die Person, die LibreYOLO9t auf diesem Bild erkennt. Unter jedem Teilbild stehen der eingestellte Wert und die Größe der größten Box, in Pixeln bei der Trainingsauflösung von 640 Pixeln. Ohne Augmentierung misst die Person 291 × 217 px.

Mehrere geometrische Einstellungen sind Bereiche: `degrees=10` bedeutet einen Winkel, der irgendwo zwischen -10 und +10 Grad gezogen wird. Ihre Teilbilder fixieren den Zufallswert auf den Wert in der Bildunterschrift und zeigen daher die stärkste Wirkung, die ein Wert zulässt, statt eines typischen Samples. Die Abbildungen zur Objekterkennung stammen aus der YOLOv9-Pipeline. YOLOv7 trainiert unverändert über sie, und RT-DETR und RT-DETRv2 verwenden ihren Mosaic-Wrapper mit einer eigenen Transformation je Sample. YOLOX, RTMDet, PicoDet und FOMO bauen das Mosaic und die Transformation auf dieselbe Weise in einem separaten Wrapper auf, dessen MixUp sich unterscheidet.

### mosaic_prob

Die Wahrscheinlichkeit, dass ein Sample ein Mosaic ist: vier Bilder auf einem Canvas mit doppelter Eingabegröße, das die affine Transformation anschließend auf die Eingabegröße abbildet. In den Pipelines im YOLOX-Stil laufen die Transformation und MixUp nur auf Mosaic-Samples, daher entfallen beide, wenn dieser Wert 0 ist. RT-DETR, RT-DETRv2, PicoDet und FOMO setzen ihn standardmäßig auf 0.

<aug-figure name="mosaic" />

Das zweite Teilbild fixiert die Skalierung der Transformation auf 0.5×, bei der das gesamte Canvas ins Bild passt. Die letzten beiden sind gewöhnliche Zufallsziehungen mit den Standardwerten von YOLOv9, und die Bildunterschrift nennt die jeweils gezogene Skalierung.

### mosaic_scale

Der Bereich, aus dem der Zoom der Transformation gezogen wird. Bei 1× behält ein Objekt die Größe, die es bei der Inferenz hat, und bei 0.5× ist es halb so groß. Der Basis-Standardwert ist `(0.1, 2.0)`, den YOLOX, YOLOv7, RTMDet, PicoDet, RT-DETR, RT-DETRv2 und FOMO beibehalten, sodass ihre Objekte auf ein Zehntel ihrer Größe schrumpfen können. YOLOv9 verwendet standardmäßig `(0.5, 1.5)`. YOLO-NAS und PP-YOLOE haben kein Mosaic und verwenden diesen Bereich als Skalierung ihrer affinen Transformation je Sample wieder.

<aug-figure name="mosaic_scale" />

Oberhalb von 1× zeigt die Ausgabe nur den oberen linken Teil des Canvas, sodass Objekte am Rand abgeschnitten werden oder das Bild verlassen.

### degrees

Der Drehbereich: Jedes transformierte Sample dreht sich um einen Winkel, der zwischen `-degrees` und `+degrees` gezogen wird. Die Drehung erfolgt um die obere linke Ecke des Canvas, sodass ein großer Winkel einen Großteil des Bildes aus dem Bildausschnitt dreht. YOLOX und YOLOv7 behalten den Basis-Standardwert 10, und YOLOv9 verwendet standardmäßig 0.

<aug-figure name="degrees" />

### translate

Der Verschiebungsbereich, als Anteil der Eingabeseite: Jedes transformierte Sample verschiebt sich bei `imgsz=640` auf jeder Achse um bis zu `translate` mal 640 px. Der Basis-Standardwert ist 0.1. YOLO-NAS und PP-YOLOE verwenden standardmäßig 0.25, RTMDet, PicoDet und FOMO 0.

<aug-figure name="translate" />

### shear

Der Scherbereich in Grad, für jede Achse getrennt gezogen. YOLOX und YOLOv7 behalten den Basis-Standardwert 2, und jede andere Familie, die diese Einstellung ausliest, verwendet standardmäßig 0.

<aug-figure name="shear" />

### perspective

Eine projektive Transformation zusätzlich zu Drehung, Skalierung, Scherung und Verschiebung. Jede Ecke des Canvas wandert pro Achse um einen zufälligen Betrag von bis zu `min(perspective * 100, 0.2)` mal die halbe Canvas-Seite nach innen, sodass `perspective=0.001` eine Ecke bei jeder Auflösung um höchstens 5 Prozent der Seite verschiebt und jeder Wert ab `0.002` dasselbe Maximum erreicht. Der Standardwert `0` hält die Transformation affin, und ein negativer Wert löst `ValueError` aus, sobald das erste Sample transformiert wird. Die Teilbilder teilen sich eine Zufallsziehung, skaliert mit dem Wert. YOLOv9 samt E2E- und P2-Varianten, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR und RT-DETRv2 wenden sie an.

<aug-figure name="perspective" />

### hsv_prob

Die Wahrscheinlichkeit eines HSV-Farb-Jitters. Nur die Wahrscheinlichkeit ist eine Einstellung. Der Jitter verschiebt den Farbton um bis zu 5 auf der OpenCV-Farbtonskala von 0 bis 179 und Sättigung und Hellwert um bis zu 30 auf der Skala von 0 bis 255, und jede der drei Verschiebungen wird mit der Wahrscheinlichkeit einhalb übersprungen. Jede Objekterkennungs-Pipeline, die `hsv_prob` ausliest, verwendet diese Beträge. Die letzten drei Teilbilder sind die sichtbarsten von 119 Zufallsziehungen.

<aug-figure name="hsv" />

### flip_prob und flipud

Die Wahrscheinlichkeiten für horizontale und vertikale Spiegelung. `flip_prob` startet in jeder Familie bei 0.5, außer bei FOMO, das standardmäßig 0 verwendet. `flipud` startet bei 0. RTMDet, PicoDet, RT-DETR, RT-DETRv2, FOMO, PP-YOLOE und die Familien im DETR-Stil haben keine vertikale Spiegelung und ignorieren diese Einstellung. Das Rezept von PP-YOLOE verwendet stattdessen eine zufällige 90-Grad-Drehung.

<aug-figure name="flip" />

### mixup_prob und mixup_scale

MixUp mischt ein zweites Sample in das aktuelle ein und behält die Boxen beider. In der YOLOv9-Pipeline ist der Partner ein ganzes Sample ohne Mosaic, das Mischverhältnis wird gleichverteilt zwischen 0 und 1 gezogen, und `mixup_scale` wird nie ausgelesen, obwohl die Spezifikation der Bibliothek es als an Mosaic gekoppelt führt. Im YOLOX-Wrapper, den YOLOX, RTMDet, PicoDet und FOMO verwenden, und in dem von YOLO-NAS wird der Partner zuerst um einen aus `mixup_scale` gezogenen Faktor neu skaliert, und die beiden Bilder werden im festen Verhältnis von 0.5 gemittelt. Der Basis-Standardwert für `mixup_prob` ist 1.0, und YOLOv9 verwendet standardmäßig 0.

<aug-figure name="mixup" />

Die letzten beiden Teilbilder mischen ein zweites Sample in ein Mosaic bei 0.5× ein. Ihre Bildunterschrift nennt das Mischverhältnis λ, das Gewicht des Mosaic-Samples.

### no_aug_epochs

`no_aug_epochs` ist überall `used`, bedeutet aber nicht überall dasselbe. In den Mosaic-Pipelines schaltet es Mosaic und MixUp für die letzten Epochen ab. In den Pipelines im DETR-Stil beendet es die fotometrischen, Zoom-out- und Zuschnitt-Augmentierungen und formt das Ende des Zeitplans. In Klassifikationspipelines deaktiviert es Auto-Augmentierung, Erasing, MixUp und CutMix, während Zuschnitt und Spiegelungen erhalten bleiben.

## Rezepte im DETR-Stil

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4 und TinyFormer trainieren über eine einzige torchvision-Transformation ohne Mosaic und ohne affine Transformation, ebenso EdgeCrafter (EC) bei der Objekterkennung. Die drei starken Operationen dieser Transformation sind Rezeptkonstanten statt Einstellungen:

- `RandomPhotometricDistort` wendet jede Farbänderung (Helligkeit 0.875 bis 1.125, Kontrast und Sättigung 0.5 bis 1.5, Farbton ±0.05 und eine Kanalvertauschung) mit der Wahrscheinlichkeit 0.5 an.
- `RandomZoomOut` legt das Bild mit der Wahrscheinlichkeit 0.5 auf ein schwarzes Canvas mit bis zu vierfacher Seitenlänge, sodass Objekte auf ein Viertel ihrer Größe schrumpfen können.
- `RandomIoUCrop`, angewendet mit der Wahrscheinlichkeit 0.8, schneidet auf einen Bereich zwischen 0.3 und 1.0 jeder Seite zu, der so gewählt wird, dass zumindest ein Teil der Boxen erhalten bleibt.

Nach diesen Schritten werden Boxen verworfen, bei denen eine Seite 1 px oder weniger misst. Alle drei Operationen enden mit der Stopp-Epoche der Familie, spätestens wenn die letzten `no_aug_epochs` Epochen beginnen. Die einzigen Einstellungen, die diese Pipeline ausliest, sind `flip_prob` und `no_aug_epochs`.

<aug-figure name="detr_ops" />

Drei Samples vom Beginn eines D-FINE-Laufs, mit allen drei Operationen aktiv:

<aug-figure name="detr_samples" />

RF-DETR hat eine eigene Pipeline: eine horizontale Spiegelung, dann mit der Wahrscheinlichkeit `crop_resize_prob` eine Skalierung der kürzeren Seite auf 400, 500 oder 600 px und ein zufälliger quadratischer Zuschnitt von 384 bis 600 px und schließlich eine direkte Skalierung auf die quadratische Trainingsauflösung, die das Bild streckt, statt es per Letterbox einzupassen. `crop_resize_prob` ist standardmäßig 0.5 und wird aus Python gesetzt.

<aug-figure name="rfdetr_crop" />

## Kleine Objekte

`erasing` gilt nicht für die Objekterkennung. Random Erasing gehört zum Klassifikationspaket, und keine Pipeline für Objekterkennung hat einen Erasing-Schritt, sodass ein Detektor mit jedem Wert davon gleich trainiert und es nichts abzuschalten gibt. Diese Schritte verkleinern oder entfernen kleine Objekte:

- `mosaic_scale`. In den Mosaic-Pipelines wird die Größe eines Objekts mit der gezogenen Skalierung multipliziert, sodass die Untergrenze festlegt, wie klein es werden kann. Beim Basis-Standardwert `(0.1, 2.0)` kann ein 20 px großes Objekt am Ende 2 px groß sein. Die YOLOv9-Pipeline verwirft dann Boxen, bei denen eine Seite 2 px oder weniger misst, und ihre letzte Transformation verwirft, wie die von YOLOX, Boxen, bei denen eine Seite 1 px oder weniger misst. Die Pixel des Objekts bleiben ohne Label im Bild. Eine Untergrenze von 0.5, wie bei `(0.5, 1.5)` von YOLOv9, hält jedes Objekt bei mindestens der Hälfte seiner Inferenzgröße.
- `degrees`, `translate`, `shear` und `perspective` schieben Objekte an den Rand, wo ihre Boxen beschnitten werden und denselben Größenfilter durchlaufen.
- In den Rezepten im DETR-Stil kann `RandomZoomOut` Objekte auf ein Viertel ihrer Größe verkleinern, und `RandomIoUCrop` kann sie wegschneiden. Keins von beiden ist eine Einstellung, und beide enden mit der Stopp-Epoche.
- Der Zuschnitt von RF-DETR kann Objekte an seinem Rand anschneiden. `crop_resize_prob=0` schaltet ihn ab.
- Die Trainingsauflösung skaliert jedes Objekt: Die Letterbox skaliert ein Bild um `imgsz` geteilt durch seine längere Seite, sodass bei `imgsz=1280` jedes Objekt die doppelte Breite und Höhe hat wie bei 640. [YOLOv9](/docs/models/yolov9) hat außerdem eine P2-Variante mit einem Head mit Stride 4 für kleine Objekte.

<code-tabs name="small" />

## Klassifikationspaket

Sechs Einstellungen wirken nur auf die Klassifikationspipeline. Familien für Objekterkennung ignorieren alle sechs.

<code-tabs name="classify" />

`scale` ist der Flächenbereich des zufälligen Zuschnitts. `scale=0.5` bedeutet `(0.5, 1.0)`, und ein explizites Paar setzt beide Grenzen. Eine kleine Untergrenze kann das Objekt vollständig wegschneiden, wie im dritten Teilbild.

<aug-figure name="cls_crop" />

`auto_augment` akzeptiert `"randaugment"`, `"autoaugment"`, `"augmix"` oder `None` und wendet eine Zufallsziehung dieser Richtlinie auf jedes Sample an.

<aug-figure name="auto_augment" />

`erasing` ist die Wahrscheinlichkeit von Random Erasing, in `[0, 1)`. Es läuft nach der Normalisierung und füllt ein Rechteck, das 2 bis 33 Prozent des Bildes bedeckt, mit Nullen, was nach der Normalisierung der mittleren ImageNet-Farbe entspricht.

<aug-figure name="erasing" />

`mixup` und `cutmix` sind Wahrscheinlichkeiten je Batch, die weiche Labels erzeugen. Höchstens eine davon läuft je Batch, MixUp zuerst, daher sind die beiden additiv und dürfen zusammen höchstens 1 ergeben. In den Teilbildern sind A und B zwei Zuschnitte desselben Fotos, und jede Bildunterschrift nennt das weiche Label, mit dem das gemischte Bild trainiert wird.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup` und `cutmix` sind standardmäßig aus, sodass das Klassifikationstraining unverändert bleibt, solange du sie nicht anforderst. `train()` und die CLI leiten `mixup` nach Aufgabe weiter: Die Klassifikation verwendet Batch-Mischung, die Objekterkennung dagegen `mixup_prob`.

`crop_pct=None` behält das Skalierungsverhältnis der Familie für die Auswertung bei. Ein abweichender Wert ändert den Auswertungszuschnitt des Validierungsdurchlaufs, der beim Training läuft; der Export behält dagegen die native Vorverarbeitung der Familie bei.

Die CLI stellt `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`, `fliplr` und `flipud` bereit. Bei der Klassifikation bedeutet `mixup` Batch-Mischung und ist standardmäßig 0.0. Horizontale Spiegelungen haben den Standardwert 0.5, vertikale 0.0.

`no_aug_epochs` deaktiviert Auto-Augmentierung, Erasing, MixUp und CutMix für die letzten Epochen und behält Zuschnitt und Spiegelungen bei. Klassifikationsrezepte setzen diesen Endabschnitt standardmäßig auf 0.

## Familienspezifische Einstellungen

Einige Einstellungen liegen in der Konfigurationsunterklasse einer Familie statt in der Basisklasse, existieren daher nur für diese Familie und haben kein CLI-Flag.

| Familie | Einstellung | Wirkung |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Wahrscheinlichkeit der Copy-Paste-Instanzaugmentierung, nur bei `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` verwendet dasselbe Sample gespiegelt wieder, `"mixup"` holt ein zweites Sample |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Wahrscheinlichkeit einer zufälligen 90-Grad-Drehung |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | Obergrenze der Ground-Truth-Einträge je Bild in den Trainingstransformationen, Standardwert 300 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-Paste bei `task="segment"`, nur im Modus `"flip"` |
| RF-DETR | `crop_resize_prob` | Wahrscheinlichkeit für zufälliges Zuschneiden und Skalieren in der nativen Pipeline, Standardwert 0.5 |
| D-FINE, EdgeCrafter | `crop_resize_prob` | Wahrscheinlichkeit für zufälliges Zuschneiden und Skalieren, nur bei `task="segment"` |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Wahrscheinlichkeiten für Jitter im Pose-Pfad und für eine Keypoint-bewusste affine Transformation |

`max_labels` ist die Einstellung, durch die Daten verloren gehen. Boxen jenseits der Obergrenze werden mit einer einmaligen Warnung statt eines Fehlers verworfen, daher muss der Wert bei dichten Bildern wie Luftaufnahmen erhöht werden. Der Speicherbedarf der Label-Zuweisung richtet sich nach dem dichtesten Bild im Batch, nicht nach der Obergrenze.

Mosaic und MixUp sind beim Training mit orientierten Boxen unabhängig von den Einstellungen deaktiviert, weil eine eckenbewusste Augmentierung für gedrehte Boxen nicht implementiert ist.

## Verwandte Themen

- [Augmentierungsmatrix](/docs/reference/augmentation-matrix) für den Zustand jeder Einstellung in jeder Familie, die Wertebereiche und die Standardwerte jeder Familie.
- [Hyperparameter](/docs/train/hyperparameters) für `no_aug_epochs` als Argument des Zeitplans und die übrigen Argumente von `train()`.
- [Datensätze](/docs/train/datasets) für die Labelformate, die diese Transformationen verarbeiten.
