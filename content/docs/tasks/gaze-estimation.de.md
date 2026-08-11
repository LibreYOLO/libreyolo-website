---
title: Blickrichtungsschätzung
seo_title: Blickrichtungsschätzung in LibreYOLO
description: >-
  Schätze in LibreYOLO Pitch und Yaw des Blicks pro Gesicht. Sage aus Python
  oder über die CLI vorher, lies Winkel im Bogenmaß und exportiere den Gaze-Head
  nach ONNX.
lead: >-
  Die Blickrichtungsschätzung liefert für jedes Gesicht in einem Bild eine
  Blickrichtung. LibreYOLO modelliert sie als zweistufige Aufgabe: Zuerst läuft
  ein Gesichtsdetektor, dann liest ein Gaze-Head Pitch und Yaw aus jedem
  Gesichtsausschnitt, den dieser zurückgibt.
keywords:
  - blickrichtung schätzen python
  - eye tracking
  - pitch yaw blickrichtung
  - L2CS-Net
  - blickrichtung erkennen
  - kopfpose
  - libreyolo gaze task
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Ohne face_detector fällt die Vorhersage auf den mitgelieferten

        # OpenCV-Detektor zurück; außer dem Checkpoint lädt nichts.

        model = LibreYOLO("LibreL2CSr50.pt")

        result = model(SAMPLE_IMAGE)


        gaze = result.gaze

        print(gaze.pitch, gaze.yaw)              # Bogenmaß, eine Zeile pro
        Gesicht

        print(gaze.pitch_deg, gaze.yaw_deg)      # dieselben Winkel in Grad

        print(gaze.direction_3d)                 # (N, 3) Einheitsvektoren
    - label: CLI
      language: bash
      code: >
        # Anders als in Python hat die CLI keinen automatischen Rückfall:

        # Gaze-Modelle brauchen einen expliziten Gesichtsdetektor, und zwar

        # einen LibreYOLO-Detektor, dessen Boxen Gesichter sind.

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: Die Gesichtsquelle wählen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Gib dem Gaze-Head Boxen eines bereits gelaufenen Detektors.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Oder nenne einen der mitgelieferten Detektoren.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## Definition

Die Blickrichtungsschätzung liefert zwei Winkel pro Gesicht. `result.gaze` ist
ein `Gaze`-Payload der Form `(N, 2)`, Spalte 0 Pitch und Spalte 1 Yaw, im
Bogenmaß, Zeile für Zeile ausgerichtet an `result.boxes`, den erkannten
Gesichtsboxen. Die Konvention ist die von L2CS-Net: Positives Yaw dreht den
Blick zur linken Seite der abgebildeten Person, positives Pitch dreht ihn nach
unten.

Dasselbe Payload stellt `pitch_deg` und `yaw_deg` für Grad bereit sowie
`direction_3d`, einen `(N, 3)`-Einheitsvektor im Kameraframe mit den Spalten
`(x, y, z)`.

Weil die Aufgabe zweistufig ist, hängt eine Vorhersage an zwei Modellen.
Gesichter, die der Detektor übersieht, haben keine Gaze-Zeile, und Boxen, die er
schlecht setzt, erzeugen Winkel aus einem schlecht zugeschnittenen Gesicht. Der
kanonische Task-Key ist `gaze`; `gaze-estimation` normalisiert darauf.

## Modelle

[L2CS-Net](/docs/models/l2cs) ist die einzige Familie für diese Aufgabe. Sie
kombiniert einen ResNet-Trunk mit zwei parallelen Klassifikations-Heads über
Winkelbins, einem für Pitch und einem für Yaw, über 448x448 großen
Gesichtsausschnitten. Architektonisch werden fünf Backbone-Tiefen unterstützt,
und eine davon, das ResNet-50, hat einen veröffentlichten Checkpoint.

Die Gewichte tragen eine Lizenzbeschränkung. Sie sind auf Gaze360 trainiert,
dessen Lizenz nur Forschung und nicht kommerzielle Nutzung erlaubt und die
Weitergabe verbietet, deshalb spiegelt LibreYOLO für diese Familie nichts. Der
eine Checkpoint, den die Bibliothek automatisch holen kann, kommt direkt aus der
Google-Drive-Distribution der Autoren, über `gdown`, nachdem die Lizenzbedingungen
ausgegeben wurden. Lies [L2CS-Net](/docs/models/l2cs), bevor du sie ausrollst.

Dieser Downloadpfad braucht das Extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Ohne es gibt die Bibliothek eine Anleitung zum manuellen Download aus, statt die
Übertragung zu versuchen. Auf einem Checkpoint, den du bereits hast,
vorherzusagen und ihn zu exportieren braucht überhaupt kein Extra.

## Vorhersage

<code-tabs name="predict" />

Die Gesichtsquelle wird auf einem von drei Wegen gewählt. `face_boxes` übergibt
Boxen, die du bereits berechnet hast, und überspringt die Erkennung.
`face_detector` nimmt `"auto"`, `"haar"`, `"yunet"`, ein
LibreYOLO-Detektionsmodell oder ein schlichtes Callable und lässt sich im
Konstruktor oder pro Aufruf setzen. Bleibt es in Python ungesetzt, fällt die
Vorhersage auf den mitgelieferten Detektor von OpenCV zurück, ein nackter Aufruf
funktioniert also ohne jede Verkabelung. Auf OpenCV 4 ist das die Haar-Kaskade
aus dem Wheel, die überhaupt keinen Download braucht; auf OpenCV 5, wo die
Haar-API entfernt wurde, ist es YuNet, das einmalig eine kleine Modelldatei aus
dem OpenCV-Zoo holt.

Die CLI teilt diesen Rückfall nicht. `libreyolo predict` weist ein Gaze-Modell
ohne `face_detector=` zurück, und der Wert, den es nimmt, ist der Name eines
LibreYOLO-Detektors oder ein Checkpoint-Pfad. Siehe
[Vorhersage](/docs/predict) für Quellen, Streaming und den Umgang mit
Ergebnissen.

## Training

Keine Familie in dieser Aufgabe trainiert innerhalb von LibreYOLO.
`LibreL2CS.train()` löst einen Fehler aus: Trainiere im Upstream-Projekt
L2CS-Net und lade das entstandene State Dict hier.

## Validierung

Die Validierung gegen Gaze-Ground-Truth-Datensätze liegt außerhalb des Umfangs,
und `val()` löst einen Fehler aus, statt Metriken zurückzugeben, die es nicht
berechnet hat. Für diese Aufgabe gibt es kein `metrics/`-Dictionary. Werte
upstream aus, auf dem Datensatz, für den der Checkpoint trainiert wurde.

## Export

<code-tabs name="export" />

Der Export-Vertrag für Gaze deckt ONNX, TorchScript, ExecuTorch, TensorRT und
OpenVINO ab. Was die Bibliothek verlässt, sind allein der ResNet-Trunk und die
beiden Winkelbin-Heads: Der Graph nimmt einen vorverarbeiteten 448x448 großen
Gesichtsausschnitt und liefert rohe Yaw- und Pitch-Logits. Gesichtserkennung,
Zuschnitt, das Softmax, der Erwartungswert über die Bins und die Umrechnung in
Winkel bleiben alle in Python, in `libreyolo.models.l2cs.utils`. Siehe
[Export](/docs/export) für die Formate und ihre Argumente.
