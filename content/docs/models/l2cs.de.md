---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net: Blickschätzung in LibreYOLO'
description: >-
  Nutze L2CS-Net in LibreYOLO für die zweistufige Schätzung von Blickneigung und
  Blickrichtung. Installiere, verwende und exportiere das Modell. Der
  Gaze360-Checkpoint ist nur für Forschungszwecke bestimmt.
lead: >-
  L2CS-Net ist ein zweistufiger Blickschätzer: Ein Gesichtsdetektor lokalisiert
  Gesichter, anschließend sagt ein ResNet-Trunk mit zwei Klassifikations-Heads
  für Winkel-Bins die Neigung und horizontale Richtung pro Gesicht voraus.
  LibreYOLO bindet das Modell nur für die Inferenz ein.
keywords:
  - L2CS-Net
  - blickschätzung
  - eye tracking
  - pitch yaw
  - Gaze360
  - gesichtserkennung
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Kein face_detector angegeben: nutzt den mit OpenCV gebündelten
        # Detektor (Haar in OpenCV 4, YuNet in OpenCV 5). Außer dem
        # L2CS-Checkpoint selbst ist daher kein weiterer Download nötig.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Gesichtsquelle
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # L2CS Boxen eines bereits ausgeführten Detektors übergeben.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Oder einen bestimmten gebündelten Gesichtsdetektor angeben.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Exportierte Datei nutzen
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # Der exportierte Graph enthält nur den ResNet-Trunk und die beiden
        # Winkel-Bin-Heads. Er nimmt einen vorverarbeiteten 448x448-
        # Gesichtsausschnitt entgegen und gibt rohe (yaw_logits, pitch_logits)
        # statt decodierter Winkel aus. Softmax, Bin-Erwartungswert und
        # Gradumrechnung verbleiben in Python. Siehe
        # libreyolo.models.l2cs.utils.bin_logits_to_angles.
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## Installation

L2CS-Net benötigt keine zusätzlichen Abhängigkeiten, um ein Modell zu
erstellen, Vorhersagen damit auszuführen oder es zu exportieren, wenn du
bereits einen Checkpoint besitzt.

```bash
pip install libreyolo
```

Der einzige Checkpoint, den LibreYOLO automatisch abrufen kann, ist ein mit
Gaze360 trainiertes ResNet-50. Er wird über `gdown` statt über einen einfachen
HTTP-Spiegel heruntergeladen, weil er auf Google Drive des Autors und nicht in
der LibreYOLO-Organisation liegt. Dieser Pfad benötigt das Extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Ohne dieses Extra gibt LibreYOLO Anweisungen für den manuellen Download aus,
anstatt ohne Meldung fehlzuschlagen.

## Vorhersage

<code-tabs name="predict" />

L2CS-Net ist ein zweistufiger Schätzer: Zuerst wird ein Gesichtsdetektor
ausgeführt. Anschließend liest der Blick-Head Neigung und horizontale Richtung
aus jedem zurückgegebenen Gesichtsausschnitt. Ohne weitere Angaben greift die
Vorhersage auf den mit OpenCV gebündelten Detektor zurück. Sobald du den
L2CS-Checkpoint besitzt, funktioniert ein einfacher Aufruf daher ohne weiteren
Download. `face_boxes` akzeptiert Boxen eines bereits ausgeführten Detektors.
`face_detector` akzeptiert `"auto"`, `"haar"`, `"yunet"`, ein
LibreYOLO-Erkennungsmodell oder ein einfaches Callable. `result.gaze` enthält
Neigung und horizontale Richtung in Radiant. Seine Zeilen sind an
`result.boxes`, also die erkannten Gesichtsboxen, ausgerichtet. Unter
[Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und
Ergebnisverarbeitung.

## Varianten

Fünf Backbone-Tiefen verwenden dieselbe Eingabeauflösung und nehmen dieselben
Argumente entgegen. Gaze360, der Datensatz hinter dem einzigen veröffentlichten
Checkpoint, wurde mit einem ResNet-50 trainiert. Die anderen vier Tiefen werden
von der Architektur unterstützt, besitzen aber keine veröffentlichten Gewichte.

## Export

<export-matrix />

<code-tabs name="export" />

## Lizenzierung

<provenance-box>

LibreYOLO hostet oder spiegelt keinen L2CS-Checkpoint. Anders als bei den
meisten anderen Familien dieser Website befindet sich für diese Familie nichts
in der LibreYOLO-Organisation auf Hugging Face. Der einzige Checkpoint, den die
Bibliothek automatisch abrufen kann, stammt direkt aus der Google-Drive-
Distribution des Autors. Vor Beginn der Übertragung wird der Lizenzhinweis zu
Gaze360 angezeigt. Es handelt sich nicht um die in der obigen Zusammenfassung
angedeutete Kopie, die erneut auf `huggingface.co/LibreYOLO` veröffentlicht
wurde.

</provenance-box>

