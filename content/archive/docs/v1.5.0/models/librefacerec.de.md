---
title: LibreFaceRec
families:
  - facerec
seo_title: 'LibreFaceRec: Gesichtserkennung und Verifizierung'
description: >-
  Nutze LibreFaceRec in LibreYOLO für Gesichtserkennung, Embeddings und
  Verifizierung. Installiere und sage vorher. Die Embedding-Gewichte stehen
  unter Apache-2.0.
lead: >-
  LibreFaceRec ist die Gesichts-Embedding-Aufgabe von LibreYOLO: Ein
  Gesichtsdetektor lokalisiert und richtet Gesichter aus. Ein Erkennungs-Head
  erzeugt für Verifizierung oder Suche ein L2-normalisiertes
  Identitäts-Embedding.
keywords:
  - librefacerec
  - gesichtserkennung python
  - gesichts embedding
  - gesichter verifizieren
  - arcface
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # librefacerec-*-Namen führen unabhängig von der Dateiendung zu dieser
        Familie

        # und laden sie bei der ersten Verwendung zusammen mit dem
        Standarddetektor

        # für Gesichter von der LibreYOLO-Organisation auf Hugging Face
        herunter.

        model = LibreYOLO("librefacerec-l.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.embeddings.data.shape)   # (N, D), L2-normalisiert
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Verifizieren
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        # Vergleicht das auffälligste Gesicht beider Bilder anhand der
        Kosinusähnlichkeit

        # ihrer L2-normalisierten Embeddings.

        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)

        print(result["similarity"], result["same_person"])
    - label: Galeriesuche
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # Gesichter dieses Bildes
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # Kosinusähnlichkeiten der Form (query_faces, N_total).
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## Installation

Der Erkennungs-Head von LibreFaceRec läuft über `onnxruntime`, das nicht zur
Basisinstallation gehört.

```bash
pip install "libreyolo[onnx]"
```

## Vorhersage

<code-tabs name="predict" />

Hinter einem Aufruf stehen zwei getrennte ONNX-Graphen für Erkennung und
Wiedererkennung. Ein Gesichtsdetektor lokalisiert jedes Gesicht und richtet es
auf einen kanonischen Zuschnitt aus. Der Erkennungs-Head gibt pro Gesicht ein
L2-normalisiertes Embedding zurück. Ohne weitere Angaben lädt und verbindet
`predict()` den mitgelieferten Standarddetektor automatisch. `face_detector`
akzeptiert ein Callable, ein LibreYOLO-Erkennungsmodell oder eine
`FaceDetector`-Instanz. Mit `face_boxes` umgehst du die Erkennung vollständig
und übergibst bereits vorhandene Boxen. `result.embeddings` enthält eine Zeile
pro erkanntem Gesicht und ist an `result.boxes` ausgerichtet. Die Methode
`.similarity()` berechnet in einem Aufruf die Kosinusähnlichkeit gegenüber
einem anderen Embedding oder einer ganzen Galerie. Zum direkten Vergleich
zweier Bilder statt zweier bereits berechneter Embeddings führt
`model.verify(image_a, image_b)` Erkennung und Embedding für beide aus und
vergleicht das Gesicht mit der höchsten Confidence. Du kannst jedes andere
ONNX-Erkennungsmodell nach der ArcFace-Konvention (ausgerichteter Zuschnitt
rein, Embeddings der Form `(N, D)` raus) einsetzen, indem du seinen Dateipfad
statt eines `librefacerec-*`-Namens übergibst. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Export

<export-matrix />

LibreFaceRec bindet bereits einen vorab exportierten ONNX-Graphen ein. Ein
erneuter Export in ein anderes Format ist nicht implementiert.

## Lizenzierung

<provenance-box>

Der mitgelieferte Standarddetektor für Gesichter ist ein zweites Artefakt mit
einer zweiten Lizenz: YuNet aus OpenCV Zoo, MIT, Copyright Shiqi Yu. Aus keinem
der beiden Projekte wurde Architekturcode portiert. Beide Graphen werden über
`onnxruntime` als undurchsichtige Artefakte verwendet. Der eigene Wrapper von
LibreYOLO enthält daher keinen Drittanbietercode und steht vollständig unter MIT.

</provenance-box>

## Zitieren

<citation-block />
