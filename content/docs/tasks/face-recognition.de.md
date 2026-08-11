---
title: Gesichtserkennung
seo_title: Gesichtserkennung in LibreYOLO
description: >-
  Erkenne, bette ein und identifiziere Gesichter in LibreYOLO. Registriere eine
  Gallery, vergleiche zwei Bilder und matche über Kosinusähnlichkeit, aus Python
  oder über die CLI.
lead: >-
  Die Gesichtserkennung ist der embed-Task, angewandt auf Gesichter. Ein
  Detektor lokalisiert und richtet jedes Gesicht aus, ein Erkennungs-Head
  liefert pro Gesicht einen L2-normalisierten Vektor, und über die Identität
  entscheidet die Kosinusähnlichkeit gegen registrierte Referenzen statt eine
  feste Klassenliste.
keywords:
  - gesichtserkennung python
  - face embedding
  - gesichtsverifikation
  - gesichter datenbank abgleichen
  - arcface onnx
  - libreyolo embed task
  - kosinusähnlichkeit gesichter
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-*-Namen führen unabhängig von der Dateiendung zur
        # Face-Embedding-Familie und laden beim ersten Aufruf zusammen mit
        # dem Standard-Gesichtsdetektor aus der LibreYOLO-Hugging-Face-Org.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # (N, 4) Gesichtsboxen
        print(result.embeddings.data.shape)  # (N, D), eine Zeile pro Gesicht
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Zwei Bilder vergleichen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Führt Erkennung und Embedding auf beiden Bildern aus und vergleicht
        # das sicherste Gesicht. Kosinusähnlichkeit liegt in [-1, 1].
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: Eine Gallery registrieren und identifizieren
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # Name ist None unter dem Schwellenwert
    - label: Registrieren und identifizieren über die CLI
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: Eigene Gesichtsboxen mitbringen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes überspringt die Erkennung ganz; face_detector nimmt
        # ein Callable, ein LibreYOLO-Modell oder eine FaceDetector-Instanz.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## Definition

Die Gesichtserkennung liefert einen Vektor pro Gesicht, kein Label. Die
Vorhersage läuft in zwei Stufen: Ein Gesichtsdetektor lokalisiert jedes Gesicht
und seine fünf Landmarken, der Ausschnitt wird auf eine kanonische
112x112-Ausrichtung entzerrt, und ein Erkennungs-Head gibt ein L2-normalisiertes
Embedding aus.

`result.embeddings` ist ein `Embeddings`-Payload der Form `(N, D)`, zeilenweise
ausgerichtet an `result.boxes`, Zeile `i` beschreibt also das Gesicht in Box `i`.
Weil die Zeilen Einheitsvektoren sind, ist die Kosinusähnlichkeit ein
Skalarprodukt, und `embeddings.similarity()` berechnet sie in einem Aufruf gegen
ein anderes `Embeddings` oder gegen eine ganze Matrix.

Ein Gesicht zu benennen ist ein eigener Schritt. Eine `Gallery` hält benannte
Referenzvektoren; `gallery=` an `predict()` zu übergeben hängt
`result.identities` an, zeilenweise ausgerichtet an den Embeddings, mit einem
Namen und seinem besten Kosinus-Score pro Gesicht. Ein Gesicht unterhalb des
Match-Schwellenwerts behält `None` als Namen, und der nächstliegende Name
unterhalb des Schwellenwerts wird nie untergeschoben.

Der kanonische Task-Key der Bibliothek ist `embed`. `face-recognition`,
`facial-recognition`, `reid` und `face` normalisieren alle darauf,
`task="face-recognition"` und `task="embed"` wählen also dasselbe aus.
Gesichter sind die Region-Form dieser weiteren Aufgabe;
[Embeddings](/docs/tasks/embeddings) behandelt die Ganzbild- und die Textform,
die gemeinsame API aus `Embeddings`, `Identities` und `Gallery` sowie die
Modelle, die Vektoren erzeugen, ohne irgendetwas zu erkennen.

## Modelle

[LibreFaceRec](/docs/models/librefacerec) ist die Familie für diese Aufgabe. Sie
besteht aus zwei ONNX-Artefakten hinter einem Aufruf: `librefacerec-l.onnx`,
einem iResNet100-Erkennungs-Head mit 512-dimensionalen Embeddings, und
`librefacerec-det.onnx`, dem Standard-Gesichtsdetektor mit fünf Landmarken, aus
dem OpenCV-Zoo übernommen. Beide werden beim ersten Aufruf aus der
LibreYOLO-Hugging-Face-Org geladen. Jede andere ONNX-Datei nach
ArcFace-Konvention (ausgerichtet 112x112 hinein, `(N, D)` heraus) kann den
Erkennungs-Head ersetzen, indem du ihren Pfad statt eines
`librefacerec-*`-Namens übergibst.

Der Task-Key `embed` ist weiter gefasst als Gesichter.
[CLIP](/docs/models/clip), [SigLIP2](/docs/models/siglip2) und
[DINOv2](/docs/models/dinov2) unterstützen ebenfalls `task="embed"` und liefern
einen Vektor pro ganzem Bild, was Bildersuche ist und nicht Gesichtsidentität.
Sie teilen sich die API aus `Gallery` und `Embeddings`, der Ablauf aus
Registrieren und Matchen unten überträgt sich also, aber sie erkennen und
richten keine Gesichter aus.

Der Erkennungs-Head läuft über `onnxruntime`, das die Basisinstallation nicht
mitbringt:

```bash
pip install "libreyolo[onnx]"
```

## Vorhersage

<code-tabs name="predict" />

Sich selbst überlassen, lädt `predict()` den Standarddetektor herunter und
koppelt ihn an. `face_detector` ersetzt ihn durch ein Callable, ein
LibreYOLO-Detektionsmodell oder eine `FaceDetector`-Instanz und lässt sich im
Konstruktor oder pro Aufruf setzen. `face_boxes` umgeht die Erkennung mit Boxen,
die du bereits hast. Auf der CLI nimmt `face_detector=` einen `.onnx`-Pfad zu
einem Gesichtsdetektor oder den Namen eines LibreYOLO-Detektors.

`model.verify(image_a, image_b)` ist die Abkürzung für zwei Bilder: Sie bettet
in jedem das sicherste Gesicht ein und liefert
`{"similarity", "same_person", "threshold"}`. `model.embed(sources)` liefert
jede Gesichtszeile über ein oder mehrere Bilder hinweg, gestapelt zu einem
einzigen `(N_total, D)`-Tensor. Siehe [Vorhersage](/docs/predict) für Quellen,
Streaming und den Umgang mit Ergebnissen.

## Datensatzformat

Die Registrierung liest einen Ordner pro Identität. Der Ordnername wird zur
Identität, und jedes Bild darin steuert Referenzen für diesen Namen bei:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` durchläuft diesen Baum und schreibt eine `.npz`-Gallery. Eine
vorhandene Gallery-Datei wird an Ort und Stelle ergänzt statt ersetzt,
Identitäten können also über die Zeit dazukommen. Galleries sind über die
Embedding-Dimension und einen Dateifingerabdruck an die Gewichte gebunden, die
sie erzeugt haben; das Matching mit einem anderen Modell löst einen Fehler aus,
statt inkompatible Vektorräume zu vergleichen.

Standardmäßig steuert jedes Quellbild eine Referenzzeile bei, das sicherste
Gesicht, ein Porträt mit Umstehenden registriert also nur sein Motiv. Übergib
`select="all"` an `Gallery.enroll`, um jede zurückgegebene Zeile zu speichern.

## Training

Keine Familie in dieser Aufgabe trainiert innerhalb von LibreYOLO.
`LibreFaceEmbedder.train()` löst einen Fehler aus: Trainiere einen
Erkennungs-Head upstream, exportiere ihn nach ArcFace-Konvention zu ONNX und
lade die Datei über ihren Pfad.

## Validierung

Für diese Aufgabe gibt es keinen Datensatz-Validator, und `val()` löst einen
Fehler aus, statt etwas anderes vorzugeben. Die Verifikations-Accuracy wird an
gelabelten Bildpaaren mit `model.verify()` gemessen, wobei du `threshold`
durchfährst, um den gewünschten Arbeitspunkt zu wählen. Die
Identifikations-Accuracy wird gemessen, indem du eine Gallery registrierst und
auf zurückgehaltenen Bildern `result.identities.name` und
`result.identities.score` liest, wobei ein Name `None` als Zurückweisung zählt.

## Export

Der Erkennungs-Head ist bereits ein ONNX-Graph, es gibt also nichts zu
konvertieren: `LibreFaceEmbedder.export()` löst einen Fehler aus. Rolle die
`.onnx`-Datei direkt aus, oder zeige LibreYOLO darauf und überlasse der Familie
Erkennung, Ausrichtung und Normalisierung.
