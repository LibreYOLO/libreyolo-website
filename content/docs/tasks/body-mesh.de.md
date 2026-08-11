---
title: Body Mesh
seo_title: Body-Mesh-Rekonstruktion in LibreYOLO
description: >-
  Rekonstruiere in LibreYOLO ein parametrisches 3D-Körpermodell pro Person. Sage
  aus Personenboxen oder mit einem Detektor vorher und lies Vertices, Gelenke
  und Kameratranslation aus.
lead: >-
  Die Body-Mesh-Rekonstruktion macht aus einem einzelnen Bild und einer Menge
  von Personenboxen einen parametrischen 3D-Körper pro Person: Form- und
  Pose-Parameter, posierte Vertices, 3D-Gelenke und die Kameratranslation, die
  sie vor dem Objektiv platziert.
keywords:
  - human mesh recovery python
  - body mesh
  - 3d körperpose aus bild
  - SAM 3D Body
  - MHR
  - parametrisches körpermodell
  - libreyolo mesh task
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Diese Familie ist nicht in der LibreYOLO()-Factory registriert und
        # wird direkt konstruiert. model_path=None startet den
        # zugangsbeschränkten Hugging-Face-Download; ein String gilt als
        # vorhandener lokaler Checkpoint und wird nie geladen. CUDA nötig.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.body_model)      # die Parametrisierung dieser Tensoren
        print(meshes.vertices.shape)  # (N, V, 3), Kameraframe, Meter
        print(meshes.joints3d.shape)  # (N, J, 3)
        print(meshes.joints2d.shape)  # (N, J, 2), Pixel auf dem Quellbild
    - label: Mit einem Personendetektor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # person_detector nimmt einen fertigen LibreYOLO-Detektor, ein
        # Callable oder eine PersonDetector-Instanz. Kein Namenskürzel.
        detector = LibreYOLO("LibreYOLO9s.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## Definition

Die Body-Mesh-Rekonstruktion liefert pro Bild ein `Meshes`-Payload, zeilenweise
ausgerichtet an `result.boxes`: Zeile `i` beschreibt die Person in Box `i`, also
derselbe Vertrag, den der Pose-Task für Keypoints nutzt.

Alles ist im Kameraframe des Originalbildes ausgedrückt. `transl` ist metrisch,
in Metern, wobei +z von der Kamera weg zeigt. `vertices` und `joints3d` sind
metrisch und enthalten `transl` bereits, müssen also nicht weiter zusammengesetzt
werden. `joints2d` liegt in Pixeln auf der ursprünglichen Bildfläche, nicht auf
dem Crop, den das Netz gesehen hat. `faces` hält die Mesh-Topologie einmal für
das ganze Bild statt einmal pro Zeile, weil sie jede Person teilt. In dieser
Version gibt es keinen Welt- oder Schwerkraftframe, und kein Feld tritt still an
seine Stelle.

Die Parameterlayouts unterscheiden sich zwischen Körpermodellen, deshalb ist an
den Formen nichts fest: `body_model` benennt die Parametrisierung, und die
Anzahlen werden aus den Tensoren zurückgelesen. Bei `"mhr"`, dem Momentum Human
Rig, sind Rotationen Eulerwinkel im Bogenmaß statt Achse-Winkel-Darstellung,
`body_pose` ist ein flacher Parametervektor pro Gelenk statt eines Tripels pro
Gelenk, und `betas` sind Koeffizienten der Identitäts-Blendshapes. Skelettskala,
Handpose und Gesichtsausdruck stecken in `extras`.

Der kanonische Task-Key ist `mesh`. `body-mesh`, `hmr` und
`human-mesh-recovery` normalisieren darauf.

## Modelle

[SAM 3D Body](/docs/models/sam-3d-body) ist die einzige Familie für diese
Aufgabe, und sie ist ein Wrapper und keine Portierung: Metas Paket `sam-3d-body`
steht unter der SAM License, von der der eigene Code von LibreYOLO nicht
abgeleitet sein darf, deshalb ist nichts davon eingebettet. Zwei Backbones
teilen sich dasselbe MHR-Körpermodell, `d3` auf einem DINOv3-ViT-H/16+-Encoder
und `h` auf dem ursprünglichen ViT-H.

Vor der ersten Vorhersage gelten drei Voraussetzungen, und keine davon ist
optional.

Das Upstream-Paket installierst du selbst, nicht LibreYOLO:

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

Zeige der Bibliothek den Klon über `sam_3d_body_path=` oder die
Umgebungsvariable `SAM_3D_BODY_PATH`. Wer diese Familie nie konstruiert, löst
den Import nie aus.

Der Checkpoint-Mirror ist zugangsbeschränkt. Akzeptiere die Lizenz auf der
Hugging-Face-Modellseite und melde dich mit `hf auth login` an, sonst schlägt
der erste Download fehl. Das MHR-Körpermodell selbst ist ein separates
Apache-2.0-Release, das von seiner eigenen öffentlichen Adresse geladen und
lokal zwischengespeichert wird.

Die Inferenz braucht ein CUDA-Gerät. Der Upstream-Estimator schiebt seinen Batch
ungeprüft auf die GPU, es gibt also keinen CPU-Pfad als Rückfallebene, und
`device="cpu"` löst einen Fehler aus.

## Vorhersage

<code-tabs name="predict" />

Personen erreichen das Modell auf zwei Wegen. `person_boxes` übergibt Boxen, die
du bereits hast, und zwar nur für ein einzelnes Bild: Ein fester Satz Boxen kann
Personen nicht über Videoframes hinweg verfolgen, deshalb löst das Argument bei
einer Videoquelle einen Fehler aus, statt still die Boxen des ersten Frames
wiederzuverwenden. `person_detector` nimmt einen fertigen LibreYOLO-Detektor,
ein Callable oder einen `PersonDetector` und ist der Weg für Video.
`focal_length` liefert ein bekanntes Kameraintrinsikum; bleibt es ungesetzt,
nutzt das Modell seine eigene Schätzung, und die meldet `meshes.focal_length`.

Diese Familie ist weder in die `LibreYOLO()`-Factory noch in den CLI-Befehl
`libreyolo predict` eingebunden. `LibreSAM3DBody` ist der einzige Einstiegspunkt.
Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und den Umgang mit
Ergebnissen.

## Training

Keine Familie in dieser Aufgabe trainiert innerhalb von LibreYOLO.
`LibreSAM3DBody.train()` löst einen Fehler aus: Trainiere im Upstream-Projekt
und lade den entstandenen Checkpoint hier.

## Validierung

Es gibt keinen Mesh-Validator, und `val()` löst einen Fehler aus. Die üblichen
Benchmarks stehen nur unter Forschungslizenz, deshalb ist keiner mitgeliefert
und keiner kann für dich geladen werden.

Die Metriken selbst gibt es als `libreyolo.validation.mesh_metrics`, um gegen
einen Datensatz auszuwerten, den du bereits hast. Die Funktion nimmt
vorhergesagte und Ziel-Gelenke, optional vorhergesagte und Ziel-Vertices, und
liefert ein Dictionary mit genau denselben Keys wie ein Validator:

`metrics/mpjpe` ist der mittlere Positionsfehler pro Gelenk nach Ausrichtung am
Wurzelgelenk, bewertet also die Pose und ignoriert, wo die Person in der Szene
steht. `metrics/pa_mpjpe` ist dieselbe Größe nach einer vollständigen
Procrustes-Ausrichtung aus Rotation, gleichmäßiger Skalierung und Translation,
was globale Orientierung und Körpergrößenfehler entfernt und die artikulierte
Pose übrig lässt. `metrics/pve` ist der mittlere Fehler pro Vertex über die
Mesh-Oberfläche nach Ausrichtung am Vertex-Schwerpunkt; anders als die
Gelenkmetriken reagiert er auf die Körperform, und er erscheint nur, wenn beide
Vertex-Arrays übergeben werden. Bei allen dreien ist kleiner besser. Die
Eingaben werden als metrisch, in Metern, angenommen, und `scale_to_mm` rechnet
die Ergebnisse in die Millimeter um, die in der Literatur berichtet werden.

## Export

Der Mesh-Export ist nicht implementiert. LibreYOLO hat für diese Aufgabe keinen
Metadatenvertrag für den exportierten Graph definiert, auch nicht dafür, wie das
MHR-Parameterlayout außerhalb von PyTorch mitgeführt wird, deshalb löst
`export()` einen Fehler aus, statt einen Graph auszugeben, dessen Ausgabe
niemand interpretieren könnte.
