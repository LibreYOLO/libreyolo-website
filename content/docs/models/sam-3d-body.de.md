---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: 'SAM 3D Body: vollständige Körper-Mesh-Rekonstruktion in LibreYOLO'
description: >-
  Nutze SAM 3D Body in LibreYOLO für die Rekonstruktion vollständiger
  menschlicher Körper-Meshes. Installiere und sage vorher. Metas SAM License
  beschränkt die Checkpoints, CUDA ist erforderlich.
lead: >-
  SAM 3D Body ist Metas Prompt-basiertes Modell zur Rekonstruktion eines
  vollständigen 3D-Körper-Meshes einschließlich Händen und Füßen aus einem
  einzelnen Bild und Personenboxen. LibreYOLO bindet das Upstream-Paket ein,
  statt es zu portieren.
keywords:
  - sam 3d body
  - human mesh recovery
  - körper mesh
  - mhr
  - momentum human rig
  - 3d pose
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Diese Familie ist nicht bei der LibreYOLO()-Factory registriert und
        wird

        # direkt erzeugt. model_path=None startet den beschränkten
        Hugging-Face-Download.

        # Ein String gilt als vorhandener lokaler Checkpoint-Pfad und wird nie
        automatisch

        # abgerufen. Die Inferenz benötigt CUDA; es gibt keinen CPU-Pfad.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.vertices.shape)    # (N, V, 3), Kamerakoordinaten, Meter

        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: Mit einem Personendetektor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Hier gibt es keinen Kurznamen: Übergib einen erzeugten
        LibreYOLO-Detektor,

        # ein einfaches Callable oder eine PersonDetector-Instanz.

        detector = LibreYOLO("LibreRFDETRn.pt")

        model = LibreSAM3DBody(None, size="d3", device="cuda")


        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## Installation

```bash
pip install libreyolo
```

Damit erhältst du nur den LibreYOLO-Adapter. SAM 3D Body selbst wird nicht
mitgeliefert, weil die Lizenz keine Ableitung des eigenen LibreYOLO-Codes
erlaubt. Klone das Upstream-Repository, installiere die Abhängigkeiten selbst
und verweise LibreYOLO anschließend auf den Klon.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

Alternativ kannst du die Umgebungsvariable `SAM_3D_BODY_PATH` setzen, statt
`sam_3d_body_path` bei jedem Aufruf zu übergeben. Wenn du diese Familie nie
erzeugst, wird der Import nie ausgelöst und du triffst nicht auf die SAM
License. Diese Familie ist weder an die Factory `LibreYOLO()` noch an den
CLI-Befehl `libreyolo predict` angeschlossen. `LibreSAM3DBody` ist der einzige
Einstiegspunkt.

## Vorhersage

<code-tabs name="predict" />

Der Checkpoint-Download ist zugangsbeschränkt. Du musst Metas Lizenz auf der
Hugging-Face-Modellseite akzeptieren und dich vor dem ersten Download mit
`hf auth login` authentifizieren. Die Inferenz selbst benötigt immer ein
CUDA-Gerät. Der Upstream-Schätzer verschiebt seinen Batch ohne Prüfung auf die
GPU. Auf einem reinen CPU-System wird daher ein Fehler ausgelöst, statt auf die
CPU zurückzufallen. `result.meshes` ist eine `Meshes`-Nutzlast, deren Zeilen
mit `result.boxes` ausgerichtet sind (eine Zeile pro erkannter Person).
`vertices` und `joints3d` sind metrisch und enthalten bereits die geschätzte
Kameratranslation. `joints2d` liegt in Pixeln des ursprünglichen Bildes vor.
Rotationen folgen der MHR-Konvention und verwenden Euler-Winkel statt
Achse-Winkel. Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming
und die Verarbeitung von Ergebnissen.

## Varianten

Es gibt zwei Backbones hinter demselben MHR-Körpermodell: `d3` verwendet einen
DINOv3-ViT-H/16+-Encoder, `h` den ursprünglichen ViT-H-Encoder.

## Export

<export-matrix />

Der Export von Körper-Meshes ist nicht implementiert. LibreYOLO hat noch
keinen Vertrag für einen exportierten Graphen der Mesh-Aufgabe definiert,
einschließlich der Darstellung des MHR-Parameteraufbaus außerhalb PyTorchs.

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Das von den Checkpoints gesteuerte Körpermodell MHR (Momentum Human Rig) ist
ein separates Meta-Release unter Apache-2.0. LibreYOLO ruft das
TorchScript-Artefakt zur Laufzeit aus dem eigenen öffentlichen Release von MHR
ab und speichert es lokal zwischen. Diese Datei wird nicht von LibreYOLO
gespiegelt und unterliegt ihren eigenen Apache-2.0-Bedingungen, nicht der SAM
License.

</provenance-box>

## Zitieren

<citation-block />
