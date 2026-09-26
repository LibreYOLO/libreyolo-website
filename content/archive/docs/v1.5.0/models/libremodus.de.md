---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreMODUS in LibreYOLO: Any-to-Any-Bildanalyse'
description: >-
  Nutze LibreMODUS in LibreYOLO für Tiefe, Normalen, Kanten und Erkennung und
  kombiniere sie mit any2any(). Reine Inferenz, Gewichte werden von EPFL-VILAB
  geladen.
lead: >-
  LibreMODUS ist eine reine Inferenzintegration des Checkpoints MODUS 14B-A7B,
  eines Any-to-Any-Modells, das eine bildabgeleitete Eingabe in eine andere
  umwandelt: RGB rein, Tiefe raus, Tiefe rein, Normalen raus, eine dieser
  Eingaben plus eine Phrase, Boxen raus. LibreYOLO unterstützt vier Aufgaben
  über die Standard-API für Vorhersagen und eine größere Auswahl über any2any().
keywords:
  - libremodus
  - modus bildmodell
  - any-to-any bildanalyse
  - tiefenschätzung
  - oberflächennormalen
  - kantenerkennung
  - referring detection
  - epfl vilab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreMODUS


        model = LibreMODUS(size="14b-a7b", task="normal")

        result = model.predict("room.jpg")

        normals = result.normal_map.data


        model.set_task("edge")

        result = model.predict("room.jpg")

        edges = result.edges.data


        # Ohne eigenes Vokabular dekodiert detect die COCO-Label-Tokens des
        Checkpoints

        # zu zusammenhängenden COCO-80-Klassen-IDs.

        model.set_task("detect")

        result = model.predict("street.jpg")

        print(result.boxes.xyxy)
    - label: Phrasen-Grounding
      language: python
      code: >
        from libreyolo import LibreMODUS


        model = LibreMODUS(task="detect")

        # set_classes() schaltet die Erkennung auf Phrasen-Grounding um: Jede
        Phrase

        # läuft einzeln und wird über denselben Boxes-Vertrag zurückgegeben.

        model.set_classes(["red bus", "cyclist"])

        result = model.predict("street.jpg", conf=0.2)

        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: >
        from libreyolo import LibreMODUS


        model = LibreMODUS()


        # Eine bis drei bildabgeleitete Eingaben (rgb, depth, normal,
        canny/edge)

        # plus optionaler Hilfstext werden zu einem Ziel kombiniert.

        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )

        normals = result.normal_map.data


        # Grounding über any2any() benötigt eine Texteingabe mit der Phrase.

        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )

        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## Installation

LibreMODUS benötigt ein eigenes Zusatzpaket. Dieses installiert `accelerate`
für die von diesem Checkpoint benötigte Verteilung großer Modelle.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO verteilt oder spiegelt keine MODUS-Gewichte. Standardmäßig lädt ein
`LibreMODUS`-Modell die benötigten Dateien direkt aus `EPFL-VILAB/MODUS` bei
einer festgeschriebenen Hugging-Face-Revision. Ein neuer Download benötigt
immer das eigene authentifizierte Hugging-Face-Konto des Nutzers, selbst wenn
die Upstream-Zugangsschranke vorübergehend offen ist. Lies und akzeptiere die
Upstream-Bedingungen und authentifiziere dich anschließend:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Um jede Netzwerkanfrage zu vermeiden, verweise auf einen bereits vorhandenen
Snapshot:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

Dieses Verzeichnis muss `model.safetensors`, `ae.safetensors`,
`llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json`
und `merges.txt` enthalten. Unter Lizenzierung findest du die erlaubte Nutzung
des Checkpoints.

## Vorhersage

<code-tabs name="predict" />

Die Standard-Aufgaben-API umfasst vier Aufgaben, die jeweils einem
MODUS-Ziel entsprechen: `depth` für relative Tiefe (`result.depth_map`),
`normal` für Oberflächennormalen (`result.normal_map`), `edge` für Kanten im
Canny-Stil (`result.edges`) und `detect` für COCO-80-Boxen (`result.boxes`),
sofern `set_classes()` nicht auf Phrasen-Grounding umschaltet. Mit `set_task()`
wechselst du auf demselben geladenen Modell zwischen ihnen. Das veröffentlichte
Rezept verwendet zehn Flow-Sampling-Schritte mit Text-Guidance 4.0 und
Bild-Guidance 2.0. Beim Erstellen kannst du sie mit `inference_steps=`,
`inference_cfg=` und `inference_image_cfg=` überschreiben.

`any2any()` stellt die größere öffentliche Analyseoberfläche bereit: eine bis
drei bildabgeleitete Eingaben (`rgb`, `depth`, `normal`, `canny`/`edge`) plus
optionaler Hilfstext, kombiniert zu einem beliebigen Ziel aus Tiefe, Normalen,
Kanten, von SAM abgeleiteten Kanten, COCO-Erkennung oder Phrasen-Grounding.
Alle bildabgeleiteten Eingaben müssen dieselbe ausgerichtete Arbeitsfläche
beschreiben. LibreMODUS lehnt unterschiedliche Breiten und Höhen ab, statt sie
unabhängig zu skalieren. `chain=(...)` erzeugt Zwischenziele und speist sie im
Drei-Bedingungen-Trainingsbudget des Checkpoints in denselben Kontext zurück.
`verify=N` (N >= 2) erzeugt N Kandidaten und behält den mit der höchsten
Bewertung in einer eingeschränkten Selbstkonsistenzprüfung. Die Bewertung ist
als `result.verification_score` verfügbar.

`dtype="bf16"` (der Standardwert) entspricht der Precision des veröffentlichten
Checkpoints. `dtype="fp8"` speichert geeignete lineare Gewichte des
Decoder-Stamms als E4M3 mit einer Skala pro Ausgabekanal, konvertiert sie
einmalig in einen lokalen Cache unter `~/.cache/libreyolo/modus/fp8` und
dequantisiert sie bei jeder Matrixmultiplikation in den Eingabedatentyp. Damit
wird Speicher eingespart, ohne die Aktivierungs-Accuracy einzutauschen.

`train()`, `val()` und `export()` lösen alle einen Fehler aus: LibreMODUS ist
reine Inferenz, bietet keine Datensatzvalidierung und besitzt keinen Exportpfad
für ONNX, TensorRT oder TFLite. Vorhersagen im Batch und Test-Time Augmentation
werden ebenfalls nicht unterstützt. Jeder Aufruf verarbeitet ein Bild.

## Lizenzierung

<provenance-box>

LibreYOLO hostet oder spiegelt den MODUS-Checkpoint nirgendwo, auch nicht in
seiner eigenen Hugging-Face-Organisation. Beim Laden wird immer die
festgeschriebene Revision direkt aus EPFL-VILAB/MODUS abgerufen oder ein unter
`checkpoint_path` bereits auf dem Datenträger vorhandener Snapshot gelesen.

</provenance-box>

## Zitieren

<citation-block />
