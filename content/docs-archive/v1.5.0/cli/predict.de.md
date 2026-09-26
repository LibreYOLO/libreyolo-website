---
title: libreyolo predict
seo_title: Befehlsreferenz für libreyolo predict
description: >-
  Inferenz von der Kommandozeile aus starten: jedes Argument, sein aus der
  CLI-Definition gelesener Standardwert und die Flags, die verändern, was auf
  stdout landet.
lead: >-
  Führt ein geladenes Modell über eine Quelle aus und gibt die Vorhersagen aus.
  Die Quelle kann ein Bild, ein Verzeichnis, ein Video, eine URL oder ein
  Live-Stream sein; das Modell kann ein Checkpoint oder ein exportiertes
  Artefakt sein.
keywords:
  - libreyolo predict cli
  - yolo inferenz kommandozeile
  - libreyolo predict befehl
  - libreyolo predict argumente
  - yolo json ausgabe terminal
last_verified: 1.5.0
meta:
  - label: Befehl
    value: libreyolo predict
    mono: true
  - label: Erforderlich
    value: source
    mono: true
  - label: Ausgabe
    value: >-
      Vorhersagen auf stdout. Mit save=true annotierte Dateien unter
      runs/detect/predict
snippets:
  examples:
    - label: Basis
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Annotierte Bilder speichern
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Gefilterte Klassen, JSON auf stdout'
      language: bash
      code: >
        # Klasse 0 ist person in der COCO-Klassenliste des Checkpoints.

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## Synopsis

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Argumente sind `key=value`-Paare. Derselbe Befehl akzeptiert auch die
POSIX-Form, also sind `conf=0.4` und `--conf 0.4` austauschbar, und ein als
`save=true` geschriebener Boolean wird zu `--save`. Namen mit Unterstrich
akzeptieren beide Schreibweisen: `max_det=50` und `--max-det 50` erreichen
dieselbe Option.

`libreyolo detect predict ...` wird akzeptiert und verhält sich identisch; das
Task-Wort wird vor dem Parsen entfernt.

## Argumente

| Argument | Standard | Bedeutung |
|---|---|---|
| `source` | | Bildpfad, Verzeichnis oder URL. Erforderlich |
| `model` | `yolox-s` | Modellname oder Pfad |
| `conf` | `0.25` | Confidence-Schwellenwert |
| `iou` | `0.45` | IoU-Schwellenwert für NMS |
| `imgsz` | | Eingabebildgröße: `640` (quadratisch) oder `480x640` (Höhe x Breite). Ohne Angabe die eigene Eingabegröße des Modells |
| `classes` | | Nach Klassen-IDs filtern, z. B. `[0,2,5]`. Eine einzelne Ganzzahl wird akzeptiert |
| `max_det` | `300` | Maximale Anzahl Detektionen pro Bild |
| `half` | `false` | FP16-Inferenz (nur CUDA, erfordert Unterstützung durch das Modell) |
| `save` | `false` | Annotierte Bilder speichern |
| `batch` | `1` | Bilder pro Forward-Pass bei Verzeichnisquellen. Über 1 läuft echte gebatchte Inferenz auf Modellen, die sie unterstützen |
| `stream` | `false` | Ergebnisse inkrementell liefern. Für Webcams und Live-Streams automatisch aktiviert |
| `stream_buffer` | `false` | Jeden Live-Frame puffern, statt nur den neuesten zu behalten |
| `vid_stride` | `1` | Jeden N-ten Video- oder Live-Frame verarbeiten |
| `show` | `false` | Video- und Live-Ergebnisse anzeigen; `q` stoppt |
| `tiling` | `false` | Gekachelte Inferenz für große Bilder |
| `overlap_ratio` | `0.2` | Überlappungsverhältnis der Kacheln |
| `output_path` | | Expliziter Ausgabepfad. Sonst `project/name`, wenn `save=true` |
| `color_format` | `auto` | Eingabefarbe: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Ausgabeformat: `jpg`, `png`, `webp` |
| `device` | `auto` | Gerät: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Modell zur Gesichtserkennung (Pfad oder CLI-Name). Für Gaze-Modelle erforderlich |
| `gallery` | | Gesichtsgalerie als `.npz` aus `libreyolo enroll`, gegen die Gesichter identifiziert werden. Nur für Face-Embedding-Modelle |
| `gallery_threshold` | `0.4` | Cosinus-Schwellenwert für eine Identitätsübereinstimmung in der Galerie |
| `project` | `runs/detect` | Wurzelverzeichnis der Ausgabe |
| `name` | `predict` | Name des Experiments |
| `exist_ok` | `false` | Vorhandenes Ausgabeverzeichnis wiederverwenden |
| `json` | `false` | JSON-Ausgabe auf stdout |
| `quiet` | `false` | stderr unterdrücken |
| `verbose` | `false` | Ausführliche Ausgabe auf stderr |
| `help_json` | `false` | Befehlsschema als JSON ausgeben und beenden |

## Beispiele

<code-tabs name="examples" />

## Hinweise

Ein exportiertes Artefakt wird genauso geladen wie ein Checkpoint, deshalb sind
`model=weights/LibreYOLO9s.onnx` und `model=weights/LibreYOLO9s.engine` gültige
Werte für `model`. Drei Optionen werden auf diesen Runtimes abgelehnt statt
ignoriert: `tiling`, `overlap_ratio` und `output_file_format` beenden sich mit
`config_unsupported`, wenn ein Runtime-Backend sie nicht erfüllen kann.

`half` geht den umgekehrten Weg. Exportierte Runtimes nehmen es an und laufen in
FP16; native PyTorch-Inferenz protokolliert, dass es ignoriert wurde, und macht
in FP32 weiter.

Gaze-Modelle sind zweistufig und haben keinen eigenen Detektor, deshalb ist
`face_detector` für sie erforderlich. `gallery` gilt nur für Modelle, deren Task
`embed` ist; die Übergabe an etwas anderes beendet sich mit
`config_unsupported`.

stdout trägt Ergebnisse und sonst nichts; Fortschritt, Warnungen und Fehler
gehen an stderr. `json=true` gibt ein JSON-Objekt pro Aufruf aus, beim Streaming
eines pro Frame, jeweils mit `schema_version`. `quiet=true` legt stderr still.
Beides zusammen gibt einem maschinellen Leser einen sauberen stdout-Stream.

Der Exit-Code ist `0` bei Erfolg, `2` bei einem Nutzungs- oder
Konfigurationsfehler, `3`, wenn die Quelle nicht gefunden wird, `4`, wenn das
Modell nicht geladen werden kann, und `1` bei sonstigen Laufzeitfehlern.

`help_json=true` gibt die Parameter, Typen, Standardwerte und Flags des Befehls
als JSON aus, ohne etwas auszuführen. Das ist der verlässliche Weg, diese
Tabelle aus einer installierten Version zurückzulesen.

Verwandt: [`libreyolo val`](/docs/cli/val) für gemessene Metriken auf einem
Datensatz, [`libreyolo export`](/docs/cli/export), um die oben genannten
Runtime-Artefakte zu erzeugen.
