---
title: Einstellungen
seo_title: LibreYOLO-Umgebungsvariablen und -Verzeichnisse
description: >-
  Alle von LibreYOLO gelesenen Umgebungsvariablen, die beschriebenen
  Verzeichnisse, erforderliche Tokens und Umschalter zur Auswahl des
  ausgeführten Codepfads.
lead: >-
  LibreYOLO besitzt keine Konfigurationsdatei. Verhalten, das nicht über ein
  Funktionsargument gesteuert wird, hängt von Umgebungsvariablen und einigen
  konventionellen Verzeichnissen ab. Alle sind hier aufgeführt.
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - libreyolo gewichte verzeichnis
  - libreyolo cache
last_verified: 1.5.0
verification: >-
  Variablen durch Suche nach os.environ und os.getenv in libreyolo/**/*.py für
  v1.5.0 ermittelt; Semantik an jeder Verwendungsstelle gelesen.
  Verzeichniskonventionen aus libreyolo/data/utils.py,
  libreyolo/utils/download.py, libreyolo/export/exporter.py,
  libreyolo/models/base/model.py und libreyolo/models/sam3dbody/mhr_body.py.
snippets:
  usage:
    - label: Datensatzstamm an einen anderen Ort legen
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Aufgelösten Wert in Python lesen
      language: python
      code: >
        from libreyolo.data import DATASETS_DIR


        # Standard ist ~/datasets; LIBREYOLO_DATASETS_DIR überschreibt ihn beim
        Import.

        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## Umgebungsvariablen

| Variable | Standardwert | Wirkung |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | Datensatzstamm. Wird beim Import einmal in `libreyolo.data.DATASETS_DIR` eingelesen |
| `LIBREYOLO_FASTER_COCO_EVAL` | nicht gesetzt | Überschreibt das Validierungs-Flag `faster_coco_eval`. `1`, `true`, `yes` oder `on` erzwingt das schnelle Backend, jeder andere Wert deaktiviert es, ohne Wert gilt das Konfigurations-Flag |
| `LIBREYOLO_KERNELS` | nicht gesetzt | Kernel-Auswahl. `off` oder `reference` erzwingt die Referenzimplementierungen, jeder andere Wert wählt nur unter diesem Namen registrierte Implementierungen aus |
| `LIBREYOLO_QUANT_KERNELS` | nicht gesetzt | Veralteter Alias für `LIBREYOLO_KERNELS`, wird nur gelesen, wenn diese Variable nicht gesetzt ist |
| `LIBREYOLO_HUB_KERNELS` | nicht gesetzt | `0`, `false`, `off` oder `no` deaktiviert das Laden von Hugging-Face-Hub-Kerneln. Jeder andere Wert einschließlich einer fehlenden Variable lässt es aktiviert |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | Speicherort des von der Aufgabe `mesh` verwendeten MHR-Körpermodells |
| `LIBRELABEL_ENABLE_LOCATE` | nicht gesetzt | Muss genau `1`, `true`, `yes` oder `on` sein, damit der LocateAnything-Assistent im Labeling-Werkzeug erscheint. Jeder andere Wert deaktiviert ihn |
| `SAM_3D_BODY_PATH` | nicht gesetzt | Pfad zum SAM-3D-Body-Paket für die Mesh-Familie, wenn er nicht dem Konstruktor übergeben wird |
| `HF_TOKEN` | nicht gesetzt | Zugriffstoken für Hugging Face, wird bei zugangsbeschränkten Repositorys verwendet |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` wird beim Import gelesen. Wenn du die Variable nach dem
Import von `libreyolo.data` setzt, ändert sich `DATASETS_DIR` nicht.

Hub-Kernel verwenden ein zweiteiliges Opt-in. Der Abruf zur Laufzeit geschieht
nur, wenn das optionale Paket `kernels` installiert ist. Die Installation von
`libreyolo[hub-kernels]` ist daher das Opt-in, und
`LIBREYOLO_HUB_KERNELS=0` das Opt-out. Eine Installation ohne das Extra ist in
beiden Fällen nicht betroffen.

Die Kernel-Auswahl beendet auch Importe vorzeitig. Wenn `LIBREYOLO_KERNELS`
`off` oder `reference` erzwingt, werden die beschleunigten, im Quellbaum
enthaltenen Provider überhaupt nicht importiert. Das von diesen drei Variablen
gesteuerte Register wird unter [Kernel](/docs/reference/kernels) beschrieben.

## Von der Bibliothek gesetzte Variablen

Diese Variablen werden geschrieben und nicht gelesen. Sie von Hand zu setzen,
ist daher kein unterstützter Pfad.

| Variable | Gesetzt von |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | DDP-Spawn-Hilfsfunktion, ein Wert pro Worker-Prozess |
| `CUDA_VISIBLE_DEVICES` | Wird während der verteilten Einrichtung vorübergehend eingeschränkt und danach wiederhergestellt |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Wird von den EC-Trainern mit `setdefault` auf `1` gesetzt, sodass ein vorhandener Wert Vorrang hat |
| `MOMENTUM_ENABLED` | Wird vom Loader der Mesh-Familie mit `setdefault` gesetzt |

`LOCAL_RANK` dient zugleich als Signal für den verteilten Modus. Anhand dieser
Variable erkennt der Trainingscode, dass er unter DDP ausgeführt wird.

## Logger-Variablen

Die optionalen Trainings-Logger verwenden Umgebungsstandardwerte als
Projektname.

| Variable | Standardwert | Verwendet von |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | Weights-and-Biases-Logger, wenn kein Projekt übergeben wird |
| `COMET_PROJECT_NAME` | `libreyolo` | Comet-Logger, wenn kein Projekt übergeben wird |

Die Authentifizierung bei diesen Diensten folgt deren eigenen Werkzeugen und
nicht LibreYOLO.

## Tokens

`HF_TOKEN` ist das Zugriffstoken für Hugging Face. Wenn die Variable nicht
gesetzt ist, wird das Token aus `~/.cache/huggingface/token` gelesen, wohin es
ein Login über die Hugging-Face-CLI schreibt. Beide Pfade funktionieren.

Ein Token ist nur für zugangsbeschränkte Repositorys erforderlich. SAM 3 ist
das enthaltene Beispiel. Seine Gewichte werden unter einer benutzerdefinierten
Lizenz aus einem zugangsbeschränkten Repository heruntergeladen. Du musst die
Bedingungen auf der Repository-Seite akzeptieren und die Sitzung
authentifizieren.

## Verzeichnisse

| Pfad | Inhalt |
|---|---|
| `weights/` | Heruntergeladene Checkpoints, Hugging-Face-Snapshots und exportierte Artefakte |
| `~/datasets` | Datensatzstamm, sofern `LIBREYOLO_DATASETS_DIR` nichts anderes angibt |
| `~/.cache/huggingface/token` | Hugging-Face-Token, wenn es nicht in `HF_TOKEN` steht |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | MHR-Körpermodell, sofern `LIBREYOLO_MHR_PATH` nichts anderes angibt |
| `runs/track/` | Standardausgabe für `model.track(save=True)` |

`weights/` ist relativ zum Arbeitsverzeichnis. Ein einfacher Dateiname wird
darüber aufgelöst. `LibreYOLO("LibreYOLO9t.pt")` sucht daher nach
`weights/LibreYOLO9t.pt` und lädt die Datei dort herunter, wenn sie fehlt.
`model.export()` schreibt ohne `output_path` in dasselbe Verzeichnis. Die
benachbarten Stufen laden Snapshots aus mehreren Dateien nach
`weights/<Prefix><size>/` herunter.

## Downloadverhalten

Downloads von Gewichten werden mit Backoff dreimal wiederholt, aus einer
partiellen Datei fortgesetzt und durch eine Sperrdatei geschützt. Zwei Prozesse
rufen daher nicht gleichzeitig denselben Checkpoint ab. Eine Familie, die von
einem Drittanbieterhost lädt, kann eine Prüfsumme festlegen und bei Abweichung
sicher fehlschlagen.

Einige Downloads zeigen vor Beginn einen Lizenzhinweis an. Diese Hinweise sind
Teil des Downloadpfads und können nicht über eine Konfiguration unterdrückt
werden.

## Validierungs-Backend

`model.val()` akzeptiert standardmäßig `faster_coco_eval=True` und fällt mit
einer einmaligen Warnung auf pycocotools zurück, wenn das Paket nicht
installiert ist. `LIBREYOLO_FASTER_COCO_EVAL` überschreibt das aufrufbezogene
Flag. Ein Benchmark-Harness, der keine Konfiguration pro Lauf ändern kann,
sollte diese Variable verwenden. Das tatsächlich ausgeführte Backend wird als
`model.last_eval_backend` gemeldet.

## Datensatz-Downloadskripte

Eine Datensatz-YAML kann ein Feld `download` mit Python-Code enthalten. Es wird
nur ausgeführt, wenn an den lesenden Aufruf `allow_download_scripts=True`
übergeben wird. Dies ist ein Funktionsargument von `val()` und `export()` und
keine Umgebungsvariable.

