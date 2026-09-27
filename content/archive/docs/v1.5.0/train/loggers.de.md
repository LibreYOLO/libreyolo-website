---
title: Experiment-Logger
seo_title: Experiment-Logger und Callbacks in LibreYOLO
description: >-
  Sende Trainingsmetriken an TensorBoard, MLflow, Weights & Biases, Comet,
  ClearML, Neptune oder DVCLive und schreibe eigene Callbacks für die vier
  Trainings-Hooks.
lead: >-
  Jede trainierbare Familie löst vier Trainingsereignisse aus. Die integrierten
  Logger sind Callback-Objekte, die dieselben Ereignisse empfangen.
  Backend-Integrationen und eigene Hooks verwenden dadurch eine gemeinsame
  Schnittstelle.
keywords:
  - TensorBoard Training
  - MLflow Tracking
  - Weights and Biases
  - ClearML
  - Comet ML
  - Neptune
  - DVCLive
  - Trainings Callbacks
  - Trainingsmetriken CSV
  - LibreYOLO Monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: Nach Name
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Konfigurierte Instanz
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import MLflowLogger

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="coco8.yaml",
            epochs=10,
            loggers=[MLflowLogger(tracking_uri="sqlite:///mlflow.db"), "tensorboard"],
        )
  callback:
    - label: Einfache Funktion
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Objekt mit mehreren Hooks
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.training import TrainEndEvent, TrainEpochEvent,
        TrainStartEvent



        class RunLog:
            def on_train_start(self, event: TrainStartEvent) -> None:
                print(f"{event.model_family}{event.model_size} -> {event.save_dir}")

            def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
                if event.is_best:
                    print(f"new best at epoch {event.epoch}: {event.best_metric}")

            def on_train_end(self, event: TrainEndEvent) -> None:
                print(f"done in {event.total_seconds:.0f}s")


        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="coco8.yaml", epochs=10, callbacks=RunLog())
  monitor:
    - label: Lauf im Browser beobachten
      language: bash
      code: |
        libreyolo monitor                     # jüngster Lauf unter runs/
        libreyolo monitor runs/train/exp      # bestimmter Lauf
source_hash: de035acbaed32804
---

## Aktivieren eines Loggers

`loggers=` akzeptiert einen registrierten Namen, eine konfigurierte Instanz oder ein Iterable mit einer Mischung aus beidem.

<code-tabs name="logger" />

Bei Namen wird die Groß- und Kleinschreibung ignoriert. Registriert sind `tensorboard`, `mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` und `dvc`. Der letzte Name ist ein Alias für `dvclive`. Jeder andere Wert löst sofort einen Fehler mit den gültigen Namen aus. Es gibt keinen Wert zur Aktivierung aller Logger und kein CLI-Flag. `loggers=` ist ein Python-Argument.

## Von jedem Backend aufgezeichnete Daten

Alle Backends schreiben dieselben Metriknamen. Ein Dashboard sieht daher unabhängig von der Auswahl gleich aus:

| Schlüssel | Wert |
|---|---|
| `train/loss` | Mittlerer Trainings-Loss der Epoche |
| `train/loss/<component>` | Jede von der Familie gemeldete Loss-Komponente |
| `lr/<group>` | Lernrate jeder Parametergruppe des Optimierers |
| `val/<metric>` | Jede Validierungsmetrik ohne ihr Präfix `metrics/` |
| `time/epoch_seconds` | Tatsächliche Dauer der Epoche |

Der Schritt ist die mit 1 beginnende Epochenzahl. Die vollständig aufgelöste Trainingskonfiguration wird zu Beginn des Trainings als Parameter protokolliert. Der Laufname lautet standardmäßig `<family><size>-<task>`, beispielsweise `yolo9s-detect`.

Am Trainingsende laden Backends mit Artefaktunterstützung vorhandene Dateien `results.csv`, `train_config.yaml` und `summary.json` hoch. Mit `log_checkpoints=True` kommt `weights/best.pt` hinzu. TensorBoard lädt nichts hoch, da es kein Artefaktkonzept besitzt. Kein Logger lädt Validierungsdarstellungen hoch.

## Fehlerverhalten

Ein fehlendes Backend-Paket löst bei der Erstellung einen Fehler aus, der den Installationsbefehl nennt. Einen Logger anzufordern und unbemerkt keine Ausgabe zu erhalten, würde einen Fehler verbergen.

Bei einem Backend-Fehler während des Laufs gilt das Gegenteil. Der erste Fehler eines Handlers deaktiviert den betreffenden Logger für den Rest des Laufs, wird protokolliert und beendet den Backend-Lauf als fehlgeschlagen. Das Training wird fortgesetzt. Der Ausfall eines Tracking-Servers kostet dich daher nicht das Training.

## Backends

Jedes benötigt ein eigenes Extra.

| Name | Extra | Konstruktor |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Importiere die Klassen aus `libreyolo.training`.

Vor dem ersten Lauf solltest du einige Backend-spezifische Hinweise kennen.

TensorBoard-Ereignisdateien werden standardmäßig unter `<save_dir>/tensorboard` gespeichert. Zeige sie mit `tensorboard --logdir runs/train` an.

MLflow 3.x hat den lokalen Dateispeicher `./mlruns` als veraltet markiert und löst einen Fehler aus, sofern `MLFLOW_ALLOW_FILE_STORE=true` nicht gesetzt ist. Übergib für lokales Tracking ohne Server stattdessen wie im obigen Snippet eine Datenbank-URI. Öffne sie mit `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases greift auf die Umgebungsvariable `WANDB_PROJECT` und anschließend auf `libreyolo` zurück. Comet verwendet zunächst `COMET_PROJECT_NAME` und dann `libreyolo`. Zugangsdaten übernimmt es aus seiner eigenen Konfiguration. `online=False` erzeugt ein Offline-Experiment. ClearML erstellt eine neue Aufgabe, meldet die Konfiguration unter `TrainConfig` und deaktiviert die automatische Framework-Erfassung, damit Metriken nicht doppelt gemeldet werden. Neptune verwendet den aktuellen Client `neptune-scale` statt des veralteten Pakets. `mode="offline"` protokolliert lokal.

DVCLive schreibt nach `<save_dir>/dvclive`. Sein Zusammenfassungsbaum wird aus `/` aufgebaut. An einem Pfad, der zugleich übergeordnetes Element ist, kann kein Gleitkommawert liegen. Deshalb wird `train/loss/box` als `train/loss.box` geschrieben, während `train/loss` seinen Namen behält. LibreYOLO deaktiviert außerdem die DVCLive-Standardwerte zum Speichern eines DVC-Experiments und Schreiben einer `dvc.yaml` im Stammverzeichnis. Ein optional aktivierter Logger erzeugt dadurch außerhalb des Laufverzeichnisses keinen Zustand für die Versionsverwaltung. Mit `save_dvc_exp=True` oder einem ausdrücklichen `dvcyaml=` aktivierst du sie wieder.

Neptune ist bewusst von `libreyolo[all]` ausgeschlossen. Sein stabiler Client benötigt protobuf unter Version 7, während das TFLite-Extra protobuf 7 benötigt. Installiere `libreyolo[neptune]` in einer Umgebung ohne TFLite-Extra.

## Schreiben eines Callbacks

Dieselben vier Ereignisse steuern alles.

<code-tabs name="callback" />

| Ereignis | Zeitpunkt | Inhalt |
|---|---|---|
| `TrainStartEvent` | Nach der Einrichtung, vor Epoche 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | Nach jeder Epoche, Training und Validierung | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | Nach Abschluss des Trainings | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | Wenn das Training einen Fehler auslöst | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Eine einfache aufrufbare Funktion erhält nur `TrainEpochEvent`. Ein Objekt kann eine beliebige Teilmenge von `on_train_start`, `on_train_epoch_end`, `on_train_end` und `on_train_exception` implementieren. Fehlende Methoden werden übersprungen.

`TrainStartEvent.config` ist die vollständig aufgelöste Konfiguration aus Benutzerargumenten und Familienstandardwerten als schreibgeschützte Zuordnung. Ereignisse sind eingefrorene Dataclasses und ihre Zuordnungen schreibgeschützt. Ein Callback kann den Lauf deshalb nicht durch Schreiben in ein Ereignis verändern.

Ein Fehler aus `on_train_start`, `on_train_epoch_end` oder `on_train_end` wird weitergereicht und beendet den Lauf. Nur `on_train_exception` ist geschützt, sodass es den ursprünglichen Fehler nicht verdecken kann.

Beim Multi-GPU-Training werden Callbacks nur auf Rang 0 ausgelöst. Beim automatischen Start von DDP müssen sie außerdem mit Pickle serialisierbar sein. Verwende deshalb eine Klasse oder Funktion auf Modulebene statt eines Closures oder Lambda-Ausdrucks. Weitere Informationen findest du unter [Multi-GPU-Training](/docs/train/multi-gpu).

## Von jedem Lauf geschriebene Dateien

Drei Dateien werden bei jeder Familie ganz ohne Konfiguration im Laufverzeichnis abgelegt:

| Datei | Schreibzeitpunkt | Inhalt |
|---|---|---|
| `status.json` | Atomar nach jeder Epoche sowie bei Start, Ende und Fehler | `state` als `running`, `completed` oder `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, aktuelle `metrics`, `best_metric`, `best_epoch` und bei einem Fehler ein Objekt `error` |
| `metrics.jsonl` | Einmal je Epoche angehängt | Eine JSON-Zeile je Epoche mit demselben Schema wie `results.csv` |
| `train.log` | Laufend | Konsolenausgabe des Laufs |

`status.json` lässt sich günstig von einem Skript oder Agenten abfragen. Durch das atomare Schreiben sieht ein Leser nie eine nur teilweise geschriebene Datei.

`results.csv` und `summary.json` sind getrennt und familienabhängig. Sie werden für YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC und DINOv2 geschrieben, nicht für die übrigen Familien. `results.csv` erhält eine Zeile je Epoche mit Loss-Komponenten, Validierungsmetriken und Lernraten als Spalten. Die Kopfzeile wird erweitert, sobald eine neue Spalte erscheint. Bei der Wiederaufnahme wird die Datei auf die Zeilen vor der fortgesetzten Epoche gekürzt, statt diese zu duplizieren.

Daneben schreibt der Trainer bei der Einrichtung immer `train_config.yaml` und die Checkpoints unter `weights/`.

## Live-Beobachtung eines Laufs

<code-tabs name="monitor" />

`libreyolo monitor` stellt über die obigen Dateien ein Browser-Dashboard bereit und verwendet nur die Standardbibliothek. Es zeigt Metrikdiagramme, das Ende des Logs und vorhandene Validierungsbilder und aktualisiert sich während des Laufs. Es besitzt ausschließlich Lesezugriff und greift nie in den Trainingsprozess ein. Du kannst es an einen aktiven Lauf anhängen, einen abgeschlossenen erneut öffnen oder einen abgestürzten untersuchen.

## Verwandte Themen

- Unter [Validierung und Metriken](/docs/train/validation) erfährst du, was die Schlüssel unter `val/` bedeuten und wie du einen Validierungs-Loss ergänzt.
- Unter [Trainingsleistung](/docs/train/performance) findest du den Profiler, ein anderes Werkzeug für eine andere Fragestellung.
