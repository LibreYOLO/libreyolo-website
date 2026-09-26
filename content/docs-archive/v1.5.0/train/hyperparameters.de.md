---
title: Hyperparameter
seo_title: Trainingshyperparameter in LibreYOLO
description: >-
  Die entscheidenden Argumente von train(): epochs, batch, lr0, optimizer, EMA,
  Autobatch, Gradientenakkumulation und Fortsetzung sowie die Gründe für
  familienspezifische Standardwerte.
lead: >-
  Jedes Trainingsargument ist ein Feld einer TrainConfig-Dataclass. Die
  Basisklasse definiert das Feld und seinen Standardwert. Jede Modellfamilie
  leitet davon eine Unterklasse ab und überschreibt die Standardwerte, die in
  ihrem veröffentlichten Trainingsrezept abweichen.
keywords:
  - LibreYOLO Trainingsparameter
  - Lernrate einstellen
  - Batchgröße Training
  - Autobatch GPU
  - exponentiell gleitender Mittelwert
  - Gradientenakkumulation
  - Training fortsetzen
  - Early Stopping Geduld
  - AMP bfloat16
  - Trainingskonfiguration YAML
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: Aufgelöste Standardwerte einer Familie auslesen
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: >
        # Gibt die Standardwerte für Training, Validierung und Vorhersage
        einschließlich familienspezifischer Überschreibungen aus.

        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # batch=-1 prüft den GPU-Speicher und ermittelt eine konkrete
        Zweierpotenz.

        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # 4 Mikro-Batches mit je 16 Elementen pro Optimiererschritt, effektive
        Batchgröße 64.

        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Checkpoint des unterbrochenen Laufs laden und anschließend die
        Fortsetzung anfordern.

        model = LibreYOLO("runs/train/exp/weights/last.pt")

        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Die Schlüssel in der YAML-Datei sind TrainConfig-Feldnamen. Explizite
        Schlüsselwortargumente haben Vorrang.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## Argumente festlegen

`train()` nimmt Schlüsselwortargumente entgegen. Die CLI verwendet dieselben Namen
in der Form `key=value`.

<code-tabs name="train" />

Beide Wege enden an derselben Stelle. Die Schlüsselwortargumente werden an
`TrainConfig.from_kwargs()` übergeben, das die Konfigurations-Dataclass der
jeweiligen Familie erstellt.

## Tippfehler lösen keinen Fehler aus

`from_kwargs()` verwirft jeden Schlüssel, der kein Feld der Konfiguration ist,
und gibt eine `UserWarning` mit seinem Namen aus. Anschließend beginnt das
Training mit dem Standardwert:

```python
# UserWarning: Unbekannte Trainingskonfigurationsschlüssel (ignoriert): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

Es tritt kein Fehler auf, der Lauf wird abgeschlossen und die Lernrate entspricht
nie dem vom Aufrufer angeforderten Wert. Lies beim ersten Durchlauf eines neuen
Rezepts die Warnungen in der ersten Epoche. Die CLI ist strenger, da sie die
Namen der Optionen vor dem Erstellen der Konfiguration prüft. Eine falsch
geschriebene CLI-Option wird deshalb direkt abgelehnt.

## Familienspezifische Standardwerte

`TrainConfig` definiert das Feld und einen Basisstandardwert. Jede Familie leitet
davon eine Unterklasse ab und überschreibt die Werte, die in ihrem veröffentlichten
Rezept abweichen. Deshalb gibt es keine allgemeingültige Antwort auf die Frage
„Wie lautet die Standardlernrate?“.

Die Basisstandardwerte sind `optimizer="sgd"`, `lr0=0.01`, `momentum=0.937`,
`weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`, `batch=16`,
`imgsz=640` und `amp=True`. Drei Beispiele dafür, wie stark eine Familie davon
abweichen kann:

| Feld | Basis | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE und DEIM werden mit `amp=False` ausgeliefert, weil der D-FINE-Decoder
Aktivierungen bei 65504 begrenzt, dem größten endlichen float16-Wert. Bei YOLO-NAS
und FOMO ist die Option ebenfalls standardmäßig deaktiviert. Die CLI-Option
`--amp` hat für jede Familie den Standardwert `True`. Sie gilt deshalb als vom
Benutzer angegeben und überschreibt den familienspezifischen Standardwert. Lass
sie unverändert, sofern du den Wert nicht bewusst ändern möchtest.

So liest du die tatsächlichen Standardwerte einer Familie aus, statt zu raten:

<code-tabs name="defaults" />

## Batchgröße

`batch` bezeichnet den globalen Batch. Beim Multi-GPU-Training lädt jeder Rang
`batch // world_size`. Die übergebene Zahl ist daher unabhängig von der Anzahl
der beteiligten GPUs die Zahl der Bilder pro Optimiererschritt. Siehe
[Multi-GPU-Training](/docs/train/multi-gpu).

`batch=-1` aktiviert Autobatch. Der Trainer prüft das Modell im Trainingsmodus
mit einem echten Rückwärtsdurchlauf für Zweierpotenzen, passt eine Gerade an die
Speicherkurve an und wählt die größte Zweierpotenz, die strikt unter dem
extrapolierten Wert liegt und innerhalb von 60 Prozent des gesamten VRAM bleibt.

<code-tabs name="autobatch" />

Die Prüfung im Trainingsmodus mit Rückwärtsdurchlauf ist entscheidend. Eine
Prüfung im Inferenzmodus berücksichtigt weder zurückgehaltene Aktivierungen noch
Gradiententensoren, die bei einem tiefen CNN ein Mehrfaches des Inferenzbedarfs
ausmachen. RF-DETR senkt den Zielanteil auf 45 Prozent, weil der synthetische
Rückwärtsdurchlauf der Prüfung die Kosten seiner Verlustfunktion und zusätzlichen
Decoder-Schichten weiterhin unterschätzt.

Autobatch ist eine CUDA-Funktion. Auf CPU oder MPS protokolliert der Trainer eine
Zeile und behält den Standard-Batch bei.

## Gradientenakkumulation

`nbs` legt die nominelle oder effektive Batchgröße fest. Der Trainer akkumuliert
`round(nbs / batch)` Mikro-Batches pro Optimiererschritt.

<code-tabs name="accumulate" />

Beim Standardwert `None` ist die Akkumulation deaktiviert und das Training bleibt
unverändert.

## Lernrate und Zeitplan

`lr0` ist die anfängliche Lernrate. `optimizer` akzeptiert `sgd`, `adam` und
`adamw`. `momentum` bezeichnet den SGD-Impuls oder Adam-Beta1, `weight_decay` ist
der L2-Term und `nesterov` gilt für SGD.

Der Zeitplan wird von `scheduler`, `warmup_epochs`, `warmup_lr_start` und
`min_lr_ratio` bestimmt. `no_aug_epochs` legt fest, wie viele abschließende Epochen
ohne starke Augmentierung ausgeführt werden. Mehrere Zeitpläne verwenden den Wert
auch zur Gestaltung ihres Endverlaufs, er ist also nicht nur ein
Augmentierungsparameter. Wie die einzelnen Familien seine Augmentierungshälfte
nutzen, steht unter [Augmentierungen](/docs/train/augmentations).

Einige Familien ergänzen eigene Lernratenparameter. `backbone_lr_mult` skaliert
die Backbone-Gruppe relativ zum Kopf, `clip_max_norm` legt die
Gradientenbegrenzung fest und SegFormer führt seinen Decoder-Kopf mit
`head_lr_mult` bei der zehnfachen Lernrate des Backbones aus. Diese Felder gehören
zur Konfigurationsunterklasse der Familie, nicht zur Basisklasse.

## EMA

`ema=True` führt neben den trainierten Gewichten einen exponentiell gleitenden
Mittelwert der Gewichte. Die Option ist überall außer bei FOMO standardmäßig
aktiviert.

`ema_decay` ist der angestrebte Abklingfaktor. Der Wert steigt schrittweise an,
statt direkt mit dem Zielwert zu beginnen. Der effektive Wert beim Update `n`
lautet `ema_decay * (1 - exp(-n / tau))`, wobei `tau` standardmäßig 2000 beträgt.
Frühe Updates folgen dem Modell dadurch enger, während späte Updates stärker
geglättet werden. Die familienspezifischen Standardwerte reichen von `0.997` bei
YOLO-NAS Pose über `0.9998` bei YOLOX bis `0.9999` bei YOLOv9 und der DETR-Reihe.

Die EMA-Gewichte werden validiert und in `best.pt` sowie `last.pt` gespeichert.
Die unverarbeiteten trainierten Gewichte werden zusätzlich unter dem Schlüssel
`train_model` abgelegt. Eine Fortsetzung setzt damit die trainierte Trajektorie
fort und nicht den Mittelwert.

## Genauigkeit

`amp=True` führt den Vorwärtsdurchlauf unter CUDA-Autocast aus. `amp_dtype` wählt
`float16` (Standard) oder `bfloat16`; `fp16` und `bf16` sind ebenfalls zulässige
Schreibweisen.

Float16 benötigt dynamische Verlustskalierung und erhält einen aktiven
`GradScaler`. Der größere Exponentenbereich von Bfloat16 macht dies überflüssig.
Sein Scaler wird daher erstellt, aber deaktiviert, sodass der Optimiererpfad
identisch bleibt. Wird bfloat16 auf einem CUDA-Gerät ohne bfloat16-Unterstützung
angefordert, tritt der Fehler bei der Einrichtung auf, statt stillschweigend auf
eine andere Genauigkeit zurückzufallen.

## Ausgabe, Checkpoints und Abbruch

Läufe werden unter `project/name` gespeichert. `project` ist überall
standardmäßig `runs/train`, während `name` zu den familienspezifischen
Überschreibungen gehört. Der Basisstandardwert ist `exp`, YOLOv9 verwendet
`yolo9_exp` und D-FINE `dfine_exp`. Bei `exist_ok=False`, dem Standardwert,
erhält ein vorhandenes Verzeichnis einen hochgezählten Suffix, statt überschrieben
zu werden.

`save_period` schreibt alle N Epochen zusätzlich eine Datei
`weights/epoch_<N>.pt`. Dazu kommen `weights/last.pt` nach jeder Epoche und
`weights/best.pt`, sobald sich die beobachtete Metrik verbessert.
`eval_interval` bestimmt, wie oft die Validierung ausgeführt wird. `patience`
beendet den Lauf nach dieser Zahl von Epochen ohne Verbesserung, wobei `0` das
frühzeitige Beenden deaktiviert.

`cache` beschleunigt wiederholte Epochen, indem decodierte Bilder im RAM (`True`
oder `"ram"`) oder als `.npy`-Dateien neben den Quellen (`"disk"`) vorgehalten
werden. Gecachte Lesevorgänge sind bytegenau identisch mit frischen. Bei
Dataloader-Workern ist `"disk"` die sicherere Wahl.

## Fortsetzung

`resume=True` setzt einen unterbrochenen Lauf fort. Der Checkpoint muss zuerst
geladen werden, weil die Fortsetzung ihn aus dem Modell und nicht aus einem
separaten Argument liest.

<code-tabs name="resume" />

Beim Fortsetzen werden die trainierten Gewichte, der Optimiererzustand, die
EMA-Gewichte und der Update-Zähler, die Verfolgung der besten Metrik, die
`GradScaler`-Skalierung sowie die Zufallszustände von PyTorch, CUDA und NumPy
wiederhergestellt. Der Lauf beginnt bei der Epoche nach der Checkpoint-Epoche und
spult den Zeitplan bis zu dieser Position vor.

Zwei Dinge sind nicht möglich. `resume=True` kann nicht mit `pretrained`
kombiniert werden und löst dann einen Fehler aus. Wenn sich der Schlüssel der
besten Metrik im Checkpoint von dem des aktuellen Laufs unterscheidet, wird deren
Verfolgung mit einer Warnung auf null zurückgesetzt, statt Werte mit
unterschiedlicher Bedeutung zu vergleichen.

## Rezepte in einer Datei

`cfg=` lädt eine YAML-Zuordnung aus `TrainConfig`-Feldnamen und führt sie unter
den expliziten Schlüsselwortargumenten zusammen. Ein Schlüsselwortargument hat
also immer Vorrang vor der Datei.

<code-tabs name="cfg" />

`size` und `num_classes` werden aus der Datei entfernt, da sie bereits der
Modellinstanz gehören. In der CLI gibt es keine Option `--cfg`. Der Dateipfad ist
ein Python-Argument.

## Verwandte Themen

- [Datensätze](/docs/train/datasets) erläutert, welche Werte `data=` akzeptiert.
- [Augmentierungen](/docs/train/augmentations) beschreibt die
  Augmentierungsparameter und welche Familien sie berücksichtigen.
- [Einfrieren von Schichten](/docs/train/layer-freezing) und [LoRA](/docs/train/lora)
  erklären das Training einer Teilmenge der Gewichte.
- [Validierung und Metriken](/docs/train/validation) beschreibt die vom Lauf
  ausgegebenen Werte.
