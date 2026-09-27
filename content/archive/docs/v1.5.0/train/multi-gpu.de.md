---
title: Multi-GPU-Training
seo_title: Multi-GPU-Training in LibreYOLO
description: >-
  Trainiere mit device="0,1" auf mehreren GPUs. Erfahre, wie die Bibliothek
  DDP-Worker startet, warum batch die globale Batch-Größe bezeichnet, wann
  sync_bn sinnvoll ist und wie der torchrun-Pfad funktioniert.
lead: >-
  Multi-GPU-Training in LibreYOLO verwendet PyTorch DistributedDataParallel: Ein
  Prozess pro GPU hält jeweils eine vollständige Modellreplik und einen Teil
  jedes Batches. Die Gradienten werden bei jedem Schritt über alle Ränge
  gemittelt.
keywords:
  - pytorch ddp training
  - multi gpu training
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm
  - globale batch-größe
  - nccl gloo backend
  - multi gpu windows
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Der __main__-Guard ist nötig: Jeder gestartete Worker importiert
        # dieses Modul erneut und würde sonst rekursiv das Training starten.
        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # globaler Batch: 16 Bilder je GPU bei zwei GPUs
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: Start
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # Einmal auf GPU 0 testen und auf ein Vielfaches der Weltgröße skalieren.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## Ausführung auf zwei GPUs

Übergib eine Geräteliste. Alles andere bleibt unverändert.

<code-tabs name="train" />

Wenn mehr als ein Gerät und keine torchrun-Umgebung vorhanden sind, speichert
`train()` des Modells die Gewichte in einer temporären Datei, ermittelt bei
Bedarf die Autobatch-Größe und startet mit `torch.multiprocessing.spawn` einen
Worker-Prozess pro GPU. Jeder Worker importiert die Modellklasse erneut, baut
sie aus den gespeicherten Gewichten auf und führt den normalen Pfad für ein
einzelnes Gerät aus. Innerhalb eines gestarteten Workers sind die
torchrun-Umgebungsvariablen gesetzt. Nach Abschluss des Laufs wird der beste
Checkpoint von Rang 0 wieder in die Modellinstanz des aufrufenden Prozesses
geladen.

`device` akzeptiert `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"`
und `"auto"`. Nur eine Liste mit mehr als einem CUDA-Index startet mehrere
Prozesse.

## Obligatorischer `__main__`-Guard

Gestartete Worker importieren das Modul erneut, aus dem sie stammen. Ohne den
Guard `if __name__ == "__main__":` führt dieser Import den Trainingsaufruf
erneut aus und jeder Worker startet eigene Worker. Die Bibliothek erkennt
diesen Fall und löst einen Fehler aus, statt die Rekursion zuzulassen:

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

Alle Daten, die an einen Worker übergeben werden, werden serialisiert. Daher
muss `callbacks=` pickle-kompatibel sein. Eine Klasse auf Modulebene
funktioniert, ein Closure oder Lambda dagegen nicht. Die Fehlermeldung weist
darauf hin und nennt die integrierten Logger als Alternative.

## `batch` als globale Batch-Größe

`batch` ist die Anzahl der Bilder pro Optimizer-Schritt über alle GPUs hinweg.
Der Dataloader jedes Rangs wird mit `batch // world_size` und einem
`DistributedSampler` erstellt. `batch=32` auf zwei GPUs bedeutet daher 16
Bilder pro GPU und nicht 32.

Eine Batch-Größe, die nicht ohne Rest durch die Weltgröße teilbar ist, löst
einen Fehler aus, statt unbemerkt mit einer anderen Größe zu trainieren:

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

DDP mittelt die Gradienten selbst. Deshalb wird der Loss ohne Skalierung
übergeben. Eine zusätzliche Multiplikation mit der Weltgröße würde die
effektive Lernrate ungefähr um die Anzahl der GPUs erhöhen.

## Autobatch unter DDP

`batch=-1` funktioniert und gibt eine durch die Weltgröße teilbare globale
Batch-Größe zurück.

<code-tabs name="autobatch" />

Beim Spawn-Pfad läuft die Messung im übergeordneten Prozess auf dem ersten
Gerät, bevor ein Worker existiert. Jeder Worker erhält daher eine konkrete
Ganzzahl und es ist keine Koordination zwischen Prozessen erforderlich. Unter
torchrun misst Rang 0 und verteilt das Ergebnis als einzelnen Long-Tensor.

Die Messung ermittelt die Kapazität einer GPU und multipliziert sie mit der
Weltgröße. Wenn `nbs` gesetzt ist, wird der globale Batch auf `nbs` begrenzt
und auf ein Vielfaches der Weltgröße abgerundet. Beim Hinzufügen von GPUs sinkt
daher die Anzahl der Akkumulationsschritte, statt dass der Batch pro GPU
kleiner wird. Die Funktionsweise der Messung selbst wird unter
[Hyperparameter](/docs/train/hyperparameters) beschrieben.

## SyncBatchNorm

Unter DDP sehen die BatchNorm-Schichten jedes Rangs nur dessen eigenen Teil.
Bei `batch // world_size` kann dieser Teil so klein sein, dass die laufenden
Statistiken das konvergierte Modell gegenüber einem Single-GPU-Lauf
verschlechtern.

`sync_bn=True` wandelt jede BatchNorm in SyncBatchNorm um, sodass die
Statistiken über den globalen Batch berechnet werden. Die Umwandlung erfolgt
nur bei aktivem verteiltem Training. Ein Single-GPU-Lauf bleibt daher
unabhängig vom Flag unverändert.

Für die BatchNorm-lastigen Convolutional-Familien ist die Option bereits
standardmäßig aktiviert: YOLOX, YOLOv7, YOLOv9 und seine Varianten, YOLO-NAS,
PicoDet, RTMDet und FOMO. Bei allen anderen Familien ist sie standardmäßig
deaktiviert. Der Trainer warnt, wenn ein Modell BatchNorm enthält, `sync_bn`
deaktiviert ist und der Batch pro Rang unter 16 liegt.

<code-tabs name="syncbn" />

Für `sync_bn` gibt es kein CLI-Flag. Es ist ein Python-Argument.

## Start mit torchrun

torchrun funktioniert ebenfalls und ist die richtige Wahl, wenn bereits ein
Cluster-Scheduler den Prozessstart steuert. Schreibe das Skript für ein
einzelnes Gerät und lasse torchrun die Rangumgebung festlegen.

<code-tabs name="torchrun" />

Kombiniere die beiden Verfahren nicht. Wenn die torchrun-Umgebung vorhanden
ist, startet `device="0,1"` keine Prozesse. Der Trainer verwendet
`cuda:LOCAL_RANK`, während torchrun die Prozessanzahl steuert.

## Verhalten der Ränge

Rang 0 steuert jeden Seiteneffekt. Er ermittelt das Ausgabeverzeichnis und
verteilt den aufgelösten Namen, damit alle Ränge übereinstimmen. Außerdem
schreibt er Checkpoints und Artefakte und löst Nutzer-Callbacks sowie Logger
aus. Die anderen Ränge trainieren und tragen Gradienten bei.

Jeder Rang initialisiert seinen Dataloader und den Zufallszahlengenerator der
Augmentierung anders. Die Werte werden aus dem konfigurierten `seed` abgeleitet,
damit die Ränge keine identischen Augmentierungen ziehen.

## Plattform und Backend

Das Backend wird automatisch gewählt: NCCL, wenn sowohl CUDA als auch NCCL
verfügbar sind, andernfalls Gloo. NCCL ist unter Windows nicht enthalten.
Windows-Läufe verwenden daher ohne weitere Konfiguration Gloo. Die
Prozessgruppe wird mit einem Timeout von drei Stunden initialisiert.

## Nicht unter DDP ausgeführte Funktionen

- CUDA-Graph-Erfassung. `cuda_graph=True` protokolliert eine Zeile und
  trainiert im Eager-Modus. Siehe
  [Trainingsperformance](/docs/train/performance).
- Der Trainings-Profiler. `profile=True` wird mit einer Warnung ignoriert.

Nicht jede Familie unterstützt den automatischen Spawn. 24 Familien tun dies
und decken die trainierbaren Familien für Erkennung, Klassifikation,
semantische Segmentierung und Restaurierung ab. Erhält eine Familie ohne diese
Unterstützung mehrere GPUs, löst sie einen Fehler aus, der die Modell-API und
den torchrun-Befehl nennt, statt unbemerkt nur auf einer GPU zu trainieren.

## Verwandte Themen

- Unter [Hyperparameter](/docs/train/hyperparameters) findest du Informationen
  zu `batch`, `nbs` und dem Fortsetzen eines Laufs.
- [Experiment-Logger](/docs/train/loggers) beschreibt die Anforderung an die
  Pickle-Kompatibilität von Callbacks.
- [Cloud-GPUs](/docs/train/cloud-gpus) beschreibt das Mieten eines
  Multi-GPU-Rechners.

