---
title: Training auf einer gemieteten GPU
seo_title: LibreYOLO auf einer gemieteten Cloud-GPU trainieren
description: >-
  Einen LibreYOLO-Trainingsjob auf einer gemieteten oder serverlosen GPU
  ausführen: Daten bereitstellen, installieren, starten, live beobachten,
  Gewichte abrufen und die Kosten stoppen.
lead: >-
  Eine gemietete GPU macht aus einem Trainingslauf einen Job mit Start, Ende und
  Rechnung. Die Arbeit entspricht dem lokalen Training. Anders sind die
  Datenübertragung, die externe Überwachung, das Abrufen der Gewichte und das
  Abschalten der Maschine.
keywords:
  - Cloud GPU Training
  - GPU mieten KI Training
  - Vast.ai Training
  - Modal Serverless GPU
  - Beam GPU
  - Training auf Remote GPU
  - Hugging Face Datensatz bereitstellen
  - GPU Kosten pro Epoche
last_verified: 1.5.0
snippets:
  install:
    - label: Auf der Maschine
      language: bash
      code: >
        pip install libreyolo


        # Nur die für den Lauf benötigten Extras hinzufügen: rfdetr für das
        RF-DETR-Training,

        # lora für parametereffizientes Fine-Tuning, onnx für den anschließenden
        Export.

        pip install "libreyolo[rfdetr,lora]"
    - label: Zuerst die GPU prüfen
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # Ein für eine andere Architektur gebautes Wheel meldet True und schlägt
        # erst beim ersten echten Kernel fehl, daher wird einer ausgeführt.
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: Einmalig auf dem eigenen Rechner packen und hochladen
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: Auf der Maschine bereitstellen
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: 'Abgekoppelt, damit der Job einen Verbindungsabbruch überlebt'
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: Multi-GPU aus einer Python-Datei
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # globaler Batch über alle GPUs
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: Ein kostengünstiger Lesevorgang
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: Aus einem Skript
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: Im Browser über einen SSH-Tunnel
      language: bash
      code: >
        # Auf der Maschine (bindet standardmäßig an 127.0.0.1:8420):

        libreyolo monitor /root/runs/run1 --no-browser


        # Auf dem eigenen Rechner ausführen und dann lokal http://localhost:8420
        öffnen:

        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: Gewichte an einen dauerhaften Speicherort übertragen
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## Vor dem Mieten

Zwei Entscheidungen verursachen später höhere Kosten als jetzt.

Lege den Datensatz zuerst auf einem CDN ab. Als einzelnes TAR-Archiv in einem
Hugging-Face-Datensatzrepository funktioniert er bei jedem Anbieter gleich, wird
von allen schnell abgerufen und benötigt bei einem privaten Repository nur ein
`HF_TOKEN` in der Jobumgebung. Das Hochladen eines Datensatzes über einen privaten
Internetanschluss oder das Abrufen von einer langsamen Quelle auf der Maschine
verbraucht kostenpflichtige GPU-Zeit fürs Warten.

<code-tabs name="stage" />

Dimensioniere danach den Datenträger. Anbieter mit Speicherabrechnung berechnen
die zugewiesene, nicht die belegte Kapazität. Ein Datenträger lässt sich nach der
Erstellung nicht verkleinern. Addiere die bereitgestellten Daten, die Checkpoints
und etwa 30 Prozent Reserve. Mehr ist nicht erforderlich.

## Installation auf der Maschine

<code-tabs name="install" />

Falls das Image noch keinen zur Karte passenden CUDA-Build enthält, installiere
zuerst PyTorch und danach LibreYOLO. So löst pip nicht selbst eine reine
CPU-Version von torch auf. Das zweite Beispiel ist keine optionale Formalität.
Ein für die falsche GPU-Architektur gebautes Wheel meldet
`torch.cuda.is_available() == True` und schlägt erst bei der ersten echten
Operation mit `CUDA error: no kernel image is available for execution on the
device` fehl. Eine Matrixmultiplikation erkennt das Problem vor einer Stunde
Einrichtungsarbeit.

Setze `HF_HOME` auf einen dauerhaften Speicher, falls der Anbieter ein Volume
bereitstellt. Dann bleiben Checkpoint- und Datensatzdownloads zwischen Läufen
erhalten.

## Start

Starte den Job abgekoppelt. Stirbt eine interaktive Sitzung mit der
Netzwerkverbindung, beendet sie sonst auch das Training.

<code-tabs name="launch" />

`batch=-1` ist hier besonders sinnvoll, weil du meist auf einer Karte arbeitest,
auf der du noch nicht trainiert hast. Die Option prüft das Modell im
Trainingsmodus mit einem echten Rückwärtsdurchlauf und wählt die größte passende
Zweierpotenz. Das geht schneller, als die Grenze nach zwanzig Minuten durch einen
Speichermangelfehler zu entdecken. Siehe
[Hyperparameter](/docs/train/hyperparameters).

Auf einer Multi-GPU-Maschine startet `device="0,1,2,3"` selbstständig einen
Worker pro GPU. `batch` bleibt dabei der globale Batch über alle GPUs hinweg. Der
`__main__`-Schutz ist zwingend erforderlich, weil jeder Worker das Skript erneut
importiert. Dies und das übrige verteilte Verhalten beschreibt
[Multi-GPU-Training](/docs/train/multi-gpu).

## Externe Überwachung

Jeder Lauf schreibt `status.json` in sein Laufverzeichnis und ersetzt die Datei
in jeder Epoche atomar. Sie lässt sich kostengünstig lesen: Einige Hundert Byte
enthalten den Status, die aktuelle Epoche, die geschätzte Restzeit und die
neuesten Metriken, ohne ein Protokoll zu parsen.

<code-tabs name="watch" />

Die danebenliegende Datei `metrics.jsonl` enthält den vollständigen Verlauf pro
Epoche, `train.log` die Konsolenausgabe. `libreyolo monitor` stellt mit
ausschließlich der Standardbibliothek ein Browser-Dashboard für alle drei bereit.
Auf der Maschine muss deshalb außer LibreYOLO nichts installiert sein. Du
erreichst es über eine SSH-Portweiterleitung.

Keine dieser Methoden verändert den Trainingsprozess. Sie können sich an einen
laufenden Job anhängen, einen abgeschlossenen erneut öffnen oder einen
abgestürzten untersuchen.

## Gewichte vor dem Zahlungsstopp abrufen

Die Maschine ist entbehrlich. Übertrage Checkpoints an Meilensteinen und nicht
erst am Ende. Sonst kann ein Absturz, eine Unterbrechung durch den Anbieter oder
aufgebrauchtes Guthaben den gesamten Lauf vernichten.

<code-tabs name="push" />

`weights/best.pt` und `weights/last.pt` werden in jeder Epoche und bei jeder
Verbesserung geschrieben. `save_period=N` ergänzt Momentaufnahmen unter
`weights/epoch_<N>.pt`, wodurch eine Übertragung während des Laufs wenig kostet.
`summary.json` und `results.csv` sind ebenfalls klein und sollten mitgenommen
werden, sofern die Familie sie schreibt.

Ein Callback für `on_train_epoch_end` ist die saubere Möglichkeit, die
Übertragung zu automatisieren. Siehe
[Experiment-Logger](/docs/train/loggers). Die gehosteten Backends stellen dort
auch die Metriken bereit, ohne direkt auf die Maschine zuzugreifen.

## Zahlungsstopp

Dieser Teil verursacht echte Kosten, wenn er schiefgeht. Die Regel hängt vom
Anbietermodell ab.

Auf einem Marktplatz für rohe Maschinen läuft die Abrechnung nach Wanduhr, bis
die Instanz zerstört wird. Eine inaktive GPU kostet genauso viel wie eine
ausgelastete. Nur den Trainingsprozess zu beenden spart daher nichts. Bei einer
gestoppten Instanz wird der Datenträger weiterhin berechnet.

Auf einer serverlosen Plattform, bei der der Job eine dekorierte Funktion ist,
skaliert der Container auf null, sobald die Funktion zurückkehrt. Eine vergessene
Maschine ist dort deutlich unwahrscheinlicher. Ein hängender Job ohne Zeitlimit
wird jedoch weiterhin berechnet, daher solltest du immer eines festlegen.

Das Stoppen statt Zerstören ist ein wirksamer Hebel und zugleich eine Falle. Bei
einer Messung auf einer gemieteten Maschine mit 8 RTX 4090 und einem
250-GB-Datenträger am 2026-07-31 kostete der Betrieb $3.4828 pro Stunde, der
gestoppte Zustand nur $0.0694 pro Stunde für den Datenträger und der zerstörte
Zustand nichts. Das spart 98 Prozent, während Umgebung, bereitgestellte Daten und
Checkpoints erhalten bleiben.

Der Preis im gestoppten Zustand lässt sich vor dem Mieten berechnen:

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

Vergleiche ihn mit den Kosten eines Neuaufbaus: erneut mieten, das Image abrufen,
installieren und die Daten wieder bereitstellen. Auf derselben Maschine dauerte
der Neuaufbau etwa 15 Minuten und erforderte 43 GB eingehenden Datenverkehr, also
insgesamt ungefähr $1.00. Gegenüber $0.0694 pro Stunde ist das Stoppen bei einer
Rückkehr innerhalb von etwa 14 Stunden günstiger. Bei einer längeren Pause ist
das Zerstören und der Neuaufbau aus der bereitgestellten Kopie günstiger.

Ein Risiko macht das Stoppen bei knapper Hardware unsicher: Beim Stoppen werden
die GPUs freigegeben. Nichts reserviert sie. Ein Neustart gelingt daher nur, wenn
sie auf dem Host weiterhin verfügbar sind. Der Datenträger ist sicher, die GPUs
sind es nicht.

## Serverlos als Funktion

Wenn du keine Maschine verwalten möchtest, führen Modal und Beam eine dekorierte
Python-Funktion auf einer GPU aus und skalieren nach ihrer Rückkehr auf null. Die
nächtliche Testsuite von LibreYOLO läuft selbst auf Modal. Die Datei
`tools/ci/modal_nightly.py` im Bibliotheksrepository ist ein funktionsfähiges
Beispiel im Repository.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # OpenCV-Systembibliotheken
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # Gewichte zwischen Läufen cachen

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # Volume dauerhaft speichern


@app.local_entrypoint()
def main():
    train.remote()
```

Führe die Datei mit `modal run modal_train.py` aus. Das Containerdateisystem ist
flüchtig. Alles Erhaltenswerte gehört deshalb in das Volume oder muss übertragen
werden. Lege `timeout=` ausdrücklich fest. Nur dieser Wert verhindert bei einem
hängenden Lauf eine zeitlich unbegrenzte Rechnung.

Beam folgt demselben Muster mit einem `@function`-Dekorator, einem `Volume` und
einem aus `__main__` aufgerufenen `train.remote()`.

## Passende Größe nach Kosten pro Job

$/hr ist die falsche Optimierungsgröße. Ein kleines Modell lastet eine große
Karte nur teilweise aus, daher ist eine günstigere und langsamere GPU pro Epoche
oft billiger. Führe den Profiler einige Schritte auf der gemieteten Karte aus,
bevor du dich auf einen langen Lauf festlegst. Lautet das Urteil `dataloader` oder
`host / launch`, bringt eine schnellere GPU nichts. Mehr Worker oder ein größerer
Batch helfen dagegen erheblich. Siehe
[Trainingsleistung](/docs/train/performance).

## Verwandte Themen

- [Datensätze](/docs/train/datasets) beschreibt die erwartete Struktur des
  bereitgestellten Archivs und den Doctor-Befehl, der Probleme erkennt, bevor
  GPU-Kosten anfallen.
- [Multi-GPU-Training](/docs/train/multi-gpu) behandelt Maschinen mit mehreren
  Karten.
