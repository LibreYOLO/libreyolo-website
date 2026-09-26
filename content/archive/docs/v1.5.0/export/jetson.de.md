---
title: NVIDIA Jetson
seo_title: LibreYOLO und PyTorch auf NVIDIA Jetson installieren
description: >-
  LibreYOLO auf einem NVIDIA Jetson installieren: die vier CUDA-Bibliotheken,
  die JetPack weglässt, der --no-deps-Schritt, den PyTorch braucht, und
  gemessene Zahlen vom Orin Nano.
lead: >-
  NVIDIA-Jetson-Boards führen LibreYOLO auf den normalen aarch64-Wheels von
  PyTorch aus. Ein Jetson-spezifischer torch-Build ist nicht beteiligt, aber
  JetPack lässt vier Bibliotheken weg, gegen die torch linkt, und die
  Installation muss sie nachliefern.
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - pytorch auf jetson installieren
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available false jetson
  - no kernel image is available for execution on the device
  - tensorrt auf jetson
  - aarch64 wheels pytorch
last_verified: 1.4.0
meta:
  - label: Board
    value: 'Jetson Orin Nano Super Developer Kit, 8 GB, GPU-Compute-Capability 8.7'
  - label: Plattform
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: Getesteter Stack
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1, am 2026-07-27
  - label: Fehlt in JetPack
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: Benchmarks
    value: >-
      223 verifizierte Läufe auf diesem Board, 58 Modelle aus 12 Familien, in
      PyTorch, ONNX Runtime und TensorRT
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: Verfolgt in
    value: Die Jetson-Hälfte von Issue 648
    links:
      - label: Issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  Installationsrezept und erwartete Ausgabe stammen aus dem Installationslauf
  vom 2026-07-27 auf einem Jetson Orin Nano Super. Die Zeilen zu Latenz und
  Accuracy kommen aus dem Snapshot der verifizierten Ergebnisse hinter
  visionanalysis.org, gefiltert auf die Hardware jetson_orin, gemessen im Juni
  2026 auf libreyolo 1.2.0.dev0. Verhalten von Export und Loader gelesen aus
  libreyolo/export/exporter.py, libreyolo/export/tensorrt.py und
  libreyolo/models/__init__.py.
snippets:
  prep:
    - label: Systempakete und eine virtuelle Umgebung
      language: bash
      code: |
        # JetPack bringt weder pip noch das venv-Modul mit.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: 'PyTorch, aus dem CUDA-13-Wheel-Index'
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: 'Die vier Bibliotheken, die JetPack nicht mitliefert'
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: 'Wenn pip cuda-toolkit 13.0.3 verlangt, mit --no-deps installieren'
      language: bash
      code: |
        # Mit --no-deps nennst du auch torchs Python-Abhängigkeiten selbst.
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: Die nächste fehlende Bibliothek benennen statt raten
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Alles, was über alle torch-Bibliotheken hinweg fehlt, in einem
        Durchgang:

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 'LibreYOLO nach torch installieren, nicht davor'
      language: bash
      code: |
        # torch ist bereits erfüllt, also lässt pip den CUDA-Build stehen.
        pip install libreyolo

        # Das ONNX-Extra wird nur zum Export gebraucht. Ein TensorRT-Export
        # läuft über ONNX, installiere es also vor dem Export-Abschnitt unten.
        pip install "libreyolo[onnx]"
  verify:
    - label: Versionen und Gerät
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: Dann einen echten Kernel ausführen
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        # Lädt den Checkpoint beim ersten Aufruf herunter.
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict --source
        https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE


        # Schreibt libreyolo9s.onnx und baut daraus libreyolo9s.engine.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # Die Engine lädt über denselben Einstiegspunkt zurück.

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Power-Modus und Taktraten
      language: bash
      code: >
        sudo nvpmodel -q      # welche Modi dieses Board bietet, und der aktive

        sudo nvpmodel -m 0    # höchster Modus auf dem hier getesteten Board

        sudo jetson_clocks


        tegrastats            # Live-Last; nvidia-smi ist auf Tegra
        eingeschränkt
source_hash: c07ff908503e89b5
---

## Was diese Seite festhält

Diese Seite hält eine Konfiguration fest, die von Anfang bis Ende verifiziert
wurde, keine Support-Matrix. Das Board war ein Jetson Orin Nano Super Developer
Kit mit 8 GB Speicher unter JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13,
Python 3.12.3), und der Stack, der darauf hochkam, war `libreyolo 1.4.0` mit
`torch 2.13.0+cu130`, OpenCV 5.0.0 und NumPy 2.5.1.
`torch.cuda.is_available()` lieferte `True`, und die GPU meldete sich als
`Orin`.

Andere JetPack-Releases, andere Jetson-Boards und andere CUDA-Versionen wurden
nicht getestet. Das Rezept unten ist das, was auf dieser Kombination
funktioniert hat.

Dieser Lauf fand am 2026-07-27 gegen LibreYOLO 1.4.0 statt und wurde auf
1.5.0-Hardware nicht wiederholt: Dies ist die eine Seite im 1.5.0-Baum, die
noch eine 1.4.0-Verifikation trägt, weshalb im Front Matter
`last_verified: "1.4.0"` steht. Nichts an den 1.5.0-Änderungen berührt den
Installationspfad, die vier fehlenden Bibliotheken oder die hier beschriebenen
Export-Flags, die Befehle sollten also weiter gelten. Die Versionsnummern in
den Ausgaben unten sind aber das, was 1.4.0 ausgegeben hat, keine
1.5.0-Messung.

Zwei Dinge daran widersprechen dem, was die meisten Jetson-Anleitungen sagen.
Die Wheels sind die gewöhnlichen aarch64-Builds für CUDA 13, ein
Jetson-spezifischer torch-Build ist also nicht nötig. Und JetPack liefert vier
Bibliotheken nicht mit, gegen die diese Wheels linken, weshalb `import torch`
eine Bibliothek nach der anderen scheitert, bis alle vier installiert sind.

## Installation

JetPack-Images kommen ohne pip und ohne das `venv`-Modul, beides kommt also
zuerst.

<code-tabs name="prep" />

Ein Board mit 8 GB ist für größere Checkpoints knapp. Legst du vorher Swap auf
der NVMe an, vermeidest du einen Out-of-Memory-Kill mitten im Lauf.

Dann PyTorch. Der CUDA-13-Index führt die aarch64-Wheels; der Extra-Index
liefert die reinen Python-Abhängigkeiten von PyPI.

<code-tabs name="torch" />

Die vier `nvidia-*-cu13`-Wheels sind der Teil, der leicht untergeht. JetPack
stellt den GPU-Treiber bereit, nicht cuDNN, NCCL, cuSPARSELt oder NVSHMEM, und
torch verweigert ohne sie den Import. Alle vier auf einmal zu installieren geht
schneller, als sie einzeln über je eine Exception zu entdecken.

Das dritte Snippet deckt einen bestimmten Fehlerfall ab: Die
Abhängigkeits-Metadaten von torch fordern für den CUDA-13-Build
`cuda-toolkit==13.0.3`, wofür es auf PyPI kein aarch64-Wheel gibt, die
Auflösung scheitert also, bevor irgendetwas heruntergeladen wird. `--no-deps`
überspringt den Resolver, damit muss jede Abhängigkeit auf der Kommandozeile
genannt werden.

LibreYOLO kommt zuletzt. Installierst du es zuerst, sucht pip sich sein eigenes
torch aus, und das ist auf dieser Plattform nicht der CUDA-Build.

<code-tabs name="install" />

Jede verbleibende Abhängigkeit löst sich zu einem vorgebauten aarch64-Wheel
auf, auch OpenCV, NumPy, SciPy, pycocotools und safetensors. Nichts wird aus
dem Quelltext kompiliert.

## Prüfen, ob CUDA funktioniert

<code-tabs name="verify" />

Das zweite Snippet zählt genauso viel wie das erste. Ein Wheel, das für die
falsche GPU-Architektur gebaut wurde, meldet trotzdem
`torch.cuda.is_available() == True` und scheitert dann bei der ersten echten
Operation mit `CUDA error: no kernel image is available for execution on the
device`. Eine Matrixmultiplikation auf dem Gerät ist die Prüfung, die das
auffängt.

## Eine Vorhersage ausführen

<code-tabs name="predict" />

`predict` liefert dasselbe `Results`-Objekt wie auf jeder anderen Plattform,
die Modellseiten gelten also unverändert.

## Export nach TensorRT

Auf diesem Board war TensorRT für alle 55 Modelle, die in jeder Runtime
gemessen wurden, schneller als PyTorch und ONNX Runtime.

<code-tabs name="export" />

`format="tensorrt"` schreibt zuerst einen ONNX-Graphen und baut die Engine
daraus, das `onnx`-Extra muss also installiert sein. `LibreYOLO()` entscheidet
anhand der Dateiendung, eine `.engine`-Datei lädt also über denselben Aufruf
wie ein `.pt`-Checkpoint.

Nutze das pip-Extra `tensorrt` auf einem Jetson nicht. Es pinnt
`tensorrt-cu12`, einen CUDA-12-Build, gegen eine CUDA-13-Plattform. Nimm
stattdessen das TensorRT, das JetPack installiert. Wenn `import tensorrt` in
der virtuellen Umgebung scheitert, außerhalb aber funktioniert, erzeuge die
Umgebung mit `--system-site-packages` neu, damit das Systemmodul sichtbar ist.

Serialisierte TensorRT-Engines sind an das Gerät, die GPU-Architektur und die
TensorRT-Version gebunden, die sie gebaut hat. Eine auf einer Workstation
gebaute Engine lädt auf einem Jetson nicht, der Build-Schritt läuft also auf
dem Board.

## Auf diesem Board gemessen

Latenz pro Bild, Batch-Größe 1, von Anfang bis Ende inklusive Vor- und
Nachverarbeitung, auf COCO val2017 (Teilmenge mit 500 Bildern) bei
`conf=0.001` und `max_det=300`. Fünf Modelle von den 58 gemessenen:

| Modell | Eingabe (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

Die mAP-Spalte ist der eigene Wert des TensorRT-FP16-Laufs. Über die 55 in
allen vier Runtimes gemessenen Modelle hinweg lag der größte Abstand zwischen
dem PyTorch-FP32-Wert und dem TensorRT-FP16-Wert bei 0.59 Punkten, bei
DEIMv2-X. Die Runtimes unterscheiden sich in der Geschwindigkeit, nicht in der
Accuracy.

TensorRT FP32 war bei allen 55 dieser Modelle schneller als PyTorch und ONNX
Runtime. TensorRT FP16 war ebenfalls bei allen 55 schneller als PyTorch FP32,
um das 1.68- bis 6.22-Fache, im Median um das 3.39-Fache. ONNX Runtime ist das,
was schwankt: Bei 23 der 55 war es langsamer als PyTorch, darunter die Zeile
RT-DETR-r18.

Bedingungen hinter jeder Zahl: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`,
Python 3.12.3, CUDA 13, Treiber 595.78, ONNX Runtime 1.24.0, gemessen im Juni
2026. Die Latenz auf einem Jetson hängt außerdem vom aktiven Power-Modus ab,
den die Benchmark-Datensätze nicht mitführen.

<code-tabs name="power" />

Alle 223 Läufe, inklusive der übrigen 53 Modelle und der vollständigen
Accuracy-Spalten, sind auf
[der Jetson-Orin-Seite bei Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin)
veröffentlicht.

## Fehlersuche

### import torch scheitert und nennt eine Shared Library

Eine der vier Bibliotheken oben fehlt. Statt zu raten, welche, lies sie aus der
Binärdatei ab:

<code-tabs name="ldd" />

Jeder fehlende Eintrag entspricht genau einem Wheel:

| Fehlende Bibliothek | Wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch warnt, dass kein Build diese GPU unterstützt

Der erste CUDA-Aufruf auf der funktionierenden Konfiguration gibt das hier aus:

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

Die Warnung ist auf diesem Board kosmetisch. Das Wheel bringt `sm_80`-Kernel
mit, und der Orin führt sie aus. Dieselbe Warnung erschien beim früheren Wheel
aus diesem Index, dem, das jede Benchmark-Zeile oben erzeugt hat. Bestätige es
mit der Matrixmultiplikation aus der CUDA-Prüfung, statt der Meldung zu glauben
oder zu misstrauen.

### CUDA error: no kernel image is available for execution on the device

Das installierte Wheel wurde für eine andere GPU-Architektur gebaut. Genau das
passiert mit Wheels aus NVIDIAs `sbsa`-Index, die auf Server-ARM-GPUs zielen
und nicht auf Jetson-Silizium. Installiere neu aus dem CUDA-13-Index im
Installationsabschnitt.

### pip findet cuda-toolkit 13.0.3 nicht

Dafür gibt es kein aarch64-Wheel. Nutze die `--no-deps`-Variante im
Installationsabschnitt und nenne torchs Abhängigkeiten explizit.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

Das aarch64-Wheel von torch linkt die NVIDIA Performance Libraries für
CPU-Mathematik. Installiere sie und lege sie auf den Bibliothekspfad:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Für diese beiden CPU-Bibliotheken ist der Index in Ordnung. Seine torch-Builds
sind die, die den Fehler „no kernel image“ oben erzeugen.

### Wheel-Quellen, die nicht zu JetPack 7.2 passen

| Quelle | Ergebnis auf dem Orin Nano Super |
|---|---|
| `pypi.jetson-ai-lab.io/sbsa/cu130` torch | Für Server-ARM-GPUs gebaut. Importiert, meldet CUDA als verfügbar, scheitert dann mit „no kernel image is available for execution on the device“. |
| `pypi.jetson-ai-lab.io/jp6/*` torch | CUDA-12- und Python-3.10-Builds. Sie installieren sich auf dem Python 3.12 dieses Images nicht. |
| PyTorch-Container von JetPack 6 | Die CUDA-Initialisierung scheitert auf einem JetPack-7-Host mit Fehler 801. |
| torch aus dem Quelltext bauen | Funktioniert, dauert auf einem 8-GB-Board aber Stunden und ist unnötig, sobald die CUDA-13-Wheels installiert sind. |

## DeepStream

Für eine vollständige Video-Pipeline statt einer Python-Schleife exportierst du
mit `deepstream=True` und lässt den Graphen durch `nvinfer` laufen. Dieser Weg
hat eine eigene Seite, inklusive der erzeugten `nvinfer`-Konfiguration, dem Bau
des Bounding-Box-Parsers und den bekannten Fallstricken:
[DeepStream](/docs/export/deepstream).

Die DeepStream-Pipeline selbst wurde auf einer dedizierten x86-GPU validiert,
nicht auf einem Jetson. Der Export-Vertrag hängt nicht von der Architektur ab,
der Pipeline-Lauf auf aarch64 steht aber noch aus.

## Nicht verifiziert

- Andere JetPack-Releases als 7.2 und andere L4T-Releases als R39.2.
- Andere Jetson-Boards als das Orin Nano Super 8 GB.
- Training auf dem Board. Inferenz und Export wurden durchgespielt, ein
  Trainingslauf nicht.
- INT8-Engines. Für dieses Board gibt es nur FP32- und FP16-Zeilen.
- Batch-Größen über 1. Jede Messung oben ist Batch 1.
