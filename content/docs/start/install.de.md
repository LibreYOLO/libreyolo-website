---
title: Installation
seo_title: LibreYOLO installieren
description: >-
  Installiere LibreYOLO von PyPI, wähle die optionalen Extras für eine
  Modellfamilie oder ein Exportziel aus und prüfe, ob PyTorch deine GPU erkennt.
lead: >-
  LibreYOLO wird auf PyPI als libreyolo veröffentlicht. Das Basispaket deckt
  Vorhersage, Training, Validierung und alle Modellfamilien ab, die außer
  PyTorch nichts benötigen. Optionale Extras ergänzen die übrigen Funktionen.
keywords:
  - libreyolo installieren
  - pip install libreyolo
  - libreyolo extras
  - libreyolo cuda
  - libreyolo gpu
  - libreyolo voraussetzungen
last_verified: "1.6.0"
meta:
  - label: Paket
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 oder neuer
  - label: Code-Lizenz
    value: MIT
  - label: Kernabhängigkeit
    value: PyTorch 2.4 oder neuer
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: Mit Extras
      language: bash
      code: |
        # Mehrere Extras durch Kommas getrennt gemeinsam installieren.
        pip install "libreyolo[rfdetr,onnx]"
    - label: Alles
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: Aus dem Quellcode
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, jede sichtbare GPU und die
        # installierten optionalen Pakete.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Modellinventar
      language: bash
      code: |
        # Alle registrierten Familien mit Aufgaben, Größen und Eingabe-
        # auflösungen. Bei fehlendem Extra wird der passende pip-Befehl gezeigt.
        libreyolo models
source_hash: "531023c2092fd751"
---

## Installation

<code-tabs name="install" />

Python 3.10 oder neuer ist erforderlich. Die Basisinstallation lädt PyTorch,
torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools,
typer, click, safetensors und SciPy. YOLOv9 und andere Familien ohne weitere
Abhängigkeiten funktionieren daher direkt nach `pip install libreyolo`.

Ein geklontes Repository verwendet `release`, den stabilen Branch, dessen Code
dieser Dokumentation entspricht. Der Integrations-Branch für unveröffentlichte
Arbeit heißt `dev`.

## Optionale Extras

Ein Extra ist ein Name in eckigen Klammern, der die für eine Modellfamilie oder
ein Exportziel benötigten Abhängigkeiten ergänzt. Sonst ändert sich nichts. Die
API bleibt mit und ohne Extra gleich.

### Modellfamilien

| Extra | Ergänzt |
|---|---|
| `ground` | VLM-Abhängigkeiten für Anweisung-zu-Punkt-Grounding |
| `vlm-train` | VLM-Stack und `peft>=0.17.0` für Qwen3-VL-Fine-Tuning |
| `vla` | `lerobot[smolvla,diffusion,dataset]>=0.6.1`, Python 3.12 oder neuer |
| `marigold` | Festgelegte Abhängigkeiten für Diffusion, Beschleunigung und Transformers |
| `molmo2` | `transformers==4.57.1`, `einops` und `accelerate` |
| `rfdetr` | `transformers`, das das RF-DETR-Backbone bereitstellt |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, das die ViT-L/16- und EfficientNet-Lite3-Encoder von MiDaS bereitstellt |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate` und außerhalb von macOS `bitsandbytes` |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` und `ftfy`, benötigt vom mitgelieferten CLIP-Text-Tokenizer |
| `siglip2` | `sentencepiece`, benötigt vom mehrsprachigen SigLIP-2-Tokenizer |
| `gaze` | `gdown`, das den automatischen Download des L2CS-Checkpoints aktiviert |
| `rtdetr` | Nichts. RT-DETR benötigt keine zusätzliche Abhängigkeit; der Name bleibt stabil |

### Export und Runtimes

| Extra | Ergänzt |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime>=1.18.0` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 und `pycuda`, außerhalb von macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, nur macOS |
| `tflite`, Alias `litert` | `libreyolo[onnx]` sowie `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` und `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` sowie `MNN` |
| `ncnn` | `pnnx` und `ncnn` |
| `paddle` | `libreyolo[onnx]` sowie `paddlepaddle` 2.6.2 und `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` für HTTP- und HTTPS-V2-Inferenz |

### Training, Evaluierung und Logging

| Extra | Ergänzt |
|---|---|
| `lora` | `libreyolo[rfdetr]` sowie `peft` für Fine-Tuning mit `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, das C++-Backend für die COCO-Evaluierung |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, Alias `dvc` | `dvclive` |

`fast-eval` ist optional und keine feste Abhängigkeit. Dadurch kann eine
Plattform ohne vorgefertigtes Wheel die einfache Installation nicht
verhindern. Wenn das Paket fehlt, fällt die COCO-Evaluierung auf pycocotools
zurück und der Lauf wird fortgesetzt.

### Werkzeuge

| Extra | Ergänzt |
|---|---|
| `hf` | `huggingface_hub>=1.0.0` zum Laden, Veröffentlichen und für den Hub-Logger |
| `llm` | `openai>=1.66.0` für kompatible API-Endpunkte |
| `fiftyone` | `fiftyone>=1.0.0` zur Datensatzkuratierung |
| `stream` | `yt-dlp`, nur zum Auflösen von YouTube-Seiten-URLs erforderlich |
| `tracking` | Nichts. Jede Tracking-Abhängigkeit ist bereits eine Kernabhängigkeit |
| `label` | `libreyolo[sam]`, das die Unterstützung von Klick-zu-Maske in `libreyolo label` aktiviert |
| `hub-kernels` | `kernels`, den optionalen Loader für kompilierte Hub-Kernel. Siehe [Kernel](/docs/reference/kernels). Dessen Installation kann RF-DETR-Vorhersagen innerhalb der Float-Toleranz verschieben |
| `clip-convert` | `libreyolo[clip]` sowie `open_clip_torch` für Gewichtskonvertierung und Paritätsprüfungen |
| `siglip2-convert` | `libreyolo[siglip2]` sowie `transformers` aus demselben Grund |

Webcams, RTSP, RTMP, TCP, UDP, HLS und lokale Listen mit mehreren Streams
benötigen kein Extra. Nur YouTube-Seiten-URLs tun dies.

### Das Sammel-Extra

`libreyolo[all]` installiert die Modell-, Export-, Tracking- und Logging-Extras
mit einem Befehl. Einige sind absichtlich ausgeschlossen. `neptune` fehlt,
weil die stabile Version von `neptune-scale` protobuf unter Version 7
erfordert, während der TFLite-Pfad protobuf 7 benötigt. `executorch` fehlt,
weil ExecuTorch die kompatible PyTorch-Version einschränkt. `coreai` fehlt,
weil `coreai-torch` PyTorch auf 2.11.x festlegt und dadurch die gesamte
Umgebung auf diese Version umstellen würde. Auch `fast-eval`, `hub-kernels`,
`clip-convert` und `siglip2-convert` sind ausgeschlossen. Installiere diese
Extras bei Bedarf einzeln. `all` enthält `hf` und `llm`; `fiftyone`, `vla`, `marigold` und `molmo2` bleiben separat. FiftyOne bringt Headless-OpenCV mit, das sich mit dem Kernpaket `cv2` überschneidet.

Die Basisinstallation benötigt Python 3.10 oder neuer und ergänzt `cloudpickle>=3.0.0` für vom Koordinator verwaltetes DDP.

Verwende eine separate Umgebung für Molmo2: Seine festgelegte Transformers-Version 4.57.1 steht im Konflikt mit den neueren VLM-, Hub- und Marigold-Stacks. Marigold legt diffusers 0.38.0, peft 0.18.1, accelerate 1.13.0 und Transformers 5.4.0 sowie bitsandbytes 0.49.2 unter Linux/Windows fest. Die standardmäßige Vier-Bit-Inferenz von Marigold benötigt CUDA. North Micro Vision benötigt Transformers 5.16 oder neuer; Gemma 4 benötigt 5.10 oder neuer, oberhalb der gemeinsamen VLM-Mindestversion.

## Plattformeinschränkungen

Drei Extras sind über Abhängigkeitsmarker auf Plattformen beschränkt. Die
Installation gelingt daher überall und installiert lediglich weniger, wo kein
Wheel existiert.

| Extra | Einschränkung |
|---|---|
| `coreai` | Nur macOS. Die Core-AI-Toolchain kann auf anderen Plattformen weder konvertieren noch ausführen |
| `tensorrt` | Wird unter macOS ohne CUDA übersprungen |
| `tflite`, `litert` | `onnx2tf` und `ai-edge-litert` benötigen Python 3.12 oder neuer |

`sensenova` überspringt `bitsandbytes` unter macOS, wo kein Wheel veröffentlicht
wird. Der Rest des Extras wird normal installiert.

Wenn der Speicherplatz knapp ist, verursacht PyTorch den größten Teil des
Bedarfs, insbesondere durch die im Standard-Wheel enthaltene CUDA-Payload. Ein
reines CPU-Wheel entfernt sie ohne Funktionsverlust. Für ONNX-Erkennung auf
einem Rechner ganz ohne torch siehe die
[schlanke Installation](/docs/lightweight-install).

## GPU und CUDA

Das Gerät wird beim Erstellen eines Modells ausgewählt. Der Standardwert
`device="auto"` verwendet CUDA, wenn `torch.cuda.is_available()` wahr ist,
anschließend Metal Performance Shaders, wenn
`torch.backends.mps.is_available()` wahr ist, und andernfalls die CPU. Kein
anderer Teil der Bibliothek prüft die Hardware. Wenn PyTorch eine GPU nicht
erkennt, kann LibreYOLO sie ebenfalls nicht sehen.

Um das Gerät festzulegen, übergib `device` an das Modell oder an `predict`,
`train`, `val` und `export`. Akzeptiert werden `"cpu"`, `"cuda"`, `"cuda:0"`,
`"mps"`, eine einfache Ganzzahl wie `0` oder ein Ziffernstring wie `"0"`. Die
beiden letzten Formen werden zu `cuda:<n>` erweitert.

Beginne mit `libreyolo checks`. Der Befehl gibt die Torch-Version, die CUDA-
und cuDNN-Versionen, für die Torch erstellt wurde, sowie jede sichtbare GPU mit
ihrem Speicher aus. Wenn auf einem Rechner mit NVIDIA-Karte kein CUDA gemeldet
wird, hat pip ein CPU-Wheel von PyTorch aufgelöst. Installiere zuerst einen
CUDA-Build aus dem PyTorch-Index und anschließend LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

Dies ist derselbe Index, den das Repository für seine von uv verwaltete
Umgebung unter Linux und Windows festlegt. Er benötigt NVIDIA-Treiber 555 oder
neuer, wie von der CUDA-12.8-Runtime verlangt. macOS verwendet weiterhin das
PyPI-Wheel, da der PyTorch-Downloadhost keine Darwin-Builds bereitstellt.

## Installation prüfen

<code-tabs name="verify" />

Mit `libreyolo models` erkennst du am schnellsten, ob ein Extra aktiv ist. Eine
Familie mit fehlender Abhängigkeit wird zusammen mit dem genauen pip-Befehl
ausgegeben, der sie aktiviert. Beide Befehle akzeptieren außerdem `--json` und
geben damit dieselben Daten als maschinenlesbares Objekt auf stdout aus.
