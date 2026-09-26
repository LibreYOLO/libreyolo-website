---
title: Instalacja
seo_title: Instalacja LibreYOLO
description: >-
  Instalacja LibreYOLO z PyPI, wybór opcjonalnych dodatków wymaganych przez
  rodzinę modeli lub cel eksportu oraz sprawdzenie, czy PyTorch widzi GPU.
lead: >-
  LibreYOLO jest publikowane w PyPI jako libreyolo. Pakiet bazowy obejmuje
  predykcję, trenowanie, walidację i rodziny modeli, które nie wymagają niczego
  poza PyTorch. Opcjonalne dodatki zapewniają pozostałe funkcje.
keywords:
  - instalacja LibreYOLO
  - pip install LibreYOLO
  - dodatki LibreYOLO
  - LibreYOLO CUDA
  - LibreYOLO GPU
  - wymagania LibreYOLO
last_verified: 1.5.0
meta:
  - label: Pakiet
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 lub nowszy
  - label: Licencja kodu
    value: MIT
  - label: Główna zależność
    value: PyTorch 2.4 lub nowszy
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: Z dodatkami
      language: bash
      code: >
        # Aby połączyć kilka dodatków w jednej instalacji, rozdziel je
        przecinkami.

        pip install "libreyolo[rfdetr,onnx]"
    - label: Wszystko
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: Ze źródeł
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, każde widoczne GPU oraz lista
        # zainstalowanych pakietów opcjonalnych.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Inwentarz modeli
      language: bash
      code: >
        # Każda zarejestrowana rodzina wraz z zadaniami, rozmiarami i
        rozdzielczościami

        # wejściowymi. Rodziny, których dodatku brakuje, są wymieniane z
        poleceniem

        # pip, które je włącza.

        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## Instalacja

<code-tabs name="install" />

Wymagany jest Python 3.10 lub nowszy. Instalacja bazowa pobiera PyTorch,
torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools,
typer, click, safetensors i SciPy. Dlatego YOLOv9 oraz inne rodziny, które nie
potrzebują niczego więcej, działają od razu po `pip install libreyolo`.

Klonowanie wybiera `release`, stabilną gałąź, której kod odpowiada tej
dokumentacji. Gałąź integracyjna zawierająca niewydane zmiany to `dev`.

## Opcjonalne dodatki

Dodatek jest nazwą w nawiasach kwadratowych, która instaluje zależności wymagane
przez jedną rodzinę modeli lub jeden cel eksportu. Nic innego się nie zmienia:
API jest takie samo niezależnie od obecności dodatku.

### Rodziny modeli

| Dodatek | Dodaje |
|---|---|
| `rfdetr` | `transformers`, dostarczające backbone RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, dostarczające enkodery ViT-L/16 i EfficientNet-Lite3 dla MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate` oraz `bitsandbytes` poza macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` i `ftfy`, wymagane przez dołączony tokenizer tekstowy CLIP |
| `siglip2` | `sentencepiece`, wymagane przez wielojęzyczny tokenizer SigLIP 2 |
| `gaze` | `gdown`, włączające automatyczne pobieranie checkpointu L2CS |
| `rtdetr` | Nic. RT-DETR nie wymaga dodatkowej zależności, ale nazwa pozostaje stabilna |

### Eksport i środowiska uruchomieniowe

| Dodatek | Dodaje |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 i `pycuda`, poza macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, tylko macOS |
| `tflite`, alias `litert` | `libreyolo[onnx]` oraz `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` i `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` oraz `MNN` |
| `ncnn` | `pnnx` i `ncnn` |
| `paddle` | `libreyolo[onnx]` oraz `paddlepaddle` 2.6.2 i `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` do inferencji V2 przez HTTP i HTTPS |

### Trenowanie, ewaluacja i rejestrowanie

| Dodatek | Dodaje |
|---|---|
| `lora` | `libreyolo[rfdetr]` oraz `peft` do dostrajania z `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, backend ewaluacji COCO w C++ |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, alias `dvc` | `dvclive` |

`fast-eval` wymaga jawnego włączenia zamiast być twardą zależnością, aby brak
gotowego pakietu dla platformy nie mógł zepsuć zwykłej instalacji. Gdy pakietu
nie ma, ewaluacja COCO przechodzi na pycocotools i przebieg jest kontynuowany.

### Narzędzia

| Dodatek | Dodaje |
|---|---|
| `stream` | `yt-dlp`, wymagane tylko do rozwiązywania adresów stron YouTube |
| `tracking` | Nic. Każda zależność śledzenia jest już zależnością bazową |
| `label` | `libreyolo[sam]`, włączające wspomaganie click-to-mask w `libreyolo label` |
| `hub-kernels` | `kernels`, opcjonalny loader skompilowanych kerneli Hub. Zobacz stronę [kerneli](/docs/reference/kernels), która informuje, że instalacja może zmienić predykcje RF-DETR w granicach tolerancji zmiennoprzecinkowej |
| `clip-convert` | `libreyolo[clip]` oraz `open_clip_torch` do konwersji wag i kontroli zgodności |
| `siglip2-convert` | `libreyolo[siglip2]` oraz `transformers` z tego samego powodu |

Kamery internetowe, RTSP, RTMP, TCP, UDP, HLS i lokalne listy wielu strumieni
nie wymagają dodatku. Potrzebują go tylko adresy stron YouTube.

### Dodatek zbiorczy

`libreyolo[all]` instaluje dodatki modeli, eksportu, śledzenia i rejestrowania
jednym poleceniem. Niektóre są celowo wyłączone. `neptune` jest wykluczone,
ponieważ stabilne `neptune-scale` wymaga protobuf poniżej 7, natomiast ścieżka
TFLite wymaga protobuf 7. `executorch` jest wykluczone, ponieważ ExecuTorch
ogranicza zgodne wersje PyTorch, a `coreai`, ponieważ `coreai-torch` przypina
PyTorch do 2.11.x i przeniosłoby całe środowisko na tę wersję. `fast-eval`,
`hub-kernels`, `clip-convert` i `siglip2-convert` również są pominięte. Każdy
z nich należy zainstalować według nazwy.

## Ograniczenia platform

Trzy dodatki są ograniczone do platformy przez znaczniki zależności, dzięki
czemu instalacja powiedzie się wszędzie i po prostu zainstaluje mniej tam,
gdzie pakiet nie istnieje.

| Dodatek | Ograniczenie |
|---|---|
| `coreai` | Tylko macOS. Zestaw narzędzi Core AI nie konwertuje ani nie działa nigdzie indziej |
| `tensorrt` | Pomijane w macOS, który nie ma CUDA |
| `tflite`, `litert` | `onnx2tf` i `ai-edge-litert` wymagają Pythona 3.12 lub nowszego |

`sensenova` pomija `bitsandbytes` w macOS, dla którego nie opublikowano pakietu.
Pozostała część dodatku instaluje się normalnie.

Jeśli ograniczeniem jest miejsce na dysku, większość zajmuje PyTorch, a większą
część PyTorch stanowią dane CUDA dołączone do domyślnego pakietu. Pakiet tylko
dla CPU usuwa je bez rezygnowania z innych funkcji. Informacje o detekcji ONNX
na maszynie, która w ogóle nie powinna zawierać torch, znajdują się na stronie
[lekkiej instalacji](/docs/lightweight-install).

## GPU i CUDA

Wybór urządzenia następuje podczas tworzenia modelu. Domyślne
`device="auto"` używa CUDA, gdy `torch.cuda.is_available()` ma wartość true,
następnie Metal Performance Shaders, gdy `torch.backends.mps.is_available()`
ma wartość true, a w przeciwnym razie CPU. Nic innego w bibliotece nie sprawdza
sprzętu, więc jeśli PyTorch nie widzi GPU, LibreYOLO również go nie zobaczy.

Aby przypiąć urządzenie, przekaż `device` do modelu albo do `predict`, `train`,
`val` i `export`. Akceptowane są `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`,
sama liczba całkowita, taka jak `0`, oraz ciąg cyfr, taki jak `"0"`. Dwie
ostatnie formy są rozwijane do `cuda:<n>`.

Zacznij od `libreyolo checks`, które wyświetla wersję Torch, wersje CUDA
i cuDNN, z którymi zbudowano Torch, oraz każde widoczne GPU wraz z pamięcią.
Jeśli na maszynie z kartą NVIDIA polecenie nie zgłasza CUDA, pip wybrał pakiet
PyTorch dla CPU. Najpierw zainstaluj kompilację CUDA z indeksu PyTorch,
a następnie LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

Jest to ten sam indeks, który repozytorium przypina we własnym środowisku
zarządzanym przez uv w Linux i Windows. Wymaga sterownika NVIDIA 555 lub
nowszego, zgodnie z wymaganiami środowiska uruchomieniowego CUDA 12.8. macOS
zachowuje pakiet z PyPI, ponieważ host pobierania PyTorch nie publikuje
kompilacji Darwin.

## Sprawdzenie instalacji

<code-tabs name="verify" />

`libreyolo models` to najszybszy sposób sprawdzenia, czy dodatek zadziałał.
Rodzina z brakującą zależnością jest wyświetlana z dokładnym poleceniem pip,
które ją włącza. Oba polecenia przyjmują również `--json`, które wyświetla te
same dane jako obiekt do odczytu maszynowego na stdout.
