---
title: Установка
seo_title: Установка LibreYOLO
description: >-
  Установка LibreYOLO с PyPI, выбор опциональных extra, которые нужны семейству
  моделей или цели экспорта, и проверка, видит ли PyTorch вашу GPU.
lead: >-
  LibreYOLO публикуется на PyPI под именем libreyolo. Базовый пакет закрывает
  предсказание, обучение, валидацию и те семейства моделей, которым не нужно
  ничего, кроме PyTorch; опциональные extra добавляют остальное.
keywords:
  - установка libreyolo
  - pip install libreyolo
  - libreyolo extras
  - libreyolo cuda
  - libreyolo не видит gpu
  - требования libreyolo
last_verified: 1.5.0
meta:
  - label: Пакет
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 или новее
  - label: Лицензия кода
    value: MIT
  - label: Основная зависимость
    value: PyTorch 2.4 или новее
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: С extra-пакетами
      language: bash
      code: |
        # Через запятую можно объединить несколько в одной установке.
        pip install "libreyolo[rfdetr,onnx]"
    - label: Всё сразу
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: Из исходников
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, все видимые GPU и то, какие
        # опциональные пакеты установлены.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Перечень моделей
      language: bash
      code: |
        # Все зарегистрированные семейства с их задачами, размерами и
        # входными разрешениями. Семейства, у которых не хватает extra,
        # выводятся вместе с командой pip, которая их включает.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## Установка

<code-tabs name="install" />

Нужен Python 3.10 или новее. Базовая установка подтягивает PyTorch, torchvision,
NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools, typer, click,
safetensors и SciPy, поэтому YOLOv9 и остальные семейства, которым больше ничего
не нужно, работают сразу после `pip install libreyolo`.

Клон переключается на `release` — стабильную ветку, код которой совпадает с
этой документацией. Интеграционная ветка, где лежат ещё не вышедшие наработки, —
`dev`.

## Опциональные extra

Extra — это имя в квадратных скобках, которое добавляет зависимости, нужные
одному семейству моделей или одной цели экспорта. Больше ничего не меняется: API
одинаков и с extra, и без него.

### Семейства моделей

| Extra | Что добавляет |
|---|---|
| `rfdetr` | `transformers`, откуда берётся бэкбон RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, откуда берутся энкодеры ViT-L/16 и EfficientNet-Lite3 для MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate`, а вне macOS ещё и `bitsandbytes` |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` и `ftfy`, нужные встроенному в пакет текстовому токенизатору CLIP |
| `siglip2` | `sentencepiece`, нужный многоязычному токенизатору SigLIP 2 |
| `gaze` | `gdown`, который включает автоскачивание чекпойнта L2CS |
| `rtdetr` | Ничего. RT-DETR не нужна дополнительная зависимость; имя сохраняется ради стабильности |

### Экспорт и среды выполнения

| Extra | Что добавляет |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 и `pycuda`, вне macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, только macOS |
| `tflite`, псевдоним `litert` | `libreyolo[onnx]` плюс `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` и `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` плюс `MNN` |
| `ncnn` | `pnnx` и `ncnn` |
| `paddle` | `libreyolo[onnx]` плюс `paddlepaddle` 2.6.2 и `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` для инференса V2 по HTTP и HTTPS |

### Обучение, оценка и логирование

| Extra | Что добавляет |
|---|---|
| `lora` | `libreyolo[rfdetr]` плюс `peft`, для дообучения с `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, бэкенд оценки COCO на C++ |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, псевдоним `dvc` | `dvclive` |

`fast-eval` подключается по желанию и не входит в жёсткие зависимости, чтобы
платформа без готового wheel-пакета не ломала обычную установку. Если пакета нет, оценка
COCO откатывается на pycocotools и запуск продолжается.

### Инструменты

| Extra | Что добавляет |
|---|---|
| `stream` | `yt-dlp`, нужен только для разбора URL страниц YouTube |
| `tracking` | Ничего. Все зависимости трекинга уже входят в основные |
| `label` | `libreyolo[sam]`, который включает режим «клик — маска» в `libreyolo label` |
| `hub-kernels` | `kernels`, опциональный загрузчик скомпилированных ядер с Hub. См. [kernels](/docs/reference/kernels), где отмечено, что его установка может сдвинуть предсказания RF-DETR в пределах точности float |
| `clip-convert` | `libreyolo[clip]` плюс `open_clip_torch`, для конвертации весов и проверок паритета |
| `siglip2-convert` | `libreyolo[siglip2]` плюс `transformers`, по той же причине |

Для веб-камер, RTSP, RTMP, TCP, UDP, HLS и локальных списков из нескольких
потоков extra не нужен. Нужен он только для URL страниц YouTube.

### Сводный extra

`libreyolo[all]` ставит extra моделей, экспорта, трекинга и логирования одной
командой. Некоторые намеренно оставлены за его пределами. `neptune` исключён,
потому что стабильный `neptune-scale` требует protobuf ниже 7, а путь TFLite
требует protobuf 7. `executorch` исключён, потому что ExecuTorch ограничивает, с
какой версией PyTorch он сочетается, а `coreai` — потому что `coreai-torch`
фиксирует PyTorch на 2.11.x и утянул бы на эту версию всё окружение.
`fast-eval`, `hub-kernels`, `clip-convert` и `siglip2-convert` тоже оставлены за
бортом. Любой из них ставится по имени.

## Ограничения платформ

Три extra ограничены платформой через маркеры своих зависимостей, поэтому
установка проходит везде и просто ставит меньше там, где wheel-пакета не
существует.

| Extra | Ограничение |
|---|---|
| `coreai` | Только macOS. Инструментарий Core AI нигде больше не конвертирует и не запускает модели |
| `tensorrt` | Пропускается на macOS, где нет CUDA |
| `tflite`, `litert` | `onnx2tf` и `ai-edge-litert` требуют Python 3.12 или новее |

`sensenova` пропускает `bitsandbytes` на macOS, где wheel-пакет не публикуется;
остальная часть extra ставится как обычно.

Если ограничение — место на диске, то большую его часть занимает PyTorch, а
большую часть PyTorch — CUDA-составляющая, которую по умолчанию тянет за собой
его wheel-пакет. Wheel-пакет только для CPU убирает её, и при этом ничего не
теряется. Для
детекции через ONNX на машине, где torch не должно быть вовсе, см.
[облегчённую установку](/docs/lightweight-install).

## GPU и CUDA

Устройство выбирается при создании модели. При значении по умолчанию
`device="auto"` берётся CUDA, когда `torch.cuda.is_available()` истинно, затем
Metal Performance Shaders, когда истинно `torch.backends.mps.is_available()`, и
CPU во всех остальных случаях. Больше
ничто в библиотеке не проверяет железо, поэтому если PyTorch не видит GPU, то не
видит её и LibreYOLO.

Чтобы вместо этого зафиксировать устройство, передайте `device` модели или в
`predict`, `train`, `val` и `export`. Параметр принимает `"cpu"`, `"cuda"`,
`"cuda:0"`, `"mps"`, голое целое число вроде `0` или строку из цифры вроде
`"0"`; последние два разворачиваются в `cuda:<n>`.

Начните с `libreyolo checks`: команда печатает версию Torch, версии CUDA и
cuDNN, с которыми Torch был собран, и каждую видимую GPU с её памятью. Если она
сообщает, что CUDA нет, а в машине стоит карта NVIDIA, значит, pip выбрал
wheel-пакет PyTorch со сборкой под CPU. Сначала поставьте сборку с CUDA из
индекса PyTorch, а потом LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

Это тот же индекс, который репозиторий фиксирует для собственного окружения под
управлением uv на Linux и Windows. Ему нужен драйвер NVIDIA 555 или новее — это
требование среды выполнения CUDA 12.8. На macOS остаётся wheel-пакет с PyPI,
поскольку хост загрузок PyTorch не публикует сборок под Darwin.

## Проверка установки

<code-tabs name="verify" />

`libreyolo models` — самый быстрый способ увидеть, подействовал ли extra:
семейство, у которого не хватает зависимости, печатается с точной командой pip,
которая его включает. Обе команды также принимают `--json`, который печатает те
же данные в виде машиночитаемого объекта в stdout.
