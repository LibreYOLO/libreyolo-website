---
title: Встановлення
seo_title: Встановлення LibreYOLO
description: >-
  Встановіть LibreYOLO з PyPI, виберіть необов'язкові набори залежностей для
  потрібного сімейства моделей або цілі експорту й переконайтеся, що PyTorch
  бачить GPU.
lead: >-
  LibreYOLO опубліковано на PyPI як libreyolo. Базовий пакет охоплює
  передбачення, навчання, валідацію та сімейства моделей, яким не потрібно
  нічого крім PyTorch; необов'язкові набори залежностей додають решту.
keywords:
  - встановити libreyolo
  - pip install libreyolo
  - додаткові залежності libreyolo
  - libreyolo cuda
  - libreyolo gpu
  - системні вимоги libreyolo
last_verified: 1.5.0
meta:
  - label: Пакет
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 або новіша
  - label: Ліцензія коду
    value: MIT
  - label: Основна залежність
    value: PyTorch 2.4 або новіша
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: З додатковими залежностями
      language: bash
      code: >
        # Щоб поєднати кілька наборів в одному встановленні, розділяйте їх
        комами.

        pip install "libreyolo[rfdetr,onnx]"
    - label: Усе
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: Із джерельного коду
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, усі видимі GPU та перелік
        # установлених необов'язкових пакетів.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Перелік моделей
      language: bash
      code: |
        # Кожне зареєстроване сімейство з його задачами, розмірами та
        # роздільною здатністю входу. Для сімейств без потрібного набору
        # залежностей наведено команду pip, яка їх вмикає.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## Встановлення

<code-tabs name="install" />

Потрібен Python 3.10 або новіший. Базове встановлення додає PyTorch,
torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools,
typer, click, safetensors і SciPy, тому YOLOv9 та інші сімейства, яким більше
нічого не потрібно, працюють одразу після `pip install libreyolo`.

Клонування отримує гілку `release`, стабільну гілку, код якої відповідає цій
документації. Інтеграційна гілка з неопублікованими змінами має назву `dev`.

## Необов'язкові набори залежностей

Додатковий набір залежностей позначається назвою в квадратних дужках і додає
залежності, потрібні одному сімейству моделей або одній цілі експорту. Решта не
змінюється: API однаковий незалежно від наявності такого набору.

### Сімейства моделей

| Набір | Додає |
|---|---|
| `rfdetr` | `transformers`, що надає бекбон RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, що надає кодувальники ViT-L/16 і EfficientNet-Lite3 для MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate` і `bitsandbytes` поза macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` і `ftfy`, потрібні вбудованому текстовому токенізатору CLIP |
| `siglip2` | `sentencepiece`, потрібний багатомовному токенізатору SigLIP 2 |
| `gaze` | `gdown`, що вмикає автоматичне завантаження контрольної точки L2CS |
| `rtdetr` | Нічого. RT-DETR не потребує додаткових залежностей; назву збережено стабільною |

### Експорт і середовища виконання

| Набір | Додає |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 і `pycuda` поза macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, лише macOS |
| `tflite`, псевдонім `litert` | `libreyolo[onnx]` разом із `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` і `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` разом із `MNN` |
| `ncnn` | `pnnx` і `ncnn` |
| `paddle` | `libreyolo[onnx]` разом із `paddlepaddle` 2.6.2 і `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` для інференсу V2 через HTTP і HTTPS |

### Навчання, оцінювання та журналювання

| Набір | Додає |
|---|---|
| `lora` | `libreyolo[rfdetr]` разом із `peft` для донавчання з `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, бекенд оцінювання COCO на C++ |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, псевдонім `dvc` | `dvclive` |

`fast-eval` є необов'язковим, а не обов'язковою залежністю, щоб відсутність
готового wheel-пакета для платформи не могла порушити звичайне встановлення.
Якщо пакета немає, оцінювання COCO використовує резервний варіант pycocotools,
і виконання продовжується.

### Інструменти

| Набір | Додає |
|---|---|
| `stream` | `yt-dlp`, потрібний лише для визначення URL-адрес сторінок YouTube |
| `tracking` | Нічого. Усі залежності відстеження вже є основними залежностями |
| `label` | `libreyolo[sam]`, що вмикає допоміжне створення масок клацанням у `libreyolo label` |
| `hub-kernels` | `kernels`, необов'язковий завантажувач скомпільованих ядер Hub. Дивіться [ядра](/docs/reference/kernels), де зазначено, що його встановлення може змінити передбачення RF-DETR у межах похибки float |
| `clip-convert` | `libreyolo[clip]` разом із `open_clip_torch` для перетворення ваг і перевірок паритету |
| `siglip2-convert` | `libreyolo[siglip2]` разом із `transformers` із тією самою метою |

Вебкамери, RTSP, RTMP, TCP, UDP, HLS і локальні списки кількох потоків не
потребують додаткового набору. Він потрібний лише URL-адресам сторінок YouTube.

### Сукупний набір

`libreyolo[all]` однією командою встановлює додаткові набори для моделей,
експорту, відстеження й журналювання. Деякі навмисно не входять до нього.
`neptune` вилучено, оскільки стабільний `neptune-scale` потребує protobuf нижче
версії 7, тоді як шлях TFLite потребує protobuf 7. `executorch` вилучено,
оскільки ExecuTorch обмежує сумісну версію PyTorch, а `coreai` тому, що
`coreai-torch` фіксує PyTorch на версії 2.11.x і перевів би на неї все
середовище. `fast-eval`, `hub-kernels`, `clip-convert` і `siglip2-convert`
також не включено. Установлюйте будь-який із них за назвою.

## Обмеження платформ

Область дії трьох наборів обмежено платформами через маркери залежностей, тому
встановлення успішне скрізь і лише встановлює менше компонентів там, де немає
wheel-пакета.

| Набір | Обмеження |
|---|---|
| `coreai` | Лише macOS. Набір інструментів Core AI не виконує ні перетворення, ні запуску на інших платформах |
| `tensorrt` | Пропускається в macOS, де немає CUDA |
| `tflite`, `litert` | `onnx2tf` і `ai-edge-litert` потребують Python 3.12 або новішого |

`sensenova` пропускає `bitsandbytes` у macOS, для якої не опубліковано
wheel-пакета; решта набору встановлюється звичайним способом.

Якщо обмеженням є місце на диску, більшість займає PyTorch, а більшу частину
PyTorch становлять компоненти CUDA, включені до його типового wheel-пакета.
Wheel-пакет лише для CPU усуває їх без жодних втрат. Для виявлення у форматі
ONNX на комп'ютері, де взагалі не має бути torch, дивіться
[полегшене встановлення](/docs/lightweight-install).

## GPU та CUDA

Пристрій вибирається під час створення моделі. Типове значення `device="auto"`
використовує CUDA, якщо `torch.cuda.is_available()` має значення true, потім
Metal Performance Shaders, якщо `torch.backends.mps.is_available()` має
значення true, інакше CPU. Інші частини бібліотеки не перевіряють обладнання,
тому якщо PyTorch не бачить GPU, LibreYOLO також його не бачить.

Щоб натомість зафіксувати пристрій, передайте `device` моделі або до `predict`,
`train`, `val` чи `export`. Він приймає `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`,
звичайне ціле число на кшталт `0` або цифровий рядок на кшталт `"0"`; останні
два варіанти розгортаються в `cuda:<n>`.

Почніть із `libreyolo checks`, що виводить версію Torch, версії CUDA та cuDNN,
для яких зібрано Torch, і всі видимі GPU з обсягом пам'яті. Якщо команда не
повідомляє про CUDA на комп'ютері з картою NVIDIA, pip вибрав збірку PyTorch
для CPU. Спочатку встановіть збірку з CUDA з індексу PyTorch, а потім
установіть LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

Це той самий індекс, який репозиторій фіксує для власного середовища під
керуванням uv у Linux і Windows. Для нього потрібний драйвер NVIDIA 555 або
новіший, що є вимогою середовища виконання CUDA 12.8. macOS зберігає
wheel-пакет із PyPI, оскільки хост завантажень PyTorch не публікує збірки
Darwin.

## Перевірка встановлення

<code-tabs name="verify" />

`libreyolo models` є найшвидшим способом перевірити, чи подіяв додатковий
набір: сімейство без залежності виводиться з точною командою pip, яка його
вмикає. Обидві команди також приймають `--json`, що виводить ті самі дані як
придатний для машинного читання об'єкт у stdout.
