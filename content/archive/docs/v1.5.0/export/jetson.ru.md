---
title: NVIDIA Jetson
seo_title: Установка LibreYOLO и PyTorch на NVIDIA Jetson
description: >-
  Установка LibreYOLO на NVIDIA Jetson: четыре библиотеки CUDA, которых нет в
  JetPack, шаг --no-deps для PyTorch и измеренные цифры на Orin Nano.
lead: >-
  Платы NVIDIA Jetson запускают LibreYOLO на обычных wheel-пакетах PyTorch для
  aarch64. Отдельная сборка torch под Jetson не нужна, но в JetPack нет четырёх
  библиотек, с которыми линкуется torch, и при установке их приходится
  доставлять отдельно.
keywords:
  - nvidia jetson
  - jetson orin nano
  - jetpack 7.2
  - установка pytorch на jetson
  - libreyolo на jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available true но ошибка
  - no kernel image is available for execution on the device
  - tensorrt на jetson
  - wheel pytorch aarch64
last_verified: 1.4.0
meta:
  - label: Плата
    value: 'Jetson Orin Nano Super Developer Kit, 8 ГБ, compute capability GPU 8.7'
  - label: Платформа
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: Проверенный стек
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1, от 2026-07-27
  - label: Чего нет в JetPack
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: Замеры
    value: >-
      223 проверенных запуска на этой плате, 58 моделей из 12 семейств, в
      PyTorch, ONNX Runtime и TensorRT
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: Отслеживается в
    value: Часть про Jetson в issue 648
    links:
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  Рецепт установки и ожидаемый вывод взяты из установочного запуска 2026-07-27
  на Jetson Orin Nano Super. Строки с задержкой и точностью взяты из снимка
  проверенных результатов за visionanalysis.org, отфильтрованного по железу
  jetson_orin, замеры за июнь 2026 на libreyolo 1.2.0.dev0. Поведение экспорта и
  загрузчика прочитано из libreyolo/export/exporter.py,
  libreyolo/export/tensorrt.py и libreyolo/models/__init__.py.
snippets:
  prep:
    - label: Системные пакеты и виртуальное окружение
      language: bash
      code: |
        # В JetPack нет предустановленных pip и модуля venv.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: PyTorch из индекса wheel-пакетов для CUDA 13
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: 'Четыре библиотеки, которых нет в JetPack'
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: Если pip требует cuda-toolkit 13.0.3 — установка с --no-deps
      language: bash
      code: |
        # С --no-deps Python-зависимости torch тоже перечисляются вручную.
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: Определение недостающей библиотеки вместо догадок
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Всё, чего не хватает во всех библиотеках torch, за один проход:

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 'Установка LibreYOLO после torch, а не до'
      language: bash
      code: |
        # torch уже установлен, поэтому pip оставляет CUDA-сборку на месте.
        pip install libreyolo

        # Дополнение ONNX нужно только для экспорта. Экспорт в TensorRT идёт
        # через ONNX, поэтому поставьте его до раздела экспорта ниже.
        pip install "libreyolo[onnx]"
  verify:
    - label: Версии и устройство
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
    - label: Затем запуск настоящего вычислительного ядра
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

        # При первом использовании чекпойнт скачивается.
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


        # Записывает libreyolo9s.onnx, затем собирает из него
        libreyolo9s.engine.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # Движок загружается обратно через ту же точку входа.

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Режим питания и частоты
      language: bash
      code: >
        sudo nvpmodel -q      # какие режимы есть на плате и какой из них
        активен

        sudo nvpmodel -m 0    # максимальный режим на плате, проверенной здесь

        sudo jetson_clocks


        tegrastats            # нагрузка вживую; nvidia-smi на Tegra ограничен
source_hash: c07ff908503e89b5
---

## Что записано на этой странице

На этой странице записана одна конфигурация, проверенная от начала до конца, а
не матрица поддержки. Плата — Jetson Orin Nano Super Developer Kit с 8 ГБ
памяти под JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13, Python 3.12.3), а
поднявшийся на ней стек — `libreyolo 1.4.0` с `torch 2.13.0+cu130`, OpenCV
5.0.0 и NumPy 2.5.1. `torch.cuda.is_available()` вернул `True`, а GPU
определился как `Orin`.

Другие релизы JetPack, другие платы Jetson и другие версии CUDA не проверялись.
Рецепт ниже — тот, что сработал на этой комбинации.

Тот запуск был 2026-07-27 на LibreYOLO 1.4.0, и на 1.5.0 он не повторялся: это
единственная страница в дереве 1.5.0, где проверка осталась от 1.4.0, поэтому во
фронтматтере стоит `last_verified: "1.4.0"`. Ни одно из изменений 1.5.0 не
затрагивает путь установки, четыре недостающие библиотеки и флаги экспорта,
описанные здесь, так что команды должны остаться верными, но номера версий в
выводе ниже — то, что печатала 1.4.0, а не замер на 1.5.0.

Два момента здесь расходятся с тем, что пишут в большинстве руководств по
Jetson. Wheel-пакеты — обычные сборки под aarch64, опубликованные для CUDA 13,
так что отдельная сборка torch под Jetson не нужна. А четырёх библиотек, с
которыми эти wheel-пакеты линкуются, в JetPack нет, поэтому `import torch`
падает по одной библиотеке за раз, пока не установлены все четыре.

## Установка

В образах JetPack нет ни pip, ни модуля `venv`, поэтому начинать нужно с них.

<code-tabs name="prep" />

Плате с 8 ГБ тесно на крупных чекпойнтах. Если перед их загрузкой добавить swap
на NVMe, процесс не убьёт по нехватке памяти на середине запуска.

Дальше PyTorch. В индексе CUDA 13 лежат wheel-пакеты под aarch64, а
дополнительный индекс отдаёт чисто питоновские зависимости с PyPI.

<code-tabs name="torch" />

Четыре wheel-пакета `nvidia-*-cu13` — та часть, которую легко пропустить.
JetPack даёт драйвер GPU, но не cuDNN, NCCL, cuSPARSELt и NVSHMEM, а без них
torch отказывается импортироваться. Поставить все четыре сразу быстрее, чем
находить их по одному исключению за раз.

Третий сниппет закрывает конкретную поломку: в метаданных зависимостей torch для
сборки под CUDA 13 указан `cuda-toolkit==13.0.3`, а wheel-пакета под aarch64 для
него на PyPI нет, поэтому разрешение зависимостей падает ещё до того, как
что-либо скачается. `--no-deps` пропускает резолвер, а значит, каждую
зависимость приходится перечислять в командной строке.

LibreYOLO ставится последним. Если поставить его первым, pip выберет torch по
своему усмотрению, а на этой платформе это не CUDA-сборка.

<code-tabs name="install" />

Все оставшиеся зависимости разрешаются в готовые wheel-пакеты под aarch64,
включая OpenCV, NumPy, SciPy, pycocotools и safetensors. Ничего не собирается из
исходников.

## Проверка, что CUDA работает

<code-tabs name="verify" />

Второй сниппет важен не меньше первого. Wheel-пакет, собранный под другую
архитектуру GPU, всё равно сообщает `torch.cuda.is_available() == True`, а потом
падает на первой же настоящей операции с `CUDA error: no kernel image is
available for execution on the device`. Умножение матриц на устройстве — та
проверка, которая это ловит.

## Запуск предсказания

<code-tabs name="predict" />

`predict` возвращает тот же объект `Results`, что и на любой другой платформе,
поэтому страницы моделей применимы без изменений.

## Экспорт в TensorRT

На этой плате TensorRT оказался быстрее и PyTorch, и ONNX Runtime на всех 55
моделях, которые замерялись во всех средах выполнения.

<code-tabs name="export" />

`format="tensorrt"` сначала записывает ONNX-граф и уже из него собирает движок,
поэтому дополнение `onnx` должно быть установлено. `LibreYOLO()` выбирает
загрузчик по суффиксу файла, так что файл `.engine` загружается тем же вызовом,
что и чекпойнт `.pt`.

На Jetson не используйте pip-дополнение `tensorrt`. Оно закрепляет
`tensorrt-cu12` — сборку под CUDA 12 — на платформе с CUDA 13. Вместо него
берите TensorRT, который ставит JetPack. Если `import tensorrt` падает внутри
виртуального окружения, но работает снаружи, пересоздайте окружение с
`--system-site-packages`, чтобы системный модуль был виден.

Сериализованные движки TensorRT привязаны к устройству, архитектуре GPU и версии
TensorRT, которая их собрала. Движок, собранный на рабочей станции, на Jetson не
загрузится, поэтому шаг сборки выполняется на самой плате.

## Замеры на этой плате

Задержка на изображение, размер батча 1, от начала до конца, включая
предобработку и постобработку, на COCO val2017 (подмножество из 500
изображений) при `conf=0.001` и `max_det=300`. Пять моделей из 58 замеренных:

| Модель | Вход (px) | PyTorch FP32 (мс) | ONNX FP32 (мс) | TensorRT FP32 (мс) | TensorRT FP16 (мс) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

В колонке mAP — собственный результат прогона TensorRT FP16. На 55 моделях,
замеренных во всех четырёх средах выполнения, наибольший разрыв между
результатом PyTorch FP32 и результатом TensorRT FP16 составил 0.59 пункта, на
DEIMv2-X. Среды выполнения различаются по скорости, а не по точности.

TensorRT FP32 был быстрее и PyTorch, и ONNX Runtime на всех 55 этих моделях.
TensorRT FP16 тоже был быстрее PyTorch FP32 на всех 55 — от 1.68x до 6.22x, с
медианой 3.39x. Разброс даёт ONNX Runtime: он оказался медленнее PyTorch на 23
моделях из 55, среди них строка RT-DETR-r18.

Условия за каждой цифрой: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`,
Python 3.12.3, CUDA 13, драйвер 595.78, ONNX Runtime 1.24.0, замеры за июнь
2026. Задержка на Jetson зависит ещё и от активного режима питания, а его записи
бенчмарков не содержат.

<code-tabs name="power" />

Все 223 запуска, включая остальные 53 модели и полные колонки точности,
опубликованы на
[странице Jetson Orin на Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Устранение неполадок

### import torch падает с именем разделяемой библиотеки

Не хватает одной из четырёх библиотек выше. Вместо догадок о том, какой именно,
прочитайте это прямо из бинарника:

<code-tabs name="ldd" />

Каждой недостающей записи соответствует один wheel-пакет:

| Недостающая библиотека | Wheel-пакет |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch предупреждает, что ни одна сборка не поддерживает GPU этой платы

Первый вызов CUDA на рабочей конфигурации печатает вот это:

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

На этой плате предупреждение косметическое. Wheel-пакет несёт ядра `sm_80`, и
Orin их выполняет. То же предупреждение выдавал и более ранний wheel-пакет из
того же индекса — тот, на котором получена каждая строка бенчмарков выше.
Проверяйте умножением матриц из раздела про CUDA, а не доверием или недоверием
к сообщению.

### CUDA error: no kernel image is available for execution on the device

Установленный wheel-пакет собран под другую архитектуру GPU. Так происходит с
wheel-пакетами из индекса `sbsa` от NVIDIA: они рассчитаны на серверные ARM GPU,
а не на кремний Jetson. Переустановите из индекса CUDA 13 из раздела установки.

### pip не находит cuda-toolkit 13.0.3

Wheel-пакета под aarch64 для него нет. Используйте форму с `--no-deps` из
раздела установки и перечислите зависимости torch явно.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

Wheel-пакет torch под aarch64 линкуется с NVIDIA Performance Libraries для
вычислений на CPU. Установите их и добавьте в путь поиска библиотек:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Для этих двух библиотек под CPU индекс подходит. А вот его сборки torch — как
раз те, что дают отказ «no kernel image» выше.

### Источники wheel-пакетов, которые не подходят для JetPack 7.2

| Источник | Результат на Orin Nano Super |
|---|---|
| torch из `pypi.jetson-ai-lab.io/sbsa/cu130` | Собран под серверные ARM GPU. Импортируется, сообщает, что CUDA доступна, а затем падает с «no kernel image is available for execution on the device». |
| torch из `pypi.jetson-ai-lab.io/jp6/*` | Сборки под CUDA 12 и Python 3.10. На Python 3.12 из этого образа они не ставятся. |
| Контейнеры PyTorch для JetPack 6 | Инициализация CUDA падает с ошибкой 801 на хосте с JetPack 7. |
| Сборка torch из исходников | Работает, но на плате с 8 ГБ занимает часы и не нужна, когда установлены wheel-пакеты для CUDA 13. |

## DeepStream

Если нужен полноценный видеопайплайн, а не цикл на Python, экспортируйте с
`deepstream=True` и запускайте граф через `nvinfer`. У этого пути своя страница
— со сгенерированным конфигом `nvinfer`, сборкой парсера ограничивающих рамок и
известными ловушками: [DeepStream](/docs/export/deepstream).

Сам пайплайн DeepStream проверяли на дискретной видеокарте под x86, а не на
Jetson. Контракт экспорта от архитектуры не зависит, но прогон пайплайна на
aarch64 всё ещё не сделан.

## Не проверено

- Релизы JetPack, кроме 7.2, и релизы L4T, кроме R39.2.
- Платы Jetson, кроме Orin Nano Super 8 ГБ.
- Обучение на плате. Инференс и экспорт прогонялись, а обучение — нет.
- Движки INT8. Для этой платы есть только строки FP32 и FP16.
- Размеры батча больше 1. Все замеры выше — на батче 1.
