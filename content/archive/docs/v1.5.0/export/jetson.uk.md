---
title: NVIDIA Jetson
seo_title: Встановлення LibreYOLO та PyTorch на NVIDIA Jetson
description: >-
  Встановлення LibreYOLO на NVIDIA Jetson: чотири бібліотеки CUDA, яких немає в
  JetPack, потрібний для PyTorch етап із --no-deps та виміряні показники Orin
  Nano.
lead: >-
  Плати NVIDIA Jetson запускають LibreYOLO на стандартних wheel-пакетах PyTorch
  для aarch64. Спеціальна збірка torch для Jetson не потрібна, але JetPack не
  містить чотирьох бібліотек, з якими скомпоновано torch, тому їх потрібно
  додати під час встановлення.
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - встановити PyTorch на Jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - TensorRT на Jetson
  - wheel aarch64
last_verified: 1.4.0
meta:
  - label: Плата
    value: >-
      Jetson Orin Nano Super Developer Kit, 8 GB, обчислювальна здатність GPU
      8.7
  - label: Платформа
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: Перевірений стек
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1, станом на 2026-07-27
  - label: Відсутнє в JetPack
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: Бенчмарки
    value: >-
      223 перевірені запуски на цій платі, 58 моделей із 12 сімейств у PyTorch,
      ONNX Runtime та TensorRT
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: Відстежується в
    value: 'Частина issue 648, що стосується Jetson'
    links:
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  Рецепт встановлення й очікуваний результат взято із запуску встановлення
  2026-07-27 на Jetson Orin Nano Super. Рядки затримки та правильності взято зі
  знімка перевірених результатів, на якому ґрунтується visionanalysis.org, із
  фільтром за обладнанням jetson_orin; вимірювання виконано в червні 2026 року
  на libreyolo 1.2.0.dev0. Поведінку експорту й завантажувача перевірено за
  файлами libreyolo/export/exporter.py, libreyolo/export/tensorrt.py та
  libreyolo/models/__init__.py.
snippets:
  prep:
    - label: Системні пакети та віртуальне середовище
      language: bash
      code: |
        # JetPack не встановлює наперед pip або модуль venv.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: PyTorch з індексу wheel-пакетів CUDA 13
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: 'Чотири бібліотеки, яких немає в JetPack'
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: 'Якщо pip вимагає cuda-toolkit 13.0.3, встановити з --no-deps'
      language: bash
      code: >
        # --no-deps означає, що залежності Python для torch також зазначаються
        вручну.

        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: Визначити наступну відсутню бібліотеку без здогадок
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Усі досі відсутні залежності в усіх бібліотеках torch за один прохід:

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 'Встановити LibreYOLO після torch, а не до нього'
      language: bash
      code: >
        # torch уже задоволено, тому pip зберігає збірку CUDA.

        pip install libreyolo


        # Додатковий пакет ONNX потрібен лише для експорту. Експорт TensorRT

        # виконується через ONNX, тому додайте його перед розділом експорту
        нижче.

        pip install "libreyolo[onnx]"
  verify:
    - label: Версії та пристрій
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
    - label: Потім запустити справжнє ядро
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

        # Завантажує контрольну точку під час першого використання.
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


        # Записує libreyolo9s.onnx, а потім будує з нього libreyolo9s.engine.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # Рушій повторно завантажується через ту саму точку входу.

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Режим живлення та частоти
      language: bash
      code: >
        sudo nvpmodel -q      # доступні на цій платі режими та активний режим

        sudo nvpmodel -m 0    # найвищий режим на перевіреній тут платі

        sudo jetson_clocks


        tegrastats            # навантаження наживо; nvidia-smi має обмеження на
        Tegra
source_hash: c07ff908503e89b5
---

## Що зафіксовано на цій сторінці

На цій сторінці зафіксовано одну конфігурацію, перевірену від початку до кінця,
а не матрицю підтримки. Використано плату Jetson Orin Nano Super Developer Kit
із 8 GB пам'яті під керуванням JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13,
Python 3.12.3), на якій запрацював стек `libreyolo 1.4.0` із
`torch 2.13.0+cu130`, OpenCV 5.0.0 та NumPy 2.5.1. Виклик
`torch.cuda.is_available()` повернув `True`, а GPU представився як `Orin`.

Інші випуски JetPack, інші плати Jetson та інші версії CUDA не перевірялися.
Наведений нижче рецепт спрацював саме для цього поєднання.

Цей запуск виконано 2026-07-27 з LibreYOLO 1.4.0 і не повторено на обладнанні з
версією 1.5.0: це єдина сторінка в дереві 1.5.0, де досі зазначено перевірку для
1.4.0, тому у frontmatter вказано `last_verified: "1.4.0"`. Жодна зміна 1.5.0
не стосується описаного тут шляху встановлення, чотирьох відсутніх бібліотек або
параметрів експорту, тож команди мають залишатися чинними. Проте номери версій
у наведених нижче вихідних даних отримано у 1.4.0, а не під час вимірювання 1.5.0.

Два аспекти суперечать більшості посібників для Jetson. Використовуються звичайні
збірки aarch64, опубліковані для CUDA 13, тому спеціальна збірка torch для Jetson
не потрібна. Водночас JetPack не містить чотирьох бібліотек, з якими скомпоновано
ці збірки, тому `import torch` послідовно повідомляє про відсутні бібліотеки,
доки не буде встановлено всі чотири.

## Встановлення

Образи JetPack постачаються без pip і модуля `venv`, тому спочатку потрібно
встановити їх.

<code-tabs name="prep" />

На платі з 8 GB пам'яті бракує ресурсів для більших контрольних точок. Додавання
swap на NVMe перед їх завантаженням запобігає завершенню процесу через нестачу
пам'яті посеред запуску.

Далі встановіть PyTorch. Індекс CUDA 13 містить wheel-пакети aarch64, а додатковий
індекс надає чисті залежності Python із PyPI.

<code-tabs name="torch" />

Чотири wheel-пакети `nvidia-*-cu13` легко пропустити. JetPack надає драйвер GPU,
але не cuDNN, NCCL, cuSPARSELt або NVSHMEM, і без них torch не імпортується.
Швидше встановити всі чотири одночасно, ніж виявляти їх по одному винятку.

Третій фрагмент стосується конкретної помилки: метадані залежностей torch для
збірки CUDA 13 вимагають `cuda-toolkit==13.0.3`, для якого немає wheel-пакета
aarch64 у PyPI, тому розв'язання залежностей завершується до початку завантаження.
Параметр `--no-deps` пропускає розв'язувач, а отже, кожну залежність потрібно
вказати в командному рядку.

LibreYOLO встановлюється останньою. Якщо встановити її першою, pip самостійно
вибере torch, який на цій платформі не буде збіркою CUDA.

<code-tabs name="install" />

Для всіх інших залежностей доступні готові wheel-пакети aarch64, зокрема для
OpenCV, NumPy, SciPy, pycocotools і safetensors. Нічого не компілюється з
початкового коду.

## Перевірка роботи CUDA

<code-tabs name="verify" />

Другий фрагмент не менш важливий за перший. Wheel-пакет, зібраний для іншої
архітектури GPU, усе одно повертає `torch.cuda.is_available() == True`, але перша
справжня операція завершується помилкою
`CUDA error: no kernel image is available for execution on the device`.
Множення матриць на пристрої виявляє цю проблему.

## Виконання передбачення

<code-tabs name="predict" />

`predict` повертає той самий об'єкт `Results`, що й на будь-якій іншій платформі,
тому сторінки моделей застосовуються без змін.

## Експорт у TensorRT

На цій платі TensorRT був швидшим за PyTorch і ONNX Runtime для всіх 55 моделей,
виміряних у кожному середовищі виконання.

<code-tabs name="export" />

Параметр `format="tensorrt"` спочатку записує граф ONNX, а потім будує з нього
рушій, тому потрібно встановити додатковий пакет `onnx`. `LibreYOLO()` виконує
диспетчеризацію за суфіксом файлу, тож файл `.engine` завантажується тим самим
викликом, що й контрольна точка `.pt`.

Не використовуйте додатковий пакет pip `tensorrt` на Jetson. Він закріплює
`tensorrt-cu12`, збірку CUDA 12, для платформи CUDA 13. Натомість використовуйте
TensorRT, встановлений JetPack. Якщо `import tensorrt` не працює у віртуальному
середовищі, але працює поза ним, відтворіть середовище з
`--system-site-packages`, щоб системний модуль був видимим.

Серіалізовані рушії TensorRT прив'язані до пристрою, архітектури GPU та версії
TensorRT, яка їх побудувала. Рушій, побудований на робочій станції, не
завантажиться на Jetson, тому етап побудови виконується на самій платі.

## Вимірювання на цій платі

Затримка на зображення з батчем 1, від початку до кінця разом із попередньою та
подальшою обробкою, на COCO val2017 (підмножина з 500 зображень) із `conf=0.001`
та `max_det=300`. П'ять моделей із 58 виміряних:

| Модель | Вхідні дані (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

У стовпці mAP наведено власну оцінку запуску TensorRT FP16. Серед 55 моделей,
виміряних у всіх чотирьох середовищах виконання, найбільша різниця між оцінкою
PyTorch FP32 та TensorRT FP16 становила 0.59 пункту для DEIMv2-X. Середовища
виконання відрізняються швидкістю, а не правильністю.

TensorRT FP32 був швидшим за PyTorch і ONNX Runtime для всіх цих 55 моделей.
TensorRT FP16 також був швидшим за PyTorch FP32 для всіх 55 моделей у 1.68x-6.22x
із медіаною 3.39x. Результати ONNX Runtime різняться: він був повільнішим за
PyTorch для 23 із 55 моделей, зокрема для RT-DETR-r18 у таблиці.

Умови всіх значень: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`, Python 3.12.3,
CUDA 13, драйвер 595.78, ONNX Runtime 1.24.0, вимірювання виконано в червні 2026
року. Затримка на Jetson також залежить від активного режиму живлення, якого
записи бенчмарків не містять.

<code-tabs name="power" />

Усі 223 запуски, зокрема інші 53 моделі й повні стовпці правильності,
опубліковано на
[сторінці Jetson Orin у Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Усунення несправностей

### import torch повідомляє про відсутню спільну бібліотеку

Відсутня одна з чотирьох наведених вище бібліотек. Не намагайтеся вгадати, а
прочитайте назву з бінарного файлу:

<code-tabs name="ldd" />

Кожному відсутньому запису відповідає один wheel-пакет:

| Відсутня бібліотека | Wheel-пакет |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELT | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch попереджає, що жодна збірка не підтримує цей GPU

Перший виклик CUDA у робочій конфігурації виводить таке повідомлення:

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

На цій платі попередження не впливає на роботу. Wheel-пакет містить ядра `sm_80`,
і Orin виконує їх. Таке саме попередження з'являлося з попереднім wheel-пакетом
із цього індексу, який створив кожен наведений вище рядок бенчмарку. Перевіряйте
роботу за допомогою множення матриць із перевірки CUDA, а не довіряйте чи не
довіряйте самому повідомленню.

### CUDA error: no kernel image is available for execution on the device

Установлений wheel-пакет зібрано для іншої архітектури GPU. Саме це відбувається
з wheel-пакетами з індексу `sbsa` компанії NVIDIA, які призначено для серверних
GPU ARM, а не кристалів Jetson. Повторно встановіть пакет з індексу CUDA 13,
наведеного в розділі встановлення.

### pip не може знайти cuda-toolkit 13.0.3

Для нього немає wheel-пакета aarch64. Скористайтеся формою з `--no-deps` із
розділу встановлення та явно зазначте залежності torch.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

Wheel-пакет torch для aarch64 скомпоновано з NVIDIA Performance Libraries для
обчислень на CPU. Встановіть їх і додайте до шляху бібліотек:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Цей індекс підходить для двох зазначених бібліотек CPU. Саме збірки torch із
нього спричиняють наведену вище помилку «no kernel image».

### Джерела wheel-пакетів, несумісні з JetPack 7.2

| Джерело | Результат на Orin Nano Super |
|---|---|
| torch із `pypi.jetson-ai-lab.io/sbsa/cu130` | Зібрано для серверних GPU ARM. Імпортується, повідомляє про доступність CUDA, а потім завершується помилкою «no kernel image is available for execution on the device». |
| torch із `pypi.jetson-ai-lab.io/jp6/*` | Збірки CUDA 12 і Python 3.10. Вони не встановлюються для Python 3.12 у цьому образі. |
| Контейнери PyTorch для JetPack 6 | Ініціалізація CUDA завершується помилкою 801 на хості з JetPack 7. |
| Побудова torch із початкового коду | Працює, але триває кілька годин на платі з 8 GB і не потрібна після встановлення wheel-пакетів CUDA 13. |

## DeepStream

Для повного відеопайплайна замість циклу Python експортуйте з
`deepstream=True` і запустіть граф через `nvinfer`. Цей шлях має окрему сторінку
з описом створеної конфігурації `nvinfer`, побудови аналізатора обмежувальних
рамок і відомих пасток: [DeepStream](/docs/export/deepstream).

Сам пайплайн DeepStream валідовано на дискретному GPU x86, а не на Jetson.
Контракт експорту не залежить від архітектури, але запуск пайплайна на aarch64
ще не перевірено.

## Не перевірено

- Випуски JetPack, крім 7.2, і випуски L4T, крім R39.2.
- Плати Jetson, крім Orin Nano Super 8 GB.
- Навчання на платі. Інференс та експорт перевірено, навчальний запуск не
  виконувався.
- Рушії INT8. Для цієї плати є лише рядки FP32 та FP16.
- Розміри батча понад 1. Усі наведені вище вимірювання виконано з батчем 1.

