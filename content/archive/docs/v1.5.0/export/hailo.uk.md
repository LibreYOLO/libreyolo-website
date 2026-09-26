---
title: Hailo
seo_title: Запуск моделей LibreYOLO на прискорювачах Hailo
description: >-
  Розгортання моделі LibreYOLO на Hailo-8 або Hailo-8L: статичний експорт ONNX,
  самостійний етап у Dataflow Compiler та архітектури, які можна скомпілювати.
lead: >-
  Прискорювачі Hailo програмують за допомогою Hailo Dataflow Compiler,
  пропрієтарного SDK, який розповсюджується через Developer Zone компанії Hailo.
  На боці LibreYOLO процес обмежується звичайним статичним експортом ONNX;
  подальший синтаксичний аналіз, квантування та компіляцію у HEF виконує DFC.
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - компіляція hef
  - hailortcli
last_verified: 1.5.0
meta:
  - label: Етап LibreYOLO
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: Не формат
    value: Формату format="hef" немає. DFC не може бути залежністю pip.
  - label: Додатково
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Хост компіляції
    value: >-
      Linux x86_64, зокрема WSL2 Ubuntu 22.04. Компіляція не може виконуватися
      на ARM.
  - label: Компілюється
    value: >-
      Чисті CNN із графами фіксованої форми. Не компілюються архітектури з
      увагою, динамічними формами та переважним використанням LayerNorm.
  - label: Стан
    value: Ще жодне сімейство LibreYOLO не пройшло весь процес у DFC до запуску HEF.
verification: >-
  Перевірено за файлами skills/libreyolo-export-hailo/SKILL.md,
  libreyolo/export/onnx.py та libreyolo/cli/commands/export.py у гілці dev.
  Обмеження DFC взято із зазначеної навички; жоден HEF для LibreYOLO ще не було
  скомпільовано та виміряно.
snippets:
  install:
    - label: На боці LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'На боці Hailo, встановлюється самостійно'
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Для Hailo потрібні батч 1, фіксована роздільна здатність і відсутність
        динамічних осей.

        # В API Python типовим є dynamic=True, тому вимкніть його явно.

        model = LibreYOLO("LibreYOLOXs.pt")

        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # У CLI типовими вже є статичні форми.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Перевірити статичність графа перед компіляцією
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 'Синтаксичний аналіз, квантування та компіляція'
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # Для YOLOX спочатку виконайте перетворення без end_node_names: у
        журналі DFC

        # буде виведено запропоновані кінцеві вузли. Запустіть ще раз із ними.

        runner.translate_onnx_model(ONNX)


        # Нормалізація має відповідати попередній обробці LibreYOLO. Для YOLOX і
        YOLO9

        # не потрібні середнє чи стандартне відхилення, лише масштабування 0-255
        у 0-1.

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # Необов'язково: доручіть NMS платформі Hailo. Конфігурація залежить і
        від

        # кількості класів, і від розміру вхідних даних, тому конфігурація
        COCO-80

        # неправильна для донавченої моделі з трьома класами. Без цього рядка
        HEF

        # видає необроблені тензори голови, а застосунок декодує їх.

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # Калібрувальні зображення мають відповідати даним розгортання.

        # Випадкові зображення дають змогу скомпілювати модель, але непомітно
        руйнують правильність.

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: Кінцеві вузли YOLO9
      language: python
      code: >
        # Графи LibreYOLO використовують префікс "/head/...", а не "model.N",

        # який трапляється в конфігураціях для інших експортів. Скопійована
        конфігурація

        # не збігатиметься. Якщо синтаксичний аналіз не вдається, перевірте
        назви у власному графі.

        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]

        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 з AI Kit або AI HAT+
      language: bash
      code: >
        sudo apt install dkms hailo-all

        hailortcli fw-control identify       # перевірка пристрою та визначення
        архітектури

        hailortcli run libreyoloxs.hef       # базова перевірка та пропускна
        здатність
source_hash: 33b077f1c23d5535
---

## Встановлення

У LibreYOLO немає `format="hef"`, і його не буде. Hailo Dataflow Compiler є
пропрієтарним SDK, що розповсюджується як приватний wheel після реєстрації в
Developer Zone, тому він не може бути залежністю чи додатковим пакетом. Розгортання
має два етапи: LibreYOLO записує статичний файл ONNX, після чого ви запускаєте для
нього DFC.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Експорт

<code-tabs name="export" />

Не передавайте `half=True`. DFC приймає ONNX у FP32 та сам виконує квантування
INT8. Також не передавайте `nms=True`: NMS виконує або Hailo через
`nms_postprocess`, або застосунок, а підграф NMS після кінцевих вузлів лише
зайвий. Типовий opset працює; якщо синтаксичний аналізатор DFC заперечує,
повторіть експорт із `opset=11`.

DFC обрізає граф на переданих кінцевих вузлах, якими є згортки голови детектора,
і відкидає все після них. Тому звичайний декодований ONNX від LibreYOLO є
прийнятним вхідним файлом: синтаксичний аналізатор просто ігнорує хвіст декодування.

## Компіляція

<code-tabs name="compile" />

Виберіть `hw_arch` відповідно до цільового пристрою: `hailo8` для Hailo-8,
AI HAT+ на 26 TOPS та модулів M.2 і PCIe; `hailo8l` для Hailo-8L, Raspberry Pi
AI Kit та AI HAT+ на 13 TOPS; `hailo10h` для Hailo-10H, якому потрібні відповідні
новіші версії DFC і Model Zoo. Якщо маєте сумніви, відповідь на пристрої надасть
команда `hailortcli fw-control identify`.

Два сімейства відповідають метаархітектурі NMS у HailoRT, тому Hailo може виконувати
пригнічення всередині скомпільованого пайплайна: YOLOX через `meta_arch=yolox`, а
YOLO9 через метаархітектуру Hailo з розділеною головою, компонування якої ідентичне.
Візьміть відповідну конфігурацію `nms_postprocess` із Hailo Model Zoo та адаптуйте
її до кількості класів і розміру вхідних даних. Усі інші згорткові детектори
компілюються як граф без відповідної метаархітектури: HEF видає необроблені тензори
голови, а застосунок виконує декодування та NMS на CPU.

Якщо щось не вдається, збережіть журнал компіляції. Кожне виправлення залежить від
точної назви шару або оператора, на якому стався збій.

## Запуск артефакту

<code-tabs name="device" />

Для інференсу в застосунку використовується API Python `hailo_platform`. Якщо
скомпільовано `nms_postprocess`, вихід має форму `(batch, num_classes, max_dets, 5)`
і містить `[y1, x1, y2, x2, score]` у координатах моделі, які потрібно самостійно
масштабувати до початкового зображення. Пайплайн `Results` бібліотеки LibreYOLO під
час виконання не залучається; HEF є самостійним артефактом, а попередню та подальшу
обробку виконує застосунок.

## Обмеження

Можливість націлити модель на Hailo-8 або Hailo-8L визначається її архітектурою,
а не назвою, тому наведене нижче правило стосується й сімейств, доданих після
написання цієї сторінки.

Модель не скомпілюється, якщо містить щось із наведеного нижче:

- Увагу будь-якого типу: самоувагу, перехресну, деформовану або віконну увагу.
  Це виключає всі детектори в стилі DETR, усі детектори з відкритим словником або
  текстовими умовами, усі бекбони ViT, а також усі мовні й візуально-мовні модулі.
  Власний каталог Hailo містить кілька вручну оптимізованих HEF для трансформерів;
  це спеціальна робота постачальника, яка не доводить, що довільний граф з увагою
  можна скомпілювати.
- Динамічні форми або залежний від даних потік керування. DFC компілює одну
  фіксовану форму вхідних даних і статичний граф, тому змінна кількість запитів,
  текстові підказки, динамічний top-k, `NonZero`, `Gather` або `TopK` із динамічними
  індексами та `grid_sample` не підтримуються.
- Архітектуру з переважним використанням LayerNorm або GELU. BatchNorm добре
  згортається у згортки; підтримка LayerNorm обмежена, а GELU не є нативною
  активацією, тому стек у стилі ConvNeXt погано підходить, хоча формально й
  залишається згортковим.
- Обробку зображення в зображення з початковою роздільною здатністю. Моделі
  відновлення працюють із повною роздільною здатністю вхідних даних і перевищують
  практичні обмеження SRAM у Hailo.

Сімейство можна розглядати як кандидата, якщо воно використовує лише згортки,
BatchNorm із ReLU або SiLU та фіксований розмір вхідних даних. У цій бібліотеці
це одностадійні детектори CNN, насамперед YOLOX і YOLO9; інші згорткові детектори,
як-от PicoDet, YOLO-NAS та RTMDet, із декодуванням на боці застосунку; класифікатори
CNN ResNet, MobileNetV4-conv та EfficientNetV2, серед яких ResNet має найкращу
підтримку, оскільки Hailo Model Zoo надає рецепти для нього; а також невеликі
згорткові голови завдань, як-от точкове виявлення FOMO та оцінювання погляду L2CS
на бекбоні ResNet, які теоретично можна скомпілювати, але для них немає рецепта Hailo.

Варто врахувати стан підтримки, через який на цій сторінці нічого не позначено як
підтримуване: ще жодне сімейство LibreYOLO не пройшло весь процес у DFC до запуску
HEF. Наведені правила прогнозують можливість компіляції за архітектурою. Поведінка
синтаксичного аналізатора, квантування та правильність залишаються неперевіреними,
доки HEF не буде скомпільовано й виміряно. Тому кожен кандидат потребує власних
зафіксованих доказів: скомпільованого HEF із точної контрольної точки із записаними
версіями DFC, Model Zoo та HailoRT, задокументованого калібрування й порівняння
правильності на пристрої з базовим рівнем FP32, а не лише показника пропускної
здатності.

Якщо модель не відповідає вимогам, використовуйте середовища виконання з
підтвердженою еквівалентністю: [ONNX](/docs/export/onnx),
[TensorRT](/docs/export/tensorrt) та [OpenVINO](/docs/export/openvino).

