---
title: Core ML
seo_title: Експорт у Core ML з LibreYOLO
description: >-
  Експорт детектора LibreYOLO у пакет Core ML .mlpackage: контракт входу
  ImageType, FP16, обчислювальні блоки, вбудований NMS і чотири підтримувані
  сімейства.
lead: >-
  Core ML є форматом моделей Apple для роботи на пристрої. LibreYOLO трасує
  детектор за обгорткою попередньої обробки, специфічною для кожного сімейства,
  тому конвертований граф завжди приймає канонічний вхід у вигляді
  RGB-зображення, а потім записує пакет .mlpackage у форматі ML Program з
  приєднаними метаданими моделі.
keywords:
  - експорт yolo у coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml nms pipeline
last_verified: 1.5.0
meta:
  - label: Прапорець
    value: export(format="coreml")
    mono: true
  - label: Записує
    value: Один пакет .mlpackage (каталог) у форматі ML Program
  - label: Додатково
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Зворотне завантаження
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") на macOS
    mono: true
  - label: Форми
    value: Фіксовані. Вхід є жорстко заданим ct.ImageType.
  - label: Точність
    value: 'FP32, FP16 (half=True). Без INT8.'
  - label: Сімейства
    value: 'Лише виявлення, для yolox, yolo9, rtdetr і rfdetr'
verification: >-
  Прочитано з libreyolo/export/coreml.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/coreml.py та pyproject.toml у
  гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записує пакет weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True конвертує з обчислювальною точністю FLOAT16
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None записує weights/<stem>.mlpackage
        )

        # dynamic приймається, але вхід є ct.ImageType з фіксованою формою,
        # і вбудовані метадані в будь-якому разі записують dynamic=False.
  nms:
    - label: Вбудувати шар NMS від Apple
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Лише виявлення для YOLOX і YOLO9, батч 1.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: 'Через LibreYOLO, на macOS'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # або cpu_and_ne, щоб закріпити Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Чистий coremltools
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # Вхід є зображенням з назвою "image" у фіксованому розмірі експорту.
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # Летербоксинг і постобробка на цьому шляху лежать на вас.
  support:
    - label: Перевірити сімейство і задачу перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## Встановлення

<code-tabs name="install" />

Передбачення потребує macOS. `LibreYOLO()` відхиляє `.mlpackage` на будь-якій
іншій платформі з повідомленням, яке називає поточну, а матриця підтримки
позначає ці комбінації як доступні, виходячи з того, що для паритету середовища
виконання потрібен раннер на macOS.

## Експорт

<code-tabs name="export" />

Пакет записується в `weights/` з основою імені контрольної точки, з доданим
`_fp16`, коли `half=True`. `.mlpackage` є каталогом, тому копіюйте все дерево.

Кожне сімейство трасується за обгорткою попередньої обробки, тому конвертований
граф приймає один канонічний вхід: RGB, `scale=1/255`, без зсуву, оголошений як
`ct.ImageType`. Обгортка вбирає власну конвенцію сімейства: BGR у діапазоні від
0 до 255 для YOLOX, середнє та стандартне відхилення ImageNet для RF-DETR, і
тотожність для YOLO9 та RT-DETR. Саме тому споживач Core ML подає звичайне
зображення, а не тензор, специфічний для сімейства.

Конвертація націлена на ML Program з мінімальною ціллю розгортання iOS 15.
`compute_units` зберігається у конвертованій моделі, і його можна перевизначити
знову під час завантаження артефакту.

Метадані моделі потрапляють до `user_defined_metadata` як рядки, і саме звідти
бекенд читає сімейство, задачу, назви класів, розмір входу та схему пози.

### Вбудований NMS

<code-tabs name="nms" />

`nms=True` загортає модель у пайплайн Core ML, який завершується шаром
`NonMaximumSuppression` від Apple. Результат має два виходи: `confidence` з
формою `N` на кількість класів і `coordinates` з формою `N` на 4 як нормалізовані
`xywh`.

Це стосується лише виявлення для YOLOX і YOLO9 та вимагає батчу 1. Сімейства у
стилі DETR відхиляються за назвою, бо передбачення множини бере top-k за
запитами та класами без кроку IoU і не може використати цей шар. `max_det` тут
теж не виноситься назовні; коли важливе обмеження на кількість виявлень,
скористайтеся [вбудованим NMS в ONNX](/docs/export/onnx).

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` розпізнає каталог із суфіксом `.mlpackage` і повертає той самий
об'єкт `Results`, що й контрольна точка. `compute_units` є єдиним аргументом,
який фабрика передає далі для цього формату, і він приймає `all`,
`cpu_and_gpu`, `cpu_and_ne` та `cpu_only`. Аргумент `device` ігнорується, бо
Core ML натомість маршрутизує через обчислювальні блоки.

Другий фрагмент показує шлях через чисте середовище виконання. Летербоксинг,
декодування, NMS і перемасштабування координат стають вашою відповідальністю, а
назви класів лежать у `user_defined_metadata`.

## Обмеження

Чотири сімейства, лише виявлення: `yolox`, `yolo9`, `rtdetr` і `rfdetr`. Усе
інше відхиляється на етапі попередньої перевірки, бо саме обгортка попередньої
обробки, яка знає сімейство, робить контракт фіксованого вхідного зображення
правильним, а сімейство поза цим списком конвертувалося б з неправильною
нормалізацією. У повідомленні про помилку як альтернативи названо ONNX і
TorchScript.

Форму входу жорстко фіксує `ct.ImageType`, тому `dynamic=True` нічого не змінює,
а метадані записують `dynamic=False`. Для другої роздільної здатності
експортуйте другий пакет.

`half=True` конвертує з обчислювальною точністю FP16. Шляху до INT8 з цього
експортера немає.

Повну сітку сімейств і задач дивіться у
[матриці експорту](/docs/reference/export-matrix). Новіший формат Apple для
роботи на пристрої описано в [Core AI](/docs/export/coreai). Для однієї
комбінації:

<code-tabs name="support" />
