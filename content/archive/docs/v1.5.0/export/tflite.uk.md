---
title: TFLite
seo_title: Експорт до TFLite (LiteRT) з LibreYOLO
description: >-
  Експортуйте модель LibreYOLO до FlatBuffer .tflite через onnx2tf: статичні
  форми, лише FP32, входи NHWC і сімейства, що перетворюються без помилок.
lead: >-
  TFLite є форматом FlatBuffer, який LiteRT виконує на мобільних і вбудованих
  цільових платформах. LibreYOLO експортує статичний граф ONNX, перетворює його
  за допомогою onnx2tf у режимі flatbuffer-direct і записує метадані моделі
  поряд з артефактом у супровідному файлі JSON.
keywords:
  - експорт yolo tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - вхід nhwc tflite
  - інференс на периферії
last_verified: 1.5.0
meta:
  - label: Прапорець
    value: export(format="tflite")
    mono: true
  - label: Створює
    value: Один файл .tflite і супровідний файл метаданих .tflite.json
  - label: Додатково
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Повторне завантаження
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: Форми
    value: Лише статичні. dynamic=True відхиляється.
  - label: Точність
    value: Лише FP32. half=True та int8=True відхиляються.
  - label: Потрібно
    value: >-
      Python 3.12 або новіша, оскільки onnx2tf 2.4.x не публікує пакунків для
      старіших версій
verification: >-
  Перевірено за файлами libreyolo/export/tflite.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py,
  libreyolo/backends/tflite.py і pyproject.toml у гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: >
        # LiteRT є сучасною назвою TensorFlow Lite від Google. Обидва набори

        # залежностей установлюють однаковий ланцюжок і створюють той самий файл
        .tflite.

        pip install "libreyolo[tflite]"
    - label: Спочатку перевірити версію Python
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Створює weights/LibreYOLO9t.tflite і weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" приймається як псевдонім і вибирає той самий експортер.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # ціле число або (висота, ширина)
            batch=1,
            simplify=True,    # onnxsim для проміжного файлу ONNX
            output_path=None, # None створює weights/<stem>.tflite
            verbose=False,    # True транслює журнал onnx2tf
        )

        # dynamic=True спричиняє ValueError: конвертеру потрібні статичні форми.
        # half=True та int8=True відхиляються перед трасуванням.
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Безпосередньо через LiteRT
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC, а не NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Назви класів, завдання й розмір входу зберігаються в супровідному
        файлі.

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Попередня обробка, транспонування NCHW у NHWC і подальша обробка
        покладаються на вас.
  support:
    - label: Перевірити сімейство й завдання перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## Встановлення

<code-tabs name="install" />

Додатковий набір установлює `onnx2tf` для перетворення й `ai-edge-litert` для
запуску результату, обидва з умовою Python 3.12. У старішому інтерпретаторі
експорт спричиняє `ImportError` із вимогою до версії замість збою всередині
конвертера.

`libreyolo[litert]` установлює те саме. Рядок формату `litert` є псевдонімом
`tflite`, і в обох випадках вихідний файл має розширення `.tflite`.

## Експорт

<code-tabs name="export" />

Сімейство й завдання перевіряються перед усіма іншими діями, тому непідтримувана
комбінація негайно спричиняє конкретну помилку конвертера або середовища
виконання, через яку її виключено, а не загальне повідомлення. Саме перетворення
виконується підпроцесом `onnx2tf` у режимі `flatbuffer_direct` для статичного
проміжного файлу ONNX.

Метадані зберігаються в супровідному файлі. `weights/LibreYOLO9t.tflite.json`
містить сімейство, завдання, назви класів, розмір входу та схему пози. Сам
FlatBuffer не має поля метаданих LibreYOLO, тому два файли переміщуються разом.

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` визначає маршрут за суфіксом `.tflite` і повертає той самий об'єкт
`Results`, що й контрольна точка. Бекенд читає супровідний файл, транспонує блоб
NCHW у NHWC, коли інтерпретатор очікує вхід із каналами в останній розмірності,
застосовує масштаб квантування й нульову точку інтерпретатора, якщо вони наявні,
та транспонує виходи назад до компонування, яке очікує подальша обробка LibreYOLO.

Другий фрагмент показує шлях безпосередньо через середовище виконання. На цьому
шляху попередня обробка, транспонування компонування, декодування, NMS і
масштабування координат покладаються на вас. Найлегше пропустити саме деталь
компонування: onnx2tf створює входи з каналами в останній розмірності, тому блоб
форми `(1, 3, 640, 640)` не приєднається.

## Обмеження

Лише статичні форми. `dynamic=True` спричиняє `ValueError` перед трасуванням, а
полотно експорту фіксується на значенні, до якого обчислено `imgsz`.

Лише FP32. `half=True` та `int8=True` відхиляються під час валідації, тому цей
експортер наразі не дає змоги розгорнути квантовану модель.

Тут охоплення вужче, ніж у форматах графів, і визначається вимірюваннями, а не
сімейством. До валідованих комбінацій належать виявлення YOLO9, YOLOX і YOLO-NAS,
семантична сегментація PIDNet, чотири сімейства класифікації CNN, ембединги
DINOv2 і SigLIP2, класифікація SigLIP2, виявлення країв TEED і DexiNed, а також
відновлення Real-ESRGAN і SwinIR. Для SwinIR є додаткове застереження: паритет
зберігається, коли розміри джерела точно відповідають полотну експорту. Менші
джерела доповнюються до полотна перед запуском трансформера, що може відрізнятися
від нативного інференсу зі змінним розміром.

У заблокованих записах указано точну причину збою, яку варто прочитати перед
спробою обхідного рішення. Кілька прикладів: виявлення RF-DETR перетворюється на
своєму нативному полотні 384, але LiteRT не може розподілити пам'ять, оскільки
`STRIDED_SLICE` отримує вхід із рангом понад підтримуваний 5-D. PicoDet
відхиляється, бо `RESHAPE` зіставляє 19,200 вхідних елементів із 9,600 вихідними.
D-FINE аварійно завершує конвертер під час обробки форми `GatherElements`.
RTMDet експортується та повторно завантажується зі збереженим паритетом сирих
виходів, але IoU публічних рамок падає до 0.911 із дрейфом координат 29.9 px.

Повну таблицю сімейств і завдань наведено в
[матриці експорту](/docs/reference/export-matrix). Для однієї комбінації разом
із текстом причини блокування:

<code-tabs name="support" />
