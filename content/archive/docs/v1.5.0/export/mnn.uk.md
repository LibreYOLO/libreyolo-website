---
title: MNN
seo_title: Експорт до MNN з LibreYOLO
description: >-
  Експортуйте детектор LibreYOLO до MNN через ONNX і mnnconvert: фіксована форма
  NCHW, FP32 на CPU та супровідні метадані, потрібні контракту середовища
  виконання.
lead: >-
  MNN є полегшеним рушієм інференсу Alibaba. LibreYOLO експортує статичний граф
  ONNX, перетворює його інструментом mnnconvert із пакета MNN і записує
  супровідний файл JSON із назвами входів та виходів, фіксованою формою входу й
  назвами класів.
keywords:
  - експорт yolo mnn
  - mnnconvert
  - mnn інференс
  - інференс мобільного детектора
  - фіксована форма nchw
last_verified: 1.5.0
meta:
  - label: Прапорець
    value: export(format="mnn")
    mono: true
  - label: Створює
    value: Один файл .mnn і супровідний файл метаданих .mnn.json
  - label: Додатково
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Повторне завантаження
    value: LibreYOLO("weights/LibreYOLO9t.mnn")
    mono: true
  - label: Форми
    value: Фіксовані NCHW. dynamic=True відхиляється.
  - label: Точність
    value: Лише FP32 і лише CPU.
  - label: Завдання
    value: У цій версії лише виявлення
verification: >-
  Перевірено за файлами libreyolo/export/mnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/mnn.py і pyproject.toml у
  гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: >
        # Додатковий набір містить libreyolo[onnx]: MNN перетворює проміжний
        файл ONNX.

        pip install "libreyolo[mnn]"
    - label: Перевірити наявність конвертера в шляху
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Створює weights/LibreYOLO9t.mnn і weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Аргументи
      language: python
      code: >
        model.export(
            format="mnn",
            imgsz=640,        # ціле число або (висота, ширина)
            batch=1,          # вбудовується в артефакт
            simplify=True,    # onnxsim для проміжного файлу ONNX
            output_path=None, # None створює weights/<stem>.mnn
            verbose=False,    # True транслює журнал mnnconvert
        )


        # dynamic=True спричиняє ValueError. half=True та int8=True
        відхиляються.
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Безпосередньо через MNN
      language: python
      code: >
        import json


        import MNN

        import numpy as np


        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))

        print(meta["mnn_input_names"], meta["mnn_output_names"],
        meta["mnn_input_shape"])


        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )

        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )


        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)

        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )

        outputs = module.forward([input_var])

        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # На цьому шляху попередня та подальша обробка покладаються на вас.
  support:
    - label: Перевірити сімейство й завдання перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 68fad34d07aea149
---

## Встановлення

<code-tabs name="install" />

Додатковий набір залежностей містить `libreyolo[onnx]`, оскільки перетворення
виконується через проміжний файл ONNX. Він також установлює виконуваний файл
`mnnconvert`, який експортер спочатку шукає поряд з активним інтерпретатором
Python, а потім у `PATH`. Якщо конвертера немає, виникає `ImportError` із назвою
команди встановлення замість збою посеред перетворення.

## Експорт

<code-tabs name="export" />

Перед передаванням графа експортер читає контракт входу ONNX і відхиляє все, що
не може виразити: більше одного входу зображення або форму входу із символьною
розмірністю. MNN у цій версії потребує повністю фіксованої форми NCHW, а `batch`
вбудовується в артефакт, а не узгоджується під час завантаження.

Супровідний файл не є необов'язковим службовим записом.
`weights/LibreYOLO9t.mnn.json` містить назви входів і виходів, фіксовану форму
входу, батч, назви класів, використану версію MNN і бекенд, для якого побудовано
артефакт. Середовище виконання перевіряє кожне з цих полів під час завантаження.

У Windows MNN 3.6.1 іноді завершує перетворення, а потім аварійно закривається
під час завершення процесу з порушенням доступу або станом fail-fast. Експортер
розпізнає ці конкретні коди виходу й вважає перетворення успішним, якщо вихідний
файл наявний.

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` визначає маршрут за суфіксом `.mnn` і повертає той самий об'єкт
`Results`, що й контрольна точка. Завантаження навмисно суворе: супровідний файл
має оголосити `format=mnn`, `mnn_backend=cpu`, `dynamic=false`,
`precision=fp32`, розмір, завдання виявлення, фіксовану додатну форму NCHW, що
узгоджується із записаним розміром зображення, і назви класів для кожного індексу
від 0 до `nc - 1`. Будь-яка невідповідність спричиняє помилку без спроб угадати
значення.

Передбачення з `imgsz`, відмінним від використаного для побудови артефакту,
також спричиняє помилку, а `device` ігнорується з попередженням, оскільки тут
експортовані моделі MNN працюють на CPU.

Другий фрагмент показує шлях безпосередньо через середовище виконання. На цьому
шляху попередня обробка, декодування, NMS і масштабування координат покладаються
на вас, а назви входів і виходів беруться із супровідного файлу, оскільки
завантажувачу модулів MNN їх потрібно передати явно.

## Обмеження

Лише виявлення. Бекенд відмовляється завантажувати будь-яке інше завдання, і
сторона експорту поводиться так само: за межами записаних комбінацій попередня
перевірка спричиняє помилку «MNN v1 не має реалізованого контракту середовища
виконання для цього сімейства й завдання».

FP32, CPU і фіксована форма. `dynamic=True` спричиняє `ValueError`, а `half=True`
та `int8=True` відхиляються під час валідації.

Валідовані сімейства виявлення: YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM і YOLO-NAS. Кожне з них охоплено
перетворенням, повторним завантаженням свіжого артефакту, виконанням MNN на CPU,
перевірками метаданих і зіставленим паритетом виявлень після NMS із моделлю
PyTorch. DEIMv2 перетворюється, повторно завантажується, виконується та зберігає
виявлення після NMS, але його проміжний шлях ONNX має неповний паритет оцінок на
рівні запитів, тому його записано як доступний, а не валідований.

Повну таблицю сімейств і завдань наведено в
[матриці експорту](/docs/reference/export-matrix). Для однієї комбінації:

<code-tabs name="support" />
