---
title: MNN
seo_title: Экспорт в MNN из LibreYOLO
description: >-
  Экспорт детектора LibreYOLO в MNN через ONNX и mnnconvert: фиксированная форма
  NCHW, FP32 на CPU и сопроводительный файл метаданных, которого требует
  контракт среды выполнения.
lead: >-
  MNN — это лёгкий движок инференса от Alibaba. LibreYOLO экспортирует
  статический ONNX-граф, конвертирует его инструментом mnnconvert из пакета MNN
  и записывает рядом сопроводительный JSON-файл, в котором сохранены имена
  входов и выходов, фиксированная форма входа и имена классов.
keywords:
  - экспорт yolo в mnn
  - mnnconvert
  - mnn инференс
  - инференс детектора на мобильных
  - фиксированная форма nchw
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="mnn")
    mono: true
  - label: Записывает
    value: Один файл .mnn плюс сопроводительный файл метаданных .mnn.json
  - label: Дополнительно
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Загружается обратно
    value: LibreYOLO("weights/LibreYOLO9t.mnn")
    mono: true
  - label: Формы
    value: Фиксированные NCHW. dynamic=True отклоняется.
  - label: Точность
    value: 'Только FP32, только CPU.'
  - label: Задачи
    value: В этой версии только детекция
verification: >-
  Прочитано из libreyolo/export/mnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/mnn.py и pyproject.toml в
  ветке dev.
snippets:
  install:
    - label: Установка
      language: bash
      code: >
        # Дополнение включает libreyolo[onnx]: MNN конвертирует из
        промежуточного ONNX.

        pip install "libreyolo[mnn]"
    - label: 'Проверка, что конвертер доступен в PATH'
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает weights/LibreYOLO9t.mnn и weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int или (height, width)
            batch=1,          # зашивается в артефакт
            simplify=True,    # onnxsim по промежуточному ONNX
            output_path=None, # None записывает weights/<stem>.mnn
            verbose=False,    # True выводит лог mnnconvert
        )

        # dynamic=True вызывает ValueError. half=True и int8=True отклоняются.
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: MNN напрямую
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

        # Предобработка и постобработка на этом пути — на вас.
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 68fad34d07aea149
---

## Установка

<code-tabs name="install" />

Дополнение включает `libreyolo[onnx]`, потому что конвертация идёт через
промежуточный ONNX. Вместе с ним ставится исполняемый файл `mnnconvert` —
экспортёр ищет его сначала рядом с активным интерпретатором Python, а затем в
`PATH`. Если конвертера нет, вместо падения посреди конвертации поднимается
`ImportError` с названием команды установки.

## Экспорт

<code-tabs name="export" />

Прежде чем передать граф дальше, экспортёр читает контракт входа ONNX и
отклоняет всё, что не может выразить: больше одного входа-изображения или форму
входа с символьной размерностью. MNN в этой версии требует полностью
фиксированную форму NCHW, а `batch` зашивается в артефакт, а не согласуется при
загрузке.

Сопроводительный файл — не необязательная бухгалтерия. В
`weights/LibreYOLO9t.mnn.json` записаны имена входов и выходов, фиксированная
форма входа, батч, имена классов, использованная версия MNN и бэкенд, под
который собран артефакт, и среда выполнения проверяет каждое из этих полей при
загрузке.

На Windows MNN 3.6.1 иногда завершает конвертацию, а потом падает при выгрузке
процесса с access violation или статусом fail-fast. Экспортёр распознаёт эти
конкретные коды выхода и считает конвертацию успешной, если выходной файл на
месте.

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` выбирает путь по суффиксу `.mnn` и возвращает тот же объект
`Results`, что и чекпойнт. Загрузка намеренно строгая: в сопроводительном файле
должны быть объявлены `format=mnn`, `mnn_backend=cpu`, `dynamic=false`,
`precision=fp32`, размер, задача детекции, фиксированная положительная форма
NCHW, согласованная с записанным размером изображения, и имена классов,
покрывающие каждый индекс от 0 до `nc - 1`. Любое несовпадение приводит к
ошибке, а не к догадкам.

Предсказание с `imgsz`, отличным от того, под который собран артефакт, тоже
приводит к ошибке, а `device` игнорируется с предупреждением, потому что
экспортированные в MNN модели здесь работают на CPU.

Второй сниппет — это путь через среду выполнения напрямую. Предобработка,
декодирование, NMS и пересчёт координат становятся там вашей задачей, а имена
входов и выходов берутся из сопроводительного файла, потому что загрузчик
модулей MNN требует их явно.

## Ограничения

Только детекция. Бэкенд отклоняет при загрузке любую другую задачу, и сторона
экспорта ведёт себя так же: вне записанных комбинаций preflight падает с
сообщением «MNN v1 has no implemented runtime contract for this family and
task».

FP32, CPU, фиксированная форма. `dynamic=True` вызывает `ValueError`, а
`half=True` и `int8=True` отклоняются при валидации.

Проверенные семейства детекции — YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM и YOLO-NAS; для каждого пройдены
конвертация, повторная загрузка свежего артефакта, выполнение MNN на CPU,
проверки метаданных и совпадение детекций после NMS с моделью PyTorch. DEIMv2
конвертируется, загружается, выполняется и сохраняет детекции после NMS, но у
его промежуточного маршрута через ONNX неполное совпадение оценок на уровне
запросов, поэтому он записан как доступный, а не как проверенный.

Полную сетку семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). Для одной комбинации:

<code-tabs name="support" />
