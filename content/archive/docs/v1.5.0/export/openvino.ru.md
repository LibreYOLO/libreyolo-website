---
title: OpenVINO
seo_title: Экспорт в OpenVINO IR из LibreYOLO
description: >-
  Конвертация модели LibreYOLO в OpenVINO IR: пара model.xml и model.bin, сжатие
  весов в FP16, INT8 через NNCF и инференс на CPU, GPU или NPU.
lead: >-
  OpenVINO IR — формат среды выполнения от Intel: граф model.xml рядом с блобом
  весов model.bin. LibreYOLO экспортирует промежуточный ONNX, конвертирует его
  через ov.convert_model и записывает metadata.yaml в ту же директорию.
keywords:
  - экспорт yolo в openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - квантизация nncf int8
  - openvino npu
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="openvino")
    mono: true
  - label: Записывает
    value: 'Директорию с model.xml, model.bin и metadata.yaml'
  - label: Дополнительно
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Загружается обратно
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: Формы
    value: 'Следуют за промежуточным ONNX: динамический батч при dynamic=True'
  - label: Точность
    value: >-
      FP32, сжатие весов в FP16 (half=True), INT8 через NNCF (int8=True вместе с
      data=)
verification: >-
  Прочитано из libreyolo/export/openvino.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/openvino.py и pyproject.toml в
  ветке dev.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        # IR конвертируется из промежуточного ONNX, поэтому нужны оба extra.
        pip install "libreyolo[onnx,openvino]"
    - label: Для INT8 дополнительно нужен NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает директорию weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True сохраняет динамическую ось батча в IR
            half=False,       # True сохраняет веса в FP16
            int8=False,       # True запускает квантизацию NNCF после обучения
            data=None,        # обязателен при int8=True
            output_path=None, # None записывает weights/<stem>_openvino
        )
  int8:
    - label: INT8 с калибровочными данными
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # обязателен: у этого формата нет значения по умолчанию
            fraction=1.0,
        )
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Выбор устройства
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto" и "cpu" отображаются на CPU, "gpu" и "cuda" — на GPU,
        # всё остальное передаётся в верхнем регистре, например "npu" -> NPU.
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: Чистый OpenVINO
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # Имена классов, задача и размер входа лежат в metadata.yaml рядом с IR.

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Предобработка и постобработка на этом пути — на вас.
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## Установка

<code-tabs name="install" />

Конвертация идёт через промежуточный ONNX, поэтому extra `onnx` входит в состав
требований, а не остаётся опциональным дополнением. NNCF ставится отдельно и
нужен только для `int8=True`.

## Экспорт

<code-tabs name="export" />

Артефакт — это директория, а не файл. В `weights/LibreYOLO9t_openvino` лежат
`model.xml`, `model.bin` и `metadata.yaml`, а при `half=True` перед суффиксом
вставляется `_fp16`. Переносить или копировать нужно директорию целиком; эти три
файла — один артефакт.

`half=True` выставляет `compress_to_fp16` при сохранении. Это сжатие весов внутри
IR, а не изменение точности инференса, которую устройство выбирает во время
выполнения.

### INT8

<code-tabs name="int8" />

`int8=True` запускает квантизацию NNCF после обучения по калибровочному
загрузчику LibreYOLO с пресетом mixed, и `data` обязателен: у этого формата нет
запасного варианта из восьми изображений. Если NNCF не установлен, поднимается
`ImportError`, в котором названа команда установки.

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` распознаёт любую директорию, в которой есть `model.xml`, и
возвращает тот же объект `Results`, что и чекпойнт, считывая имена классов,
задачу, размер входа и схему позы из `metadata.yaml`.

Строка устройства не передаётся напрямую, а отображается на цель. `auto` и `cpu`
компилируются под CPU, `gpu` и `cuda` — под GPU, а любое другое значение
переводится в верхний регистр и отдаётся в OpenVINO; так и задаётся цель NPU.

Третий сниппет — для читателей, у которых LibreYOLO не установлен. Предобработка,
декодирование, NMS и пересчёт координат там ложатся на вас, а имена классов
существуют только в `metadata.yaml`.

## Ограничения

IR без своего `metadata.yaml` всё равно загрузится, но бэкенд тогда откатывается
к 80 классам и задаче детекции, что неверно для всего остального. Директорию
нужно держать целой.

Заблокировано до трассировки: сегментация YOLO9, сегментация RTMDet-Ins, детекция
SSD, Faster R-CNN и RetinaNet, а также маттинг BiRefNet или FeyNobg — там
OpenVINO 2026.2 не может преобразовать стандартную ONNX-операцию `DeformConv-19`
из общего декодера матта.

Если комбинация не проверена и не заблокирована, путь через конвертер доступен, а
паритет со средой выполнения OpenVINO для неё в проекте не зафиксирован.
Несколько комбинаций проверены с явно указанным контекстом: например,
семантическая сегментация DeepLabV3 при фиксированном входе 520 на 520 в
OpenVINO 2026.2 с точностью инференса CPU по умолчанию и оценка взгляда L2CS при
фиксированном кропе лица 448 на 448. `libreyolo formats` печатает этот контекст
для каждой комбинации.

Полную таблицу семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). Для одной комбинации:

<code-tabs name="support" />
