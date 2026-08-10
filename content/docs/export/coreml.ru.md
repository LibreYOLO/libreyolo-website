---
title: Core ML
seo_title: "Экспорт в Core ML из LibreYOLO"
description: "Экспорт детектора LibreYOLO в Core ML .mlpackage: контракт входа ImageType, FP16, compute units, встроенный NMS и четыре поддерживаемых семейства."
lead: "Core ML — это формат моделей Apple для работы на устройстве. LibreYOLO трассирует детектор за обёрткой предобработки, своей для каждого семейства, поэтому конвертированный граф всегда принимает канонический вход-изображение RGB, а затем записывает .mlpackage в формате ML Program с приложенными метаданными модели."
keywords:
  - экспорт yolo в coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml nms пайплайн
last_verified: "1.5.0"
meta:
  - label: Флаг
    value: 'export(format="coreml")'
    mono: true
  - label: Записывает
    value: "Один бандл .mlpackage (директория) в формате ML Program"
  - label: Дополнительно
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Загружается обратно
    value: 'LibreYOLO("weights/LibreYOLO9t.mlpackage") на macOS'
    mono: true
  - label: Формы
    value: "Фиксированные. Вход — жёстко заданный ct.ImageType."
  - label: Точность
    value: "FP32, FP16 (half=True). INT8 нет."
  - label: Семейства
    value: "Только детекция, для yolox, yolo9, rtdetr и rfdetr"
verification: "Прочитано из libreyolo/export/coreml.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/coreml.py и pyproject.toml в ветке dev."
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает бандл weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True конвертирует с вычислительной точностью FLOAT16
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None записывает weights/<stem>.mlpackage
        )

        # dynamic принимается, но вход — это ct.ImageType с фиксированной формой,
        # и встроенные метаданные в любом случае записывают dynamic=False.
  nms:
    - label: Встраивание слоя NMS от Apple
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Только детекция YOLOX и YOLO9, батч 1.
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
    - label: Через LibreYOLO, на macOS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # или cpu_and_ne, чтобы закрепить Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Голый coremltools
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # Вход — изображение с именем "image" фиксированного размера экспорта.
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # Леттербоксинг и постобработка на этом пути на вас.
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Установка

<code-tabs name="install" />

Для предсказания нужен macOS. На любой другой платформе `LibreYOLO()` отказывается
принимать `.mlpackage` и называет в сообщении текущую платформу, а матрица
поддержки отмечает эти комбинации как доступные на том основании, что для паритета
среды выполнения нужен macOS-раннер.

## Экспорт

<code-tabs name="export" />

Бандл записывается в `weights/` под именем чекпойнта, с добавлением `_fp16`, когда
`half=True`. `.mlpackage` — это директория, поэтому копировать нужно всё дерево.

Каждое семейство трассируется за обёрткой предобработки, поэтому конвертированный
граф принимает один канонический вход: RGB, `scale=1/255`, без смещения,
объявленный как `ct.ImageType`. Обёртка вбирает в себя собственное соглашение
семейства: BGR в диапазоне от 0 до 255 для YOLOX, среднее и стандартное отклонение
ImageNet для RF-DETR, тождественное преобразование для YOLO9 и RT-DETR. Именно
поэтому потребитель Core ML подаёт обычное изображение, а не тензор конкретного
семейства.

Конвертация нацелена на ML Program с минимальной целевой платформой iOS 15.
`compute_units` сохраняется в конвертированной модели, и его можно переопределить
ещё раз при загрузке артефакта.

Метаданные модели попадают в `user_defined_metadata` в виде строк — оттуда бэкенд
читает семейство, задачу, имена классов, размер входа и схему позы.

### Встроенный NMS

<code-tabs name="nms" />

`nms=True` оборачивает модель в Core ML-пайплайн, который заканчивается слоем
`NonMaximumSuppression` от Apple. У результата два выхода: `confidence` формы `N`
на число классов и `coordinates` формы `N` на 4 — нормализованные `xywh`.

Это работает только для детекции в YOLOX и YOLO9 и требует батча 1. Семейства в
стиле DETR отклоняются по имени, потому что предсказание множества берёт top-k по
запросам и классам без шага IoU и не может использовать этот слой. `max_det` здесь
тоже не выведен наружу; когда важен предел на число детекций, используйте
[встроенный NMS в ONNX](/docs/export/onnx).

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` распознаёт директорию с суффиксом `.mlpackage` и возвращает тот же
объект `Results`, что и чекпойнт. `compute_units` — единственный аргумент, который
фабрика пробрасывает для этого формата, и он принимает `all`, `cpu_and_gpu`,
`cpu_and_ne` и `cpu_only`. Аргумент `device` игнорируется, потому что Core ML
маршрутизирует вычисления через compute units.

Второй сниппет — путь через голую среду выполнения. Леттербоксинг, декодирование,
NMS и пересчёт координат становятся там вашей задачей, а имена классов лежат в
`user_defined_metadata`.

## Ограничения

Четыре семейства, только детекция: `yolox`, `yolo9`, `rtdetr` и `rfdetr`. Всё
остальное отклоняется на предварительной проверке, потому что именно обёртка
предобработки, знающая своё семейство, делает контракт фиксированного
входа-изображения корректным, а семейство вне этого списка сконвертировалось бы с
неправильной нормализацией. В ошибке как альтернативы названы ONNX и TorchScript.

Форма входа жёстко зафиксирована через `ct.ImageType`, поэтому `dynamic=True`
ничего не меняет, а метаданные записывают `dynamic=False`. Для второго разрешения
экспортируйте второй бандл.

`half=True` конвертирует с вычислительной точностью FP16. Пути к INT8 у этого
экспортёра нет.

Полную сетку семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). О более новом формате Apple для
работы на устройстве — [Core AI](/docs/export/coreai). Для одной комбинации:

<code-tabs name="support" />
