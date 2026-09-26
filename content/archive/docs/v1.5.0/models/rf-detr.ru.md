---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: обучение, дообучение и экспорт под MIT'
description: >-
  Использование RF-DETR в LibreYOLO для детекции, сегментации экземпляров,
  оценки позы и повёрнутых рамок. Установка, предсказание, обучение, валидация и
  экспорт — всё под лицензией MIT.
lead: >-
  Трансформер для детекции, который предсказывает фиксированный набор объектов
  вместо плотной сетки, поэтому на инференсе ему не нужен NMS. LibreYOLO
  поддерживает его для четырёх задач.
keywords:
  - RF-DETR
  - DETR
  - трансформер для детекции объектов
  - детекция объектов в реальном времени
  - сегментация экземпляров python
  - оценка позы python
  - повёрнутые рамки obb
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: 'LibreRFDETRs, детекция на видео при 512 px.'
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Видео
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # Любой источник, который принимает библиотека: файл, папка, URL,
        # индекс веб-камеры, RTSP-поток или список .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() возвращает обычный dict, а не объект
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Валидация на COCO
      language: bash
      code: |
        # Во встроенный yaml для COCO зашит скрипт скачивания, поэтому нужно
        # явное разрешение, если датасет ещё не лежит локально.
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)

        # Аргументы, которые принимает любой формат:
        #
        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"
        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"
        #             | "tflite" | "coreml" | "coreai".
        #             "engine" — псевдоним tensorrt, "litert" — tflite.
        #   imgsz     int или (высота, ширина). По умолчанию — родное
        #             разрешение чекпойнта.
        #   batch     int, по умолчанию 1.
        #   half      bool, экспорт в FP16. По умолчанию False.
        #   int8      bool, экспорт в INT8. По умолчанию False. Нужен `data`.
        #   data      путь к YAML датасета, по нему калибруется int8.
        #   fraction  float, доля этого калибровочного набора. По умолчанию 1.0.
        #   dynamic   bool, динамические оси. По умолчанию True.
        #   simplify  bool, упрощение графа ONNX. По умолчанию True.
        #   opset     int, opset ONNX. Если не задан, выбирается по семейству.
        #   device    str, устройство для трассировки. По умолчанию — устройство
        #             модели.
        #   output_path  str, по умолчанию имя, производное от чекпойнта.
        #   verbose   bool, по умолчанию False.
        #   allow_download_scripts  bool, по умолчанию False. Разрешает
        #             встроенный Python в YAML датасета, который нужно скачать.
        #
        # Некоторые форматы принимают собственные дополнительные аргументы,
        # например целевую платформу RKNN. Они описаны на странице каждого
        # формата.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreRFDETRs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
    - label: Без LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Запуск графа напрямую означает, что предобработку и постобработку

        # придётся писать самому. Прежде чем что-то подключать, посмотрите

        # на сигнатуру.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Установка

RF-DETR требует собственного extra, который подтягивает `transformers` для бэкбона.

```bash
pip install "libreyolo[rfdetr]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена на другой детектор занимает одну строку. `conf` и `max_det` фильтруют
отбор запросов; шага NMS, который надо было бы настраивать, здесь нет. Про
источники, стриминг и обработку результатов см. [предсказание](/docs/predict).

## Варианты

Четыре размера и четыре задачи на одной архитектуре: сегментация, оценка позы и
повёрнутые рамки переиспользуют декодер детекции с другой головой, поэтому
принимают те же аргументы. По числу параметров размеры близки и различаются в
основном входным разрешением.

<benchmark-table task="detect" />

<va-embed />

## Обучение

Обучение начинается с опубликованного чекпойнта — для всех четырёх задач.
RF-DETR числит `pretrained` среди аргументов, которые его собственный обучающий
код игнорирует, поэтому передача `pretrained=False` не даст здесь модель со
случайной инициализацией.

<code-tabs name="train" />

Здесь два аргумента важнее, чем в CNN-детекторе. Держите `lr0` на `1e-4` или
ниже: трансформерные детекторы расходятся при той скорости обучения, которую
модель YOLO переносит спокойно. Оставьте `imgsz` на родном разрешении
чекпойнта, если нет причины его менять. Вход должен нацело делиться на размер
патча бэкбона, умноженный на число окон; LibreYOLO проверяет это до старта
запуска и называет ближайшие допустимые размеры.

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность, полноту,
mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, на котором вы
обучались.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запуск графа в голой среде выполнения, без установленного
LibreYOLO, тоже поддерживается, но тогда предобработку и постобработку придётся
писать самостоятельно.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
