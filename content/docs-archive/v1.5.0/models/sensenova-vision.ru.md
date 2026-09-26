---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'SenseNova-Vision в LibreYOLO: 7 задач, один чекпойнт'
description: >-
  Используйте SenseNova-Vision в LibreYOLO для детекции, сегментации,
  паноптической сегментации, оценки позы, точек, глубины и OCR из одного
  генеративного чекпойнта, управляемого промптом.
lead: >-
  SenseNova-Vision — унифицированная мультимодальная модель, которая сводит
  задачи компьютерного зрения к генерации по промпту на общем декодере: рамки,
  точки, ключевые точки и слова OCR выходят как размеченный тегами текст, а
  карты глубины, маски и паноптические карты выходят как изображения, которые
  отрисовывает декодер. LibreYOLO загружает её через LibreVLM и поддерживает
  семь задач из одного чекпойнта на 7B.
keywords:
  - SenseNova-Vision
  - SenseTime
  - унифицированная мультимодальная модель
  - Bagel
  - детекция по текстовому промпту
  - плотные предсказания
  - сегментация по описанию
  - паноптическая сегментация python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() переключает задачи на той же загруженной модели.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Сегментация по описанию и паноптическая сегментация
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # Сегментация здесь по описанию: нужна фраза с описанием цели, а не
        список классов.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Без своего словаря паноптическая сегментация откатывается к

        # паноптическим категориям COCO, на которых настраивали чекпойнт.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Точки, поза и OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Если словарь не задан, поза откатывается к "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Установка

SenseNova-Vision нужен свой extra: он подтягивает `accelerate` для диспетчеризации большой модели, без которой этот чекпойнт не запустить, и, на всех платформах, кроме macOS, `bitsandbytes` для загрузки в 4 бита.

```bash
pip install "libreyolo[sensenova]"
```

Чекпойнт зеркалируется на Hugging Face в собственной организации LibreYOLO и скачивается автоматически при первом использовании; он под CC BY-NC 4.0, только некоммерческое использование, и загрузчик выводит это уведомление перед каждой автоматической загрузкой. См. «Лицензирование» ниже.

## Предсказание

<code-tabs name="predict" />

Каждое предсказание — это диффузионное декодирование поверх общего бэкбона Bagel-MoT, поэтому это модель возможностей, а не модель реального времени: задержка на изображение заметно выше, чем у специализированного детектора или сегментатора. `dtype="auto"` (по умолчанию) загружает bf16 на GPU с достаточным объёмом памяти, а в остальных случаях откатывается к 4-битной квантизации NF4, для которой нужен `bitsandbytes`; передайте `dtype="bf16"`, чтобы принудительно включить полную точность на достаточно большой GPU. `noise_seed=42` при создании модели фиксирует зерно диффузионного сэмплера для воспроизводимых плотных выходов; передайте `noise_seed=None`, чтобы отключить фиксацию.

Семь задач делят один загруженный чекпойнт: `set_task()` переключает между ними без перезагрузки. `set_classes()` задаёт активный словарь; детекция, точки, поза и паноптическая сегментация принимают список классов, а сегментация работает по описанию (referring): ей нужна именно та фраза, которая описывает выделяемый объект. Каждая задача возвращает стандартный объект `Results`, в котором заполнено своё поле: `boxes` для detect, `points` для point, `boxes` и `keypoints` для pose, `ocr` для OCR, `depth_map` для depth, `masks` для segment и `panoptic` (вместе с `segments_info`) для panoptic. Про источники, стриминг и обработку результатов см. [предсказание](/docs/predict).

## Чекпойнты

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
