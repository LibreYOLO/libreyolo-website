---
title: Depth Anything V2
families:
  - depth_anything
seo_title: 'Depth Anything V2: предсказание и валидация монокулярной глубины'
description: >-
  Используйте Depth Anything V2 в LibreYOLO для монокулярной оценки глубины.
  Установка, предсказание и валидация; Small выходит под Apache-2.0, Base и
  Large — под CC-BY-NC-4.0.
lead: >-
  Depth Anything V2 — энкодер DINOv2 в паре с декодером DPT, который по одному
  изображению предсказывает плотную карту относительной обратной глубины. В
  LibreYOLO он поддержан для задачи depth: предсказание и zero-shot валидация,
  без пути обучения.
keywords:
  - Depth Anything V2
  - оценка глубины по одному изображению
  - монокулярная оценка глубины python
  - карта глубины из фото
  - DPT
  - DINOv2
  - относительная глубина
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Чтение карты глубины
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: плотная карта (H, W), больше
        значение — ближе

        raw = depth.data                # тензор, без метрических единиц и
        общего масштаба между изображениями

        normalized = depth.normalized() # приведено к [0, 1] для визуализации
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## Установка

Depth Anything V2 не требует установки дополнительных extra-пакетов. Всё, что он импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`result.depth_map` содержит плотную карту относительной обратной глубины:
большие значения означают, что точка ближе к камере, а у самих значений нет ни
метрических единиц, ни общего масштаба между изображениями. `save=True`
записывает на диск раскрашенную визуализацию этой карты; `Results.plot()` это
семейство не покрывает, поскольку определён только для карт нормалей и краёв.
Разрешение входа должно нацело делиться на 14 — это шаг сетки патчей DINOv2, на
которой строится голова DPT; LibreYOLO проверяет это перед запуском и вызывает
ошибку, если условие не выполнено. Об источниках, стриминге и обработке
результатов — в разделе [предсказание](/docs/predict).

## Варианты

Четыре размера энкодера, s/b/l/g, соответствующие ViT-S/B/L/G. В таблице
чекпойнтов ниже перечислены только s, b и l; чекпойнт Giant не опубликован. У
всех четырёх одинаковое входное разрешение, поэтому выбор размера меняет ёмкость
энкодера, а не размер изображения. Ещё один фактор — лицензирование: чекпойнт
Small выходит под Apache-2.0, а Base и Large — под CC-BY-NC-4.0, см. раздел
«Лицензирование» ниже.

Обучение и дообучение для этого семейства не предусмотрены.
`LibreDepthAnythingV2.train()` безусловно вызывает `NotImplementedError`; вместо
этого сконвертируйте совместимый чекпойнт из upstream-проекта скриптом
`weights/convert_depth_anything_v2_weights.py`.

## Валидация

`val()` запускает общий валидатор глубины: он выравнивает каждое предсказание с
эталонной разметкой (ground truth), подбирая по методу наименьших квадратов
масштаб и сдвиг для каждого изображения, а затем выдаёт стандартные метрики
относительной глубины в режиме zero-shot — AbsRel, RMSE и три пороговых
значения delta.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`, только с `depth_map` вместо рамок. В разделе
[экспорт](/docs/export) перечислены аргументы, которые принимает каждый формат.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
