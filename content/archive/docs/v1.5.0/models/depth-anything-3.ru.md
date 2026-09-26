---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: 'Depth Anything 3: монокулярная оценка глубины в LibreYOLO'
description: >-
  Использование Depth Anything 3 в LibreYOLO для монокулярной оценки глубины.
  Установка, предсказание, валидация и экспорт чекпойнта DA3MONO-LARGE, лицензия
  Apache-2.0.
lead: >-
  Depth Anything 3 — обычный трансформер DINOv2, обученный предсказывать глубину
  и геометрию камеры по одному или нескольким видам без какой-либо архитектурной
  специализации. В LibreYOLO портирован его чекпойнт DA3MONO-LARGE для задачи
  глубины: предсказание и zero-shot валидация, без пути обучения.
keywords:
  - Depth Anything 3
  - DA3
  - монокулярная оценка глубины
  - DINOv2
  - карта глубины по фото python
  - относительная глубина нейросеть
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Чтение карты глубины
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnything3l-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: плотная (H, W), больше = ближе

        raw = depth.data                # тензор, без метрических единиц и
        общего масштаба между изображениями

        normalized = depth.normalized() # приведена к [0, 1] для визуализации
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный
        артефакт

        # загружается как любой чекпойнт и возвращает тот же объект Results.

        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---

## Установка

Depth Anything 3 не нужны опциональные extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`result.depth_map` содержит плотную карту относительной обратной глубины:
большие значения означают, что точка ближе к камере, а сами значения не имеют
ни метрических единиц, ни общего масштаба между изображениями. Оригинальный
чекпойнт выдаёт положительную относительную глубину; сетевая обёртка LibreYOLO
инвертирует её и воспроизводит официальную обработку неба, чтобы выход
следовал общему контракту глубины в LibreYOLO. `save=True` сохраняет на диск
визуализацию этой карты в цветовой схеме; `Results.plot()` это семейство не
покрывает, потому что определён только для нормалей к поверхности и границ.
Про источники, стриминг и обработку результатов — в разделе
[предсказание](/docs/predict).

## Варианты

Один размер, `l`, с фиксированным разрешением входа. Оригинальный проект DA3
публикует ещё any-view чекпойнты Small и Base, чекпойнт метрической глубины, а
также чекпойнты Nested и Giant; LibreYOLO не отдаёт ни один из них. Метрической
глубине нужен другой публичный контракт, не тот, что у задачи относительной
обратной глубины в LibreYOLO, а any-view и Nested чекпойнтам нужен API камеры
для нескольких изображений, которого в LibreYOLO нет. У any-view чекпойнтов
Large и Giant к тому же лицензия CC-BY-NC-4.0, и ни один путь скачивания в
LibreYOLO на них не ссылается.

Обучение для этого семейства не предусмотрено. `LibreDepthAnything3.train()`
безусловно выбрасывает `NotImplementedError`; обучайте в оригинальном проекте и
конвертируйте совместимый чекпойнт DA3MONO-LARGE скриптом
`weights/convert_depth_anything3_weights.py`.

## Валидация

`val()` запускает общий валидатор глубины: он выравнивает каждое предсказание с
эталонной разметкой (ground truth) по масштабу и сдвигу, посчитанным методом
наименьших квадратов для каждого изображения, а затем выдаёт стандартные метрики
zero-shot относительной глубины — AbsRel, RMSE и три delta-порога.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Для этого семейства экспорт ограничен пятью форматами: ONNX, TorchScript,
ExecuTorch, TensorRT и OpenVINO. Запрос любого другого формата выбрасывает
`NotImplementedError`, а не пытается сделать непроверенную конверсию.
Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`, только с `depth_map` вместо рамок.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
