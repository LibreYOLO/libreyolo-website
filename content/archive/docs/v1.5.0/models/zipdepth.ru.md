---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: лёгкая монокулярная глубина в LibreYOLO'
description: >-
  Используйте ZipDepth в LibreYOLO для лёгкой монокулярной оценки глубины.
  Установка, предсказание, валидация и экспорт двух чекпойнтов под лицензией
  MIT.
lead: >-
  ZipDepth — компактная репараметризуемая CNN, дистиллированная из Depth
  Anything V2 Large, которая предсказывает плотную карту относительной обратной
  глубины. LibreYOLO поддерживает его в задаче depth: предсказание и zero-shot
  валидация, обучение не предусмотрено.
keywords:
  - ZipDepth
  - монокулярная оценка глубины
  - лёгкая модель глубины для edge
  - карта глубины python
  - относительная глубина
  - репараметризуемая CNN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Чекпойнт для NPU/edge
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Тот же энкодер, но голова апсемплинга без unfold — для компиляторов
        # без поддержки gather/unfold. Выход визуально совпадает с чекпойнтом b.
        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreZipDepthb-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Установка

ZipDepth не требует установки дополнительных extra-пакетов. Всё, что он
импортирует, входит в базовую установку.

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
семейство не покрывает, поскольку определён только для нормалей поверхности и
краёв. Об источниках, стриминге и обработке результатов — в разделе
[предсказание](/docs/predict).

## Варианты

Два чекпойнта с одинаковой ёмкостью энкодера различаются только обученной
головой апсемплинга. `b` использует выпуклый апсемплинг и работает на GPU или
CPU. В `bnpu` вместо неё стоит декодер без unfold — для NPU и edge-компиляторов,
где нет поддержки gather/unfold; его выход задокументирован как визуально
эквивалентный `b`. Берите `bnpu`, когда целевая среда выполнения ограничена, и
`b` во всех остальных случаях.

Оба чекпойнта дистиллированы из псевдоразметки Depth Anything V2 Large, так что
это семейство — компактный, ориентированный на edge уровень задачи depth в
LibreYOLO, наряду с более крупными энкодерами Depth Anything V2.

Обучение для этого семейства не предусмотрено. `LibreZipDepth.train()`
всегда выбрасывает `NotImplementedError`: upstream-рецепт дистиллирует
псевдоразметку по большому набору изображений, который невозможно воспроизвести
как запуск обучения в LibreYOLO. Обучайте в upstream-проекте
[fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth) и конвертируйте
результат скриптом `weights/convert_zipdepth_weights.py`.

## Валидация

`val()` запускает общий валидатор глубины: он выравнивает каждое предсказание с
эталонной разметкой (ground truth), подбирая по методу наименьших квадратов
масштаб и сдвиг для каждого изображения, а затем выдаёт стандартные метрики
относительной глубины в режиме zero-shot — AbsRel, RMSE и три пороговых значения
delta.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспорт работает по контракту плотного выхода с фиксированным разрешением:
исходное изображение растягивается до размеров экспортированного холста, а полученная
карта глубины после этого возвращается к исходному размеру. Экспортированный
артефакт загружается обратно через `LibreYOLO()` по суффиксу файла, поэтому файл
`.onnx` или `.ncnn` ведёт себя как чекпойнт и возвращает тот же `Results`,
только с `depth_map` вместо рамок.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
