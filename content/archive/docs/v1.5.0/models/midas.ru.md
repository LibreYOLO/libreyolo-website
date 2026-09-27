---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: монокулярная оценка глубины в LibreYOLO'
description: >-
  Используйте MiDaS в LibreYOLO для монокулярной оценки глубины. Установка,
  предсказание, валидация и экспорт двух вариантов под лицензией MIT, которые
  скачиваются из isl-org.
lead: >-
  MiDaS — монокулярная оценка относительной глубины, обученная на смеси
  датасетов с функцией потерь, инвариантной к масштабу и сдвигу; именно эта
  линия работ задала протокол zero-shot переноса глубины, который переиспользуют
  более поздние семейства. В LibreYOLO он поддержан для задачи depth:
  предсказание и zero-shot валидация, без пути обучения.
keywords:
  - MiDaS
  - монокулярная оценка глубины
  - оценка глубины по одному изображению python
  - карта глубины из фото
  - DPT
  - относительная глубина
  - zero-shot глубина
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Файла ещё нет на диске: LibreYOLO скачивает его из официального релиза

        # isl-org/MiDaS на GitHub и перед использованием сверяет с
        зафиксированным SHA-256.

        model = LibreYOLO("LibreMiDaSl-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)


        depth = result.depth_map

        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Вариант Small
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Энкодер EfficientNet-Lite3, меньше и быстрее, чем размер l на
        DPT-Large.

        model = LibreYOLO("LibreMiDaSs-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный

        # артефакт загружается как любой чекпойнт и возвращает тот же объект
        Results.

        model = LibreYOLO("LibreMiDaSl-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: ce2fbf3ae43e9be4
---

## Установка

MiDaS не требует установки дополнительных extra-пакетов. Всё, что он импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

MiDaS — единственное семейство для глубины, которое LibreYOLO не перепубликует в
своей организации на Hugging Face. Запрос чекпойнта по его имени файла в
LibreYOLO скачивает соответствующий официальный файл напрямую из релизов
`isl-org/MiDaS` на GitHub, сверяет его с зафиксированным SHA-256 и перед первым
использованием оборачивает метаданными чекпойнта LibreYOLO; при последующих
запусках берётся закэшированный локальный файл. Почему так — см. раздел
«Лицензирование».

<code-tabs name="predict" />

`result.depth_map` содержит плотную карту относительной обратной глубины:
большие значения означают, что точка ближе к камере, а у самих значений нет ни
метрических единиц, ни общего масштаба между изображениями. `save=True`
записывает на диск раскрашенную визуализацию этой карты; `Results.plot()` это
семейство не покрывает, поскольку определён только для карт нормалей и краёв.
Об источниках, стриминге и обработке результатов — в разделе
[предсказание](/docs/predict).

## Варианты

Два варианта с разными энкодерами, а не просто разные масштабы одного и того же.
`s` — это MiDaS v2.1 Small, энкодер EfficientNet-Lite3. `l` — это DPT-Large,
энкодер ViT-L/16 с декодером DPT, который MiDaS предложил для плотного
предсказания. Различается и предобработка: `s` масштабирует изображение с
сохранением пропорций по верхней границе (upper bound) и нормализует по среднему
и стандартному отклонению ImageNet, а `l` использует минимальное масштабирование
с сохранением пропорций (minimal), со средним и стандартным отклонением 0.5.
Берите `s`, если нужна более лёгкая CNN, и `l` — ради точности трансформерного
декодера.

Обучение для этого семейства не предусмотрено. `LibreMiDaS.train()` безусловно
вызывает `NotImplementedError`.

## Валидация

`val()` запускает общий валидатор глубины: он выравнивает каждое предсказание с
эталонной разметкой (ground truth), подбирая по методу наименьших квадратов
масштаб и сдвиг для каждого изображения, а затем выдаёт стандартные метрики
относительной глубины в режиме zero-shot — AbsRel, RMSE и три пороговых значения
delta.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`, только с `depth_map` вместо рамок.

<code-tabs name="export" />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
