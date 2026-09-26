---
title: FCN
families:
  - fcn
seo_title: 'FCN: предсказание и экспорт ResNet FCN под лицензией BSD-3-Clause'
description: >-
  Используйте FCN в LibreYOLO для семантической сегментации. Установка,
  предсказание, валидация и экспорт чекпойнтов FCN на ResNet с расширенными
  свёртками из torchvision.
lead: >-
  Плотный попиксельный классификатор, который заменяет полносвязные слои
  детектора свёртками, поэтому выдаёт карту классов в полном разрешении, а не
  рамки. LibreYOLO поставляет его только для семантической сегментации.
keywords:
  - FCN
  - fully convolutional network
  - семантическая сегментация python
  - попиксельная сегментация изображений
  - fcn onnx
  - ResNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreFCNr50.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) id классов

        print(mask.classes)      # отсортированные id классов, присутствующих на
        изображении
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreFCNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Установка

FCN не требует опциональных extra. Всё, что он импортирует, входит в базовую
установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Семантическая сегментация возвращает по одному id класса на пиксель, а не
рамки, поэтому `result.semantic_mask` хранит массив `(H, W)` в `.data` и список
id классов, присутствующих на изображении, в `.classes`. `conf`, `iou` и
`max_det` принимаются ради совместимости API, но ни на что не влияют: модель
назначает класс каждому пикселю по argmax, без порога уверенности и без шага
NMS. Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

## Варианты

Две глубины ResNet, обе с фиксированным входом 520 px. Граф инференса в
библиотеке — это FCN на ResNet с расширенными свёртками из torchvision, а не
исходная сеть FCN-8s из статьи, построенная на VGG и со skip-связями.

LibreYOLO не обучает FCN: `train()` вызывает `NotImplementedError` для этого
семейства, и [уровень поддержки](/docs/models) выше помечает его как только
инференс. Два опубликованных чекпойнта — это собственные веса torchvision,
обученные на COCO и конвертированные для загрузчика LibreYOLO.

## Валидация

`val()` возвращает `metrics/mIoU` и `metrics/pixel_accuracy`, измеренные на
любом датасете в том формате, в котором вы обучали.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. [Экспорт](/docs/export) перечисляет аргументы, которые
принимает каждый формат.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>
