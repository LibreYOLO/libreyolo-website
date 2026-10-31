---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: предсказание и экспорт семантической сегментации на ASPP'
description: >-
  Используйте DeepLabv3 в LibreYOLO для семантической сегментации. Установка,
  предсказание, валидация и экспорт чекпойнтов torchvision на ResNet и
  MobileNetV3.
lead: >-
  Сеть семантической сегментации, которая параллельно агрегирует признаки на
  нескольких коэффициентах расширения (atrous spatial pyramid pooling), прежде
  чем классифицировать каждый пиксель. LibreYOLO поставляет её только для
  семантической сегментации.
keywords:
  - DeepLabv3
  - ASPP
  - atrous spatial pyramid pooling
  - семантическая сегментация python
  - сегментация изображений по пикселям
  - deeplabv3 onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) id классов

        print(mask.classes)      # отсортированные id классов, присутствующих на
        изображении
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---

## Установка

DeepLabv3 не требует опциональных extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.
Суффикс `-sem` в имени файла обязателен для этого семейства.

<code-tabs name="predict" />

Семантическая сегментация возвращает по одному id класса на пиксель, а не
рамки, поэтому `result.semantic_mask` хранит массив `(H, W)` в `.data` и список
id классов, присутствующих на изображении, в `.classes`. `conf`, `iou` и
`max_det` принимаются ради совместимости API, но ни на что не влияют: модель
назначает класс каждому пикселю по argmax, без порога уверенности и без шага
NMS. Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

## Варианты

Три бэкбона с расширенными свёртками: ResNet-50, ResNet-101 и
MobileNetV3-Large. Это DeepLabv3, а не DeepLabv3+, поэтому здесь нет ни стадии
декодера, ни уточнения через CRF — так устроена реализация torchvision, а не
референсный код самой статьи.

LibreYOLO не обучает DeepLabv3: `train()` вызывает `NotImplementedError` для
этого семейства, и [уровень поддержки](/docs/models) выше помечает его как
только инференс. Три опубликованных чекпойнта — это собственные веса
torchvision, обученные на COCO с метками VOC и конвертированные для загрузчика
LibreYOLO.

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
