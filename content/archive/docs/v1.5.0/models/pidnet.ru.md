---
title: PIDNet
families:
  - pidnet
seo_title: 'PIDNet: предсказание и экспорт сегментации в реальном времени под MIT'
description: >-
  Используйте PIDNet в LibreYOLO для семантической сегментации в реальном
  времени. Установка, предсказание, валидация и экспорт чекпойнтов s/m/l на
  Cityscapes под MIT.
lead: >-
  Трёхветвевая сеть семантической сегментации, которая добавляет отдельную ветвь
  границ к архитектуре, вдохновлённой
  пропорционально-интегрально-дифференциальным регулятором, и нацелена на
  инференс в реальном времени. LibreYOLO поставляет её только для семантической
  сегментации.
keywords:
  - PIDNet
  - семантическая сегментация в реальном времени
  - сегментация с учётом границ
  - Cityscapes
  - pidnet onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePIDNets-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) id классов

        print(mask.classes)      # отсортированные id классов, присутствующих на
        изображении
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibrePIDNets-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## Установка

PIDNet не требует опциональных extra. Всё, что он импортирует, входит в
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

Три размера, все с фиксированным входом 1024 px. Опубликованные чекпойнты —
это конвертации официальных весов PIDNet на Cityscapes, 19 классов.

LibreYOLO не обучает PIDNet: `train()` вызывает `NotImplementedError` для
этого семейства, и [уровень поддержки](/docs/models) выше помечает его как
только инференс.

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

## Цитирование

<citation-block />
