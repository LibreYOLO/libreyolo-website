---
title: EoMT
families:
  - eomt
seo_title: 'EoMT: семантическая, паноптическая сегментация и сегментация экземпляров'
description: >-
  Используйте EoMT в LibreYOLO для семантической, паноптической сегментации и
  сегментации экземпляров на обычном vision-трансформере DINOv2, без отдельного
  декодера. Лицензия MIT.
lead: >-
  Сеть сегментации на обычном vision-трансформере без выделенного пиксельного
  декодера: маски предсказывают дополнительные обучаемые запросы (queries),
  добавленные прямо в энкодер. В LibreYOLO она поддержана для семантической,
  паноптической сегментации и сегментации экземпляров.
keywords:
  - EoMT
  - encoder-only mask transformer
  - DINOv2
  - паноптическая сегментация python
  - сегментация экземпляров
  - семантическая сегментация изображений
  - сегментация на трансформере без декодера
last_verified: 1.5.0
snippets:
  predict:
    - label: Семантическая сегментация
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreEoMTl-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) идентификаторы классов

        print(mask.classes)      # отсортированные id классов, найденных на
        изображении
    - label: Сегментация экземпляров
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суффикс -seg в имени файла выбирает задачу сегментации экземпляров,
        # поэтому аргумент task здесь не нужен.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Паноптическая сегментация
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) идентификаторы сегментов
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Семантическая сегментация
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Сегментация экземпляров
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # маски
        print(metrics["metrics/mAP50-95(B)"])   # рамки
    - label: Паноптическая сегментация
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## Установка

EoMT не требует установки дополнительных extra-пакетов. Всё, что он
импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.
Задачу выбирает суффикс в имени файла (`-sem`, `-seg`, `-panoptic`), а
`LibreYOLO()` определяет её по этому имени, поэтому аргумент `task=` не нужен.

<code-tabs name="predict" />

Семантическая сегментация заполняет `result.semantic_mask` — массив `(H, W)` с
идентификаторами классов в `.data`. Сегментация экземпляров заполняет
`result.boxes` и `result.masks` — ровно в той же форме, что возвращают
остальные семейства сегментации. Паноптическая сегментация заполняет
`result.panoptic`: карту идентификаторов сегментов `(H, W)` в `.data` и
`.segments_info` — список словарей `{"id", "category_id"}`, по одному на
сегмент. `conf` фильтрует отбор запросов, а `iou` на семантическую задачу не
влияет, потому что она берёт argmax по каждому пикселю и шага NMS в ней нет. Об
источниках, стриминге и обработке результатов — в разделе
[предсказание](/docs/predict).

## Варианты

Три размера энкодера — s/b/l, все на DINOv2. Семантический чекпойнт обучен на
ADE20K при 512 px; чекпойнты для экземпляров и паноптики обучены на COCO при
640 px, плюс второй чекпойнт для экземпляров обучен при 1280 px. В исходном
проекте веса на DINOv2 для сегментации экземпляров есть только в размере l; s и
b опубликованы только для семантической и паноптической задач. Варианты EoMT на
DINOv3 в исходном проекте есть, но здесь не поставляются, потому что они
зависят от весов DINOv3 с ограниченным доступом и некоммерческой лицензией.

LibreYOLO не обучает EoMT: `train()` для этого семейства вызывает
`NotImplementedError`, и [уровень поддержки](/docs/models) выше отмечает это
семейство как «только инференс».

## Валидация

`val()` выбирает ветку по задаче. Семантическая возвращает `metrics/mIoU` и
`metrics/pixel_accuracy`. Сегментация экземпляров возвращает те же ключи mAP по
маскам и рамкам, что и другие семейства сегментации. Паноптическая возвращает
Panoptic Quality как `metrics/PQ`, разложенную на `metrics/SQ` (качество
сегментации) и `metrics/RQ` (качество распознавания), а также
`metrics/PQ_things` и `metrics/PQ_stuff`.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Сейчас экспортируется только семантическая задача: сегментация экземпляров и
паноптическая сегментация на вызове `export()` получают `NotImplementedError`,
потому что для их вывода из масок-запросов пока нет контракта экспорта в среды
выполнения. Экспортированный семантический артефакт загружается обратно через
`LibreYOLO()` по суффиксу файла, поэтому файл `.onnx` или `.engine` ведёт себя
как чекпойнт и возвращает тот же `Results`.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
