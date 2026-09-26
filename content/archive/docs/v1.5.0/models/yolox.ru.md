---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: предсказание, обучение и экспорт под Apache-2.0'
description: >-
  Использование YOLOX в LibreYOLO для детекции объектов: установка,
  предсказание, обучение, валидация и экспорт под Apache-2.0.
lead: >-
  YOLOX — одностадийный детектор без якорей (anchor-free) с разделённой головой
  классификации и регрессии, который обучается с назначением меток по SimOTA.
  LibreYOLO поддерживает его для детекции.
keywords:
  - YOLOX
  - детекция объектов python
  - yolox обучение на своём датасете
  - детектор без якорей anchor-free
  - разделённая голова
  - SimOTA
  - детекция объектов в реальном времени
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Валидация на COCO
      language: bash
      code: |
        # Встроенный yaml для COCO содержит скрипт скачивания, поэтому нужно
        # явное разрешение, если датасет ещё не лежит локально.
        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreYOLOXs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Установка

YOLOX не требует никаких extra сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращают все семейства, поэтому
замена на другой детектор — правка в одну строку. `conf` задаёт порог
уверенности, а `iou` — порог NMS, который применяется к трём разделённым
масштабам предсказания. Про источники, стриминг и обработку результатов — в
разделе [предсказание](/docs/predict).

## Варианты

Шесть размеров используют общий бэкбон CSP и neck PAFPN. Два самых маленьких,
`n` и `t`, работают на меньшем фиксированном разрешении входа, чем остальные
четыре; точное значение для каждого приведено в таблице бенчмарков ниже.

<benchmark-table task="detect" />

<va-embed />

## Обучение

<code-tabs name="train" />

Если ничего не менять, обучение идёт 300 эпох с `lr0=0.01`, SGD с моментумом
0.9, прогревом в 5 эпох и отключёнными аугментациями mosaic и mixup на
последних 15 эпохах. `train()` также принимает аргумент `pretrained`, но внутри
метода это значение никогда не читается: обучение всегда продолжается с тех
весов, с которыми была создана модель, поэтому `pretrained=False` не
переинициализирует сеть.

По умолчанию `imgsz` берёт фиксированное значение из базового конфига обучения,
а не родное разрешение загруженного чекпойнта. Это касается в
первую очередь чекпойнтов `n` и `t`: если продолжить обучение любого из них, не
задав `imgsz` явно, разрешение поднимется до большего значения по умолчанию
вместо меньшего, с которым чекпойнт был опубликован.

Про датасеты, аугментацию, обучение на нескольких GPU и логгеры — в разделе
[обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность, полноту,
mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в котором вы
обучали модель.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запуск графа в голой среде выполнения, без установленного
LibreYOLO, тоже поддерживается, но тогда предобработку и постобработку придётся
писать самостоятельно. Экспорт в CoreML умеет вшивать NMS в граф с `nms=True`;
YOLOX и YOLOv9 — единственные два семейства, для которых этот флаг сейчас
принимается.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
