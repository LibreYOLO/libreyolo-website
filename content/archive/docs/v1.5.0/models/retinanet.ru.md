---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet в LibreYOLO: предсказание, валидация и экспорт'
description: >-
  Запуск RetinaNet в LibreYOLO для одностадийной детекции объектов с функцией
  потерь focal loss. Установка, предсказание, валидация и экспорт порта
  torchvision под BSD-3-Clause.
lead: >-
  RetinaNet — одностадийный детектор, который обучается с функцией потерь focal
  loss: она снижает вклад простых отрицательных примеров, поэтому плотной сетке
  якорей больше не нужен отдельный этап генерации предложений, чтобы оставаться
  точной. LibreYOLO переносит реализацию из torchvision для детекции.
keywords:
  - RetinaNet
  - focal loss
  - детекция объектов python
  - одностадийный детектор без предложений
  - retinanet экспорт в onnx
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный
        артефакт

        # загружается как любой чекпойнт и возвращает тот же объект Results.

        model = LibreYOLO("LibreRetinaNetr50v2.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Установка

RetinaNet не нужны опциональные extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращают все семейства, поэтому
замена на другой детектор — правка в одну строку. `conf` и `iou` задают пороги
уверенности и NMS; RetinaNet сохраняет шаг NMS из оригинальной реализации по
плотной сетке якорей. Про источники, стриминг и обработку результатов — в
разделе [предсказание](/docs/predict).

## Варианты

Два размера, оба на ResNet-50 с пирамидой признаков: `r50` — оригинальная
голова, а `r50v2` заменяет её головой с GroupNorm и более широким блоком P6,
который питается с последней стадии бэкбона, а не с выхода FPN.

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность,
полноту, mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в
котором вы обучали модель.

<code-tabs name="val" />

## Экспорт

<export-matrix />

RetinaNet экспортируется только в ONNX и только с размером батча 1. RetinaNet
масштабирует вход к переменному размеру с сохранением пропорций, поэтому
LibreYOLO принудительно ставит `dynamic=True` независимо от переданного
значения — так граф остаётся корректным для источников разной формы.
Экспортированный файл `.onnx` загружается обратно через `LibreYOLO()` по
суффиксу файла и возвращает тот же `Results`.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>
