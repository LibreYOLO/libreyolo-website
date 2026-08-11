---
title: FCOS
families:
  - fcos
seo_title: 'FCOS в LibreYOLO: предсказание, валидация и экспорт'
description: >-
  Запуск FCOS в LibreYOLO для детекции объектов без якорей. Установка,
  предсказание, валидация и экспорт порта из torchvision под BSD-3-Clause,
  ResNet-50/FPN.
lead: >-
  FCOS детектирует объекты попиксельно, а не опирается на набор заранее заданных
  якорных рамок: он предсказывает рамку и оценку центральности (centerness) в
  каждой позиции карты признаков. LibreYOLO переносит реализацию из torchvision
  для детекции.
keywords:
  - FCOS
  - детекция объектов python
  - детектор без якорей anchor-free
  - одностадийный детектор
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный
        артефакт

        # загружается как любой чекпойнт и возвращает тот же объект Results.

        model = LibreYOLO("LibreFCOSr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## Установка

FCOS не нужны опциональные extra. Всё, что он импортирует, входит в базовую
установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращают все семейства, поэтому
замена на другой детектор — правка в одну строку. Если вызвать модель без
аргументов с порогами, применяются собственные опубликованные значения по
умолчанию для FCOS: `conf=0.2`, `iou=0.6` и `max_det=100`; передайте любой из
трёх, чтобы их переопределить. FCOS сохраняет финальный шаг NMS поверх своих
попиксельных предсказаний. Про источники, стриминг и обработку результатов — в
разделе [предсказание](/docs/predict).

## Варианты

Один размер: ResNet-50 с пирамидой признаков — единственный вариант, который
распознаёт это семейство.

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность,
полноту, mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в
котором вы обучали модель.

<code-tabs name="val" />

## Экспорт

<export-matrix />

FCOS экспортируется в ONNX, TorchScript и OpenVINO. Перед запуском графа FCOS
сохраняет исходное соотношение сторон, поэтому для путей ONNX и OpenVINO
LibreYOLO принудительно ставит `dynamic=True` независимо от того, что было
передано, — чтобы граф оставался корректным для входных форм с паддингом.
Экспортированный файл `.onnx` загружается обратно через `LibreYOLO()` по
суффиксу файла и возвращает тот же `Results`.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
