---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: 'Faster R-CNN в LibreYOLO: предсказание, валидация и экспорт'
description: >-
  Запуск Faster R-CNN в LibreYOLO для детекции объектов на четырёх бэкбонах.
  Установка, предсказание, валидация и экспорт порта из torchvision под
  лицензией BSD-3-Clause.
lead: >-
  Faster R-CNN детектирует объекты сетью предложений регионов, которая подаёт их
  в двухстадийный классификатор, — архитектура, сделавшая предложения регионов
  частью той же обучаемой сети, а не отдельным шагом. LibreYOLO портирует
  реализацию из torchvision для детекции.
keywords:
  - Faster R-CNN
  - детекция объектов python
  - region proposal network
  - двухстадийный детектор
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает загрузчик по расширению файла, поэтому
        экспортированный

        # артефакт загружается как любой чекпойнт и возвращает тот же объект
        Results.

        model = LibreYOLO("LibreFasterRCNNl.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## Установка

Faster R-CNN не требует опциональных extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена одного детектора на другой сводится к одной строке. `conf` и `iou`
задают пороги уверенности и NMS; в отличие от детектора на запросах,
Faster R-CNN сохраняет шаг NMS из оригинальной реализации. Про источники,
стриминг и обработку результатов см. [предсказание](/docs/predict).

## Варианты

Четыре размера, каждый — отдельная конфигурация torchvision, а не
масштабированная версия одной и той же: `n` — это MobileNetV3-Large со входом
320 px, `s` — тот же бэкбон при 800 px, `m` — ResNet-50 с пирамидой признаков,
а `l` — ревизия v2, с более глубокой головой предложений регионов и головой
рамок из четырёх свёрток вместо той, что у `m`. `n` и `s` меняют точность на
более лёгкий бэкбон.

## Валидация

`val()` возвращает словарь с ключами `metrics/`, куда входят точность, полнота,
mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в котором вы
обучали.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Faster R-CNN экспортируется только в ONNX и только с размером батча 1.
Экспортированный граф содержит внутри шаг изменения размера из оригинальной
реализации, поэтому LibreYOLO принудительно ставит `dynamic=True` независимо от
того, что передано, чтобы граф оставался корректным для неквадратных
источников. Экспортированный файл `.onnx` загружается обратно через
`LibreYOLO()` по расширению файла и возвращает тот же `Results`.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
