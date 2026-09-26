---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: детекция объектов в LibreYOLO'
description: >-
  Запуск EfficientDet D0-D4 в LibreYOLO: детекторы на BiFPN для предсказания,
  валидации и экспорта в ONNX, TensorRT и OpenVINO под лицензией Apache-2.0.
lead: >-
  EfficientDet соединяет бэкбон EfficientNet с повторяющейся двунаправленной
  пирамидой признаков (BiFPN) и масштабирует глубину, ширину и разрешение
  одновременно — в пяти размерах. В LibreYOLO он поставляется как детектор
  только для инференса.
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - детекция объектов python
  - составное масштабирование compound scaling
  - экспорт efficientdet в onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный
        артефакт

        # загружается как любой чекпойнт и возвращает тот же объект Results.

        model = LibreYOLO("LibreEfficientDetd0.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Установка

EfficientDet не нужны опциональные extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращают все семейства, поэтому
замена на другой детектор — правка в одну строку. EfficientDet декодирует
кандидатов на основе якорей, а затем выполняет non-maximum suppression по
классам, поэтому `conf`, `iou` и `max_det` здесь реально влияют на результат.
Про источники, стриминг и обработку результатов — в разделе
[предсказание](/docs/predict).

## Варианты

Пять размеров, от D0 до D4. Каждый следующий шаг соединяет более крупный бэкбон
EfficientNet с более глубоким и широким BiFPN и более глубокой головой
предсказания, поэтому число параметров и объём вычислений растут вместе — по
правилу составного масштабирования из статьи.

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность, полноту,
mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в котором вы
обучали модель.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

Чекпойнты D0-D4 в LibreYOLO сконвертированы через проект
rwightman/efficientdet-pytorch под лицензией Apache-2.0, который сам зеркалит
официальные веса, обученные в TensorFlow, из google/automl, не меняя обученные
тензоры. Исходный код проекта zylo117/Yet-Another-EfficientDet-Pytorch под
лицензией LGPL не изучался и не использовался.

</provenance-box>
