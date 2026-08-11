---
title: YOLOv1
families: [yolo1]
seo_title: "YOLOv1 в LibreYOLO: предсказание, валидация, экспорт"
description: "Запуск оригинального детектора YOLOv1 в LibreYOLO: замороженное музейное семейство только для инференса. Предсказание, валидация и экспорт под лицензией общественного достояния."
lead: "YOLOv1 — оригинальный детектор 2016 года, давший имя всему семейству YOLO: одна свёрточная сеть с полносвязной головой предсказывает все рамки и оценки классов за один проход, без якорей. LibreYOLO поставляет его как замороженный экспонат только для инференса."
keywords: [YOLOv1, YOLO v1, Darknet, "детекция объектов python", Pascal VOC, "музейные модели libreyolo"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO1b.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же объект Results.
        model = LibreYOLO("LibreYOLO1b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Установка

YOLOv1 не требует ничего сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Это семейство доступно только для инференса: `train()` выбрасывает
`NotImplementedError`, поэтому на этой странице нет раздела про обучение.
Предсказание, валидация и экспорт поддерживаются. Веса скачиваются с
Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращают все семейства, поэтому
замена на другой детектор — правка в одну строку. Две вещи специфичны для
этого семейства. Опубликованный чекпойнт обучен на Pascal VOC (2007+2012), а не
на COCO, поэтому `box.cls` индексирует 20 категорий VOC (aeroplane, bicycle,
bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse,
motorbike, person, pottedplant, sheep, sofa, train, tvmonitor), а не 80
категорий COCO. А полносвязная голова детекции принимает по одному изображению
за раз, поэтому список источников обрабатывается в цикле, а не как настоящий
батч. Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность,
полноту, mAP 50 и mAP 50-95, измеренные на датасете в том же пространстве
меток в стиле VOC, на котором обучался чекпойнт.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запуск графа в чистой среде выполнения, без установленной
LibreYOLO, тоже поддерживается, но тогда препроцессинг и постпроцессинг
придётся писать самостоятельно.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>
