---
title: YOLOv3
families:
  - yolo3
seo_title: 'YOLOv3 в LibreYOLO: предсказание, валидация, экспорт'
description: >-
  Запуск YOLOv3 в LibreYOLO: замороженное музейное семейство только для
  инференса с размерами tiny, base и SPP. Предсказание, валидация и экспорт под
  лицензией public domain.
lead: >-
  YOLOv3 — детектор на Darknet-53, который добавил в линейку YOLO
  многомасштабное предсказание и независимые логистические классификаторы. В
  LibreYOLO он поставляется как замороженный музейный экспонат только для
  инференса в размерах tiny, base и SPP.
keywords:
  - YOLOv3
  - Darknet
  - Darknet-53
  - детекция объектов python
  - запустить yolov3
  - многомасштабная детекция
  - музейные модели yolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Размер SPP
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Вариант SPP добавляет блок spatial pyramid pooling перед головами
        # детекции и работает со своим родным размером входа.
        model = LibreYOLO("LibreYOLO3spp.pt")
        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreYOLO3b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## Установка

Для YOLOv3 не нужны дополнительные extra сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Это семейство доступно только для инференса: `train()` выбрасывает
`NotImplementedError`, поэтому на этой странице нет раздела про обучение.
Предсказание, валидация и экспорт поддерживаются. Веса скачиваются с Hugging
Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращают все семейства, поэтому
замена на другой детектор — правка в одну строку. `conf` фильтрует по порогу
уверенности, а `iou` — по порогу NMS; оба применяются на каждом масштабе до
того, как рамки со всех трёх голов объединяются. Про источники, стриминг и
обработку результатов — в разделе [предсказание](/docs/predict).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность,
полноту, mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в
котором вы проводите валидацию.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запуск графа в чистой среде выполнения, без установленной
библиотеки LibreYOLO, тоже поддерживается, но тогда предобработку и
постобработку придётся писать самостоятельно.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>
