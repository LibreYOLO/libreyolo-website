---
title: YOLOv4
families:
  - yolo4
seo_title: 'YOLOv4: запуск, валидация и экспорт в LibreYOLO'
description: >-
  Запуск YOLOv4 в LibreYOLO: замороженное музейное семейство только для
  инференса с бэкбоном CSPDarknet-53. Предсказание, валидация и экспорт под
  лицензией общественного достояния.
lead: >-
  YOLOv4 соединяет бэкбон CSPDarknet-53, блок SPP и neck PANet с активациями
  Mish. LibreYOLO поставляет его как замороженный экспонат только для инференса
  в размерах tiny и base.
keywords:
  - YOLOv4
  - Darknet
  - CSPDarknet-53
  - PANet
  - детекция объектов python
  - активация Mish
  - музейные модели libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO4b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO4b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO4b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO4b.pt format=onnx
        libreyolo export model=LibreYOLO4b.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный

        # артефакт загружается как любой чекпойнт и возвращает тот же объект
        Results.

        model = LibreYOLO("LibreYOLO4b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 6070bb4a09d75416
---

## Установка

YOLOv4 не требует ничего сверх базового пакета.

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
замена на другой детектор — правка в одну строку. `conf` фильтрует по порогу
уверенности, а `iou` — по порогу NMS; оба применяются после собственного
масштабирования центров `scale_x_y` в каждой голове. Про источники, стриминг и
обработку результатов см. [предсказание](/docs/predict).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность,
полноту, mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в
котором вы запускаете валидацию.

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

## Цитирование

<citation-block />
