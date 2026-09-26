---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: предсказание и экспорт, лицензия Apache-2.0'
description: >-
  Запуск LW-DETR в LibreYOLO для детекции объектов в реальном времени.
  Установка, предсказание, валидация и экспорт пяти размеров на основе ViT, все
  под лицензией Apache-2.0.
lead: >-
  Трансформерный детектор на чистом ViT, который компания Baidu позиционировала
  как альтернативу детекторам YOLO для работы в реальном времени. LibreYOLO
  поставляет пять размеров для детекции, только для инференса.
keywords:
  - LW-DETR
  - трансформерный детектор
  - plain ViT
  - детекция объектов python
  - детекция объектов в реальном времени
  - DETR
  - Baidu
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() возвращает обычный dict, а не объект
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreLWDETRt.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Установка

LW-DETR не требует опциональных extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена одного детектора на другой сводится к одной строке. `conf` и `max_det`
фильтруют отбор запросов; `iou` принимается ради совместимости API, но ни на что
не влияет, потому что декодер предсказывает набор целиком и шага NMS в нём нет.
Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

В LibreYOLO LW-DETR доступен только для инференса. Оригинальная реализация
обучается с сопоставлением «один ко многим» по нескольким группам запросов в
духе Group-DETR и с функцией потерь классификации, учитывающей IoU; этот рецепт
здесь не реализован, поэтому `train()` выбрасывает `NotImplementedError`.

## Варианты

Пять размеров, у всех общий энкодер на чистом ViT, многомасштабный проектор и
декодер deformable DETR, и все работают с одинаковым разрешением на входе. Два
самых маленьких используют одинаковую ширину энкодера и различаются глубиной по
числу блоков; следующие два делят более широкий энкодер и различаются тем,
сколько уровней проектора подаётся в декодер; самый большой переходит к самому
широкому энкодеру.

## Валидация

`val()` возвращает словарь с ключами `metrics/`, куда входят точность, полнота,
mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в котором вы
обучали.

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
