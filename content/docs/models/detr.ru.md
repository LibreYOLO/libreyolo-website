---
title: DETR
families:
  - detr
seo_title: 'DETR: предсказание и экспорт, лицензия Apache-2.0'
description: >-
  Запуск DETR, исходного трансформерного детектора, в LibreYOLO. Установка,
  предсказание, валидация и экспорт четырёх размеров на основе ResNet, все под
  лицензией Apache-2.0.
lead: >-
  DETR — исходный трансформерный детектор: он предсказывает фиксированный набор
  объектов трансформерным декодером с венгерским сопоставлением, без якорей и
  плотной сетки. LibreYOLO поставляет четыре размера для детекции, только для
  инференса.
keywords:
  - DETR
  - трансформерный детектор
  - детекция объектов python
  - венгерское сопоставление
  - трансформерный декодер
  - Meta AI
  - Facebook AI Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() возвращает обычный dict, а не объект
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает загрузчик по расширению файла, поэтому
        экспортированный

        # артефакт загружается как любой чекпойнт и возвращает тот же объект
        Results.

        model = LibreYOLO("LibreDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## Установка

DETR не требует опциональных extra. Всё, что он импортирует, входит в базовую
установку.

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

В LibreYOLO DETR доступен только для инференса. Оригинальная реализация
обучается 500 эпох с венгерским сопоставлением; этот рецепт здесь не реализован,
поэтому `train()` выбрасывает `NotImplementedError`.

## Варианты

Четыре чекпойнта сочетают две глубины бэкбона, ResNet-50 или ResNet-101, с
опциональной стадией C5 с dilated-свёрткой: варианты DC5 оставляют последнюю
стадию бэкбона в полном разрешении вместо дальнейшего понижения, поэтому декодер
читает более детальную карту признаков при том же размере входа. У всех четырёх
одинаковые 100 обучаемых объектных запросов и шестислойный трансформерный
энкодер-декодер, и все работают на одном и том же разрешении входа.

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
