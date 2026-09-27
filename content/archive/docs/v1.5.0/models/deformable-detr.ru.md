---
title: Deformable DETR
families:
  - deformable_detr
seo_title: 'Deformable DETR: предсказание и экспорт, лицензия Apache-2.0'
description: >-
  Запуск Deformable DETR в LibreYOLO для детекции объектов. Установка,
  предсказание, валидация и экспорт пяти размеров с разреженным вниманием, все
  под лицензией Apache-2.0.
lead: >-
  Deformable DETR заменяет плотное перекрёстное внимание из DETR разреженной
  многомасштабной выборкой вокруг каждой опорной точки — именно это сделало
  обучение трансформерных детекторов практичным. LibreYOLO поставляет пять
  размеров для детекции, только для инференса.
keywords:
  - Deformable DETR
  - трансформерный детектор
  - deformable attention
  - разреженное внимание
  - детекция объектов python
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() возвращает обычный dict, а не объект
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## Установка

Deformable DETR не требует опциональных extra. Всё, что он импортирует, входит
в базовую установку, а в основе лежит многомасштабное деформируемое внимание на
чистом PyTorch.

```bash
pip install libreyolo
```

Установка `libreyolo[hub-kernels]` опциональна. Как только пакет `kernels`
доступен, LibreYOLO во время работы скачивает с Hugging Face Hub
скомпилированное ядро многомасштабного деформируемого внимания и использует его
вместо реализации на чистом PyTorch; `LIBREYOLO_HUB_KERNELS=0` выключает это
обратно.

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена одного детектора на другой сводится к одной строке. `conf` и `max_det`
фильтруют отбор запросов; `iou` принимается ради совместимости API, но ни на что
не влияет, потому что декодер предсказывает набор целиком и шага NMS в нём нет.
Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

В LibreYOLO Deformable DETR доступен только для инференса. Оригинальная
реализация обучается с венгерским сопоставлением и focal-функцией потерь для
классификации; этот рецепт здесь не реализован, поэтому `train()` выбрасывает
`NotImplementedError`.

## Варианты

Пять чекпойнтов покрывают опубликованные конфигурации, все с одинаковым
разрешением на входе. `r50ss` ограничивает внимание одним масштабом признаков;
`r50ssdc5` добавляет поверх этого стадию бэкбона C5 с dilated-свёрткой. `r50` —
стандартная многомасштабная конфигурация, которая делает выборку по четырём
уровням карт признаков. `r50refine` добавляет итеративное уточнение
ограничивающих рамок по слоям декодера, а `r50twostage` формирует начальные
предложения областей из выхода энкодера, а не из обучаемых запросов.

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
