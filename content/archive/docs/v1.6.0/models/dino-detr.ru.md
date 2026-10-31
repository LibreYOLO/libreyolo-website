---
title: DINO-DETR
families:
  - dinodetr
seo_title: 'DINO-DETR: предсказание и экспорт, лицензия Apache-2.0'
description: >-
  Запуск DINO-DETR в LibreYOLO для детекции объектов. Установка, предсказание,
  валидация и экспорт трёх размеров с шумоподавляющими якорями, все под
  лицензией Apache-2.0.
lead: >-
  DINO-DETR, опубликованный IDEA Research под названием DINO, сочетает
  контрастивное обучение с шумоподавлением и смешанный отбор запросов поверх
  разреженного внимания из Deformable DETR. LibreYOLO поставляет три размера для
  детекции, только для инференса.
keywords:
  - DINO-DETR
  - DINO
  - трансформерный детектор
  - denoising anchor boxes
  - mixed query selection
  - детекция объектов python
  - IDEA Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() возвращает обычный dict, а не объект
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreDINODETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## Установка

DINO-DETR не требует опциональных extra. Всё, что он импортирует, входит в
базовую установку, а в основе лежит то же многомасштабное деформируемое внимание
на чистом PyTorch, что и в семействе Deformable DETR из LibreYOLO.

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

В LibreYOLO DINO-DETR доступен только для инференса. Оригинальная реализация
обучается с контрастивным шумоподавлением и венгерским сопоставлением; этот
рецепт здесь не реализован, поэтому `train()` выбрасывает `NotImplementedError`.

## Варианты

Три чекпойнта, все с одинаковым разрешением на входе. `r50` и `r50s5`
используют общий бэкбон ResNet-50 и различаются тем, сколько масштабов карт
признаков подаётся в декодер: четыре против пяти. `swinl` меняет бэкбон на
Swin-L и тоже делает выборку по пяти масштабам.

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

<provenance-box>

Три официальных чекпойнта взяты из папки релиза в Google Drive от авторов, а не
из карточки модели на Hugging Face. Оригинальный репозиторий объявляет
Apache-2.0 на уровне репозитория, но не прикладывает к самим чекпойнтам ни файла
лицензии, ни метаданных о ней, поэтому основанием для распространения служит
именно это объявление на уровне репозитория, а не отдельное разрешение для
конкретного чекпойнта. Каждое зеркало LibreYOLO поставляется с дословным текстом
лицензии Apache-2.0 из оригинала и с примечанием, которое это объясняет.

</provenance-box>

## Цитирование

<citation-block />
