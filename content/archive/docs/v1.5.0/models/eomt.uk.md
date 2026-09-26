---
title: EoMT
families:
  - eomt
seo_title: >-
  EoMT: передбачення семантичної сегментації, сегментації екземплярів і
  паноптичної сегментації
description: >-
  Використання EoMT у LibreYOLO для семантичної сегментації, сегментації
  екземплярів і паноптичної сегментації на звичайному візуальному трансформері
  DINOv2 без декодера. Ліцензія MIT.
lead: >-
  Мережа сегментації на основі звичайного візуального трансформера без
  спеціалізованого піксельного декодера: маски передбачають додаткові навчені
  запити, додані до самого кодера. LibreYOLO підтримує її для семантичної
  сегментації, сегментації екземплярів і паноптичної сегментації.
keywords:
  - EoMT
  - encoder-only mask transformer
  - DINOv2
  - паноптична сегментація
  - сегментація екземплярів
  - семантична сегментація
last_verified: 1.5.0
snippets:
  predict:
    - label: Семантична сегментація
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreEoMTl-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # ідентифікатори класів (H, W)

        print(mask.classes)      # відсортовані ідентифікатори класів, наявних
        на зображенні
    - label: Сегментація екземплярів
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суфікс -seg у назві файлу вибирає завдання екземплярів, тому
        # аргумент завдання тут не потрібен.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Паноптична сегментація
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # ідентифікатори сегментів (H, W)
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Семантична сегментація
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Сегментація екземплярів
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # маски
        print(metrics["metrics/mAP50-95(B)"])   # рамки
    - label: Паноптична сегментація
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreEoMTl-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## Встановлення

EoMT не потребує додаткових пакетів. Усі його імпорти входять до базового
встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face та кешуються
локально. Суфікс завдання в назві файлу (`-sem`, `-seg`, `-panoptic`) вибирає
завдання, а `LibreYOLO()` визначає його за назвою, тому аргумент `task=` не
потрібен.

<code-tabs name="predict" />

Семантична сегментація заповнює `result.semantic_mask`, масив ідентифікаторів
класів форми `(H, W)` у `.data`. Сегментація екземплярів заповнює
`result.boxes` і `result.masks` тієї самої форми, яку повертають інші сімейства
сегментації. Паноптична сегментація заповнює `result.panoptic`: карту
ідентифікаторів сегментів форми `(H, W)` у `.data` та `.segments_info`, список
словників `{"id", "category_id"}`, по одному на сегмент. Параметр `conf`
фільтрує вибір запитів; `iou` не впливає на семантичне завдання, оскільки воно
вибирає argmax для кожного пікселя без етапу NMS. Джерела, потокове оброблення
та роботу з результатами описано на сторінці [передбачення](/docs/predict).

## Варіанти

Доступні три розміри кодера, s/b/l, усі на основі DINOv2. Контрольну точку
семантичної сегментації навчено на ADE20K із 512 px; контрольні точки сегментації
екземплярів і паноптичної сегментації навчено на COCO із 640 px, а другу
контрольну точку сегментації екземплярів навчено із 1280 px. Першоджерело
публікує ваги сегментації екземплярів DINOv2 лише для розміру l; s і b
опубліковано тільки для семантичної та паноптичної сегментації. Варіанти EoMT
на основі DINOv3 існують у першоджерелі, але не постачаються тут, оскільки
залежать від закритих некомерційних ваг DINOv3.

LibreYOLO не навчає EoMT: `train()` спричиняє `NotImplementedError` для цього
сімейства, яке наведений вище [рівень підтримки](/docs/models) позначає як
призначене лише для інференсу.

## Валідація

Метод `val()` виконує диспетчеризацію за завданням. Семантична сегментація
повертає `metrics/mIoU` і `metrics/pixel_accuracy`. Сегментація екземплярів
повертає ті самі ключі mAP масок і рамок, що й інші сімейства сегментації.
Паноптична сегментація повертає Panoptic Quality як `metrics/PQ`, розділену на
`metrics/SQ` (якість сегментації) та `metrics/RQ` (якість розпізнавання), а також
`metrics/PQ_things` і `metrics/PQ_stuff`.

<code-tabs name="val" />

## Експорт

<export-matrix />

Наразі експортується лише семантичне завдання: для сегментації екземплярів і
паноптичної сегментації виклик `export()` спричиняє `NotImplementedError`,
оскільки їхній вихід запитів і масок ще не має контракту експорту для середовища
виконання. Експортований семантичний артефакт повторно завантажується через
`LibreYOLO()` за суфіксом файлу, тому файл `.onnx` або `.engine` поводиться як
контрольна точка й повертає той самий `Results`.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />

