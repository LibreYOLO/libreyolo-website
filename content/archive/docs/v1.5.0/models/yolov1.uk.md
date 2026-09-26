---
title: YOLOv1
families:
  - yolo1
seo_title: 'YOLOv1 у LibreYOLO: передбачення, валідація, експорт'
description: >-
  Запускайте початковий детектор YOLOv1 у LibreYOLO: заморожене музейне
  сімейство лише для інференсу. Виконуйте передбачення, валідацію та експорт за
  умовами ліцензії суспільного надбання.
lead: >-
  YOLOv1 є початковим детектором 2016 року, який дав назву сімейству YOLO: одна
  згорткова мережа з повнозв'язною головою передбачає всі рамки й оцінки класів
  за один прохід без якірних рамок. LibreYOLO зберігає її як заморожений
  експонат лише для інференсу.
keywords:
  - YOLOv1
  - YOLO v1
  - Darknet
  - детекція об'єктів
  - Pascal VOC
  - музейне сімейство моделей
last_verified: 1.5.0
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
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
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
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreYOLO1b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## Встановлення

Для YOLOv1 не потрібні додаткові залежності понад базовий пакет.

```bash
pip install libreyolo
```

## Передбачення

Це сімейство призначене лише для інференсу: `train()` спричиняє
`NotImplementedError`, тому на цій сторінці немає розділу «Навчання».
Передбачення, валідація та експорт підтримуються. Під час першого використання
ваги завантажуються з Hugging Face і кешуються локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. Це сімейство має дві особливості.
Опубліковану контрольну точку навчено на Pascal VOC (2007+2012), а не COCO,
тому `box.cls` індексує 20 категорій VOC (aeroplane, bicycle, bird, boat,
bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person,
pottedplant, sheep, sofa, train, tvmonitor), а не 80 категорій COCO.
Повнозв'язна голова виявлення приймає по одному зображенню, тому список джерел
обробляється циклом, а не як справжній батч. Типи джерел, потокове передбачення
та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на датасеті з тим самим простором міток у стилі VOC, на
якому навчено контрольну точку.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results`. Граф також можна запускати
безпосередньо в середовищі виконання без установленої LibreYOLO, але тоді
попередню та подальшу обробку потрібно реалізувати самостійно.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>
