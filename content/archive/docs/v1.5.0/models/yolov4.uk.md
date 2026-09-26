---
title: YOLOv4
families:
  - yolo4
seo_title: 'YOLOv4: запуск, валідація та експорт у LibreYOLO'
description: >-
  Запускайте YOLOv4 у LibreYOLO: заморожене музейне сімейство лише для інференсу
  з бекбоном CSPDarknet-53. Виконуйте передбачення, валідацію та експорт за
  умовами ліцензії суспільного надбання.
lead: >-
  YOLOv4 поєднує бекбон CSPDarknet-53, блок SPP і neck PANet з активаціями Mish.
  LibreYOLO зберігає її як заморожений експонат лише для інференсу в розмірах
  tiny і base.
keywords:
  - YOLOv4
  - Darknet
  - CSPDarknet-53
  - PANet
  - детекція об'єктів
  - активація Mish
  - музейне сімейство моделей
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
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreYOLO4b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 6070bb4a09d75416
---

## Встановлення

Для YOLOv4 не потрібні додаткові залежності понад базовий пакет.

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
детектора потребує зміни одного рядка. `conf` фільтрує за порогом упевненості,
а `iou` задає поріг NMS. Обидва параметри застосовуються після власного
масштабування центра `scale_x_y` кожної голови. Типи джерел, потокове
передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому виконується
валідація.

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

## Цитування

<citation-block />
