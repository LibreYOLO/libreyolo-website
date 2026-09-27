---
title: FCOS
families:
  - fcos
seo_title: 'FCOS у LibreYOLO: передбачення, валідація та експорт'
description: >-
  Запускайте FCOS у LibreYOLO для виявлення об'єктів без якорів. Установлюйте,
  виконуйте передбачення, валідацію та експорт порту torchvision ResNet-50/FPN
  під ліцензією BSD-3-Clause.
lead: >-
  FCOS виявляє об'єкти в кожному пікселі замість використання набору попередньо
  визначених якірних рамок, передбачаючи рамку й оцінку центральності в кожній
  позиції карти ознак. LibreYOLO переносить реалізацію torchvision для
  виявлення.
keywords:
  - FCOS
  - детекція без якорів
  - виявлення об'єктів
  - одностадійний детектор
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreFCOSr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## Встановлення

FCOS не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. Якщо викликати модель без аргументів
порогів, застосовуються власні опубліковані типові значення FCOS:
`conf=0.2`, `iou=0.6` і `max_det=100`. Передайте будь-який із трьох
параметрів, щоб його перевизначити. FCOS зберігає завершальний етап NMS над
передбаченнями для кожного пікселя. Типи джерел, потокове передбачення та
обробку результатів описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступний один розмір: ResNet-50 із пірамідою ознак. Це єдиний варіант, який
розпізнає сімейство.

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому проводилося
навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

FCOS експортується до ONNX, TorchScript і OpenVINO. FCOS зберігає пропорції
джерела до запуску графа, тому LibreYOLO примусово вмикає `dynamic=True` для
шляхів ONNX і OpenVINO незалежно від переданого значення, щоб граф залишався
коректним для доповнених вхідних форм. Експортований файл `.onnx` знову
завантажується через `LibreYOLO()` за суфіксом і повертає той самий об'єкт
`Results`.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
