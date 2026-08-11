---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet у LibreYOLO: передбачення, валідація та експорт'
description: >-
  Запускайте RetinaNet у LibreYOLO для одностадійного виявлення об'єктів із
  фокальною функцією втрат. Установлюйте, виконуйте передбачення, валідацію та
  експорт порту torchvision під ліцензією BSD-3-Clause.
lead: >-
  RetinaNet є одностадійним детектором, навченим із фокальною функцією втрат,
  яка зменшує вагу простих негативних прикладів, тому щільна сітка якорів
  зберігає правильність без окремого етапу пропозицій. LibreYOLO переносить
  реалізацію torchvision для виявлення.
keywords:
  - RetinaNet
  - фокальна функція втрат
  - детекція об'єктів
  - одностадійний детектор
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreRetinaNetr50v2.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Встановлення

RetinaNet не потребує додаткових залежностей. Усе, що вона імпортує, входить
до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. `conf` і `iou` задають пороги
впевненості та NMS. RetinaNet зберігає початковий етап NMS над щільною сіткою
якорів. Типи джерел, потокове передбачення та обробку результатів описано в
розділі [передбачення](/docs/predict).

## Варіанти

Доступні два розміри, обидва з ResNet-50 і пірамідою ознак: `r50` має
початкову голову, а `r50v2` замінює її головою GroupNorm і ширшим блоком P6,
який отримує дані з останнього етапу бекбона, а не з виходу FPN.

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому проводилося
навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

RetinaNet експортується лише до ONNX із розміром батча 1. RetinaNet змінює
розмір зі збереженням пропорцій до змінного входу, тому LibreYOLO примусово
вмикає `dynamic=True` незалежно від переданого значення, щоб граф залишався
коректним для джерел різної форми. Експортований файл `.onnx` знову
завантажується через `LibreYOLO()` за суфіксом і повертає той самий об'єкт
`Results`.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>
