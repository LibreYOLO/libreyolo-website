---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: передбачення та експорт семантичної сегментації ASPP'
description: >-
  Використовуйте DeepLabv3 у LibreYOLO для семантичної сегментації.
  Установлюйте, виконуйте передбачення, валідацію та експорт контрольних точок
  ResNet і MobileNetV3 від torchvision.
lead: >-
  Мережа семантичної сегментації паралельно агрегує ознаки з кількома
  коефіцієнтами розширення (просторовий пірамідальний пулінг з отворами) перед
  класифікацією кожного пікселя. LibreYOLO постачає її лише для семантичної
  сегментації.
keywords:
  - DeepLabv3
  - atrous spatial pyramid pooling
  - ASPP
  - семантична сегментація
  - щільне передбачення
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # Ідентифікатори класів (H, W)

        print(mask.classes)      # Відсортовані ідентифікатори класів на
        зображенні
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---

## Встановлення

DeepLabv3 не потребує додаткових залежностей. Усе, що вона імпортує, входить
до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально. Суфікс `-sem` у назві файлу обов'язковий для цього сімейства.

<code-tabs name="predict" />

Семантична сегментація повертає один ідентифікатор класу на піксель, а не
рамки, тому `result.semantic_mask` містить масив `(H, W)` у `.data` і
список наявних на зображенні ідентифікаторів класів у `.classes`. Параметри
`conf`, `iou` і `max_det` приймаються для узгодженості API, але не
впливають на результат: модель призначає клас кожному пікселю за argmax без
порога впевненості чи етапу NMS. Типи джерел, потокове передбачення та обробку
результатів описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступні три бекбони: розширений ResNet-50, розширений ResNet-101 і розширений
MobileNetV3-Large. Це DeepLabv3, а не DeepLabv3+, тому тут немає етапу декодера
чи уточнення CRF. Реалізація відповідає torchvision, а не власному еталонному
коду зі статті.

LibreYOLO не навчає DeepLabv3: `train()` спричиняє `NotImplementedError`
для цього сімейства, яке [рівень підтримки](/docs/models) вище позначає як
призначене лише для інференсу. Три опубліковані контрольні точки є власними
вагами torchvision, навченими на COCO з мітками VOC і перетвореними для
завантажувача LibreYOLO.

## Валідація

`val()` повертає `metrics/mIoU` і `metrics/pixel_accuracy`, виміряні на
будь-якому датасеті у форматі, на якому проводилося навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results`. У розділі
[експорту](/docs/export) наведено аргументи, які приймає кожен формат.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>
