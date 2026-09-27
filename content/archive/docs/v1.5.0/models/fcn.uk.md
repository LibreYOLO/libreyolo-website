---
title: FCN
families:
  - fcn
seo_title: 'FCN: передбачення та експорт FCN із ResNet під ліцензією BSD-3-Clause'
description: >-
  Використовуйте FCN у LibreYOLO для семантичної сегментації. Установлюйте,
  виконуйте передбачення, валідацію та експорт контрольних точок FCN із
  розширеним ResNet від torchvision.
lead: >-
  Щільний попіксельний класифікатор замінює повнозв'язні шари детектора
  згортками, тому виводить повнорозмірну карту класів замість рамок. LibreYOLO
  постачає його лише для семантичної сегментації.
keywords:
  - FCN
  - повністю згорткова мережа
  - семантична сегментація
  - щільне передбачення
  - ResNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreFCNr50.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # Ідентифікатори класів (H, W)

        print(mask.classes)      # Відсортовані ідентифікатори класів на
        зображенні
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreFCNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Встановлення

FCN не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Семантична сегментація повертає один ідентифікатор класу на піксель, а не
рамки, тому `result.semantic_mask` містить масив `(H, W)` у `.data` і
список наявних на зображенні ідентифікаторів класів у `.classes`. Параметри
`conf`, `iou` і `max_det` приймаються для узгодженості API, але не
впливають на результат: модель призначає клас кожному пікселю за argmax без
порога впевненості чи етапу NMS. Типи джерел, потокове передбачення та обробку
результатів описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступні дві глибини ResNet, обидві з фіксованим входом 520 px. Граф інференсу
бібліотеки є FCN від torchvision із розширеним ResNet, а не мережею FCN-8s на
основі VGG із пропускними з'єднаннями з початкової статті.

LibreYOLO не навчає FCN: `train()` спричиняє `NotImplementedError` для цього
сімейства, яке [рівень підтримки](/docs/models) вище позначає як призначене
лише для інференсу. Дві опубліковані контрольні точки є власними вагами
torchvision, навченими на COCO й перетвореними для завантажувача LibreYOLO.

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
