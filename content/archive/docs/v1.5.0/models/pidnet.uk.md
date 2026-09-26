---
title: PIDNet
families:
  - pidnet
seo_title: 'PIDNet: передбачення та експорт сегментації в реальному часі під ліцензією MIT'
description: >-
  Використовуйте PIDNet у LibreYOLO для семантичної сегментації в реальному
  часі. Установлюйте, виконуйте передбачення, валідацію та експорт контрольних
  точок s/m/l для Cityscapes під ліцензією MIT.
lead: >-
  Трьохгілкова мережа семантичної сегментації додає окрему гілку меж до
  архітектури, натхненої пропорційно-інтегрально-диференціальним регулятором і
  розрахованої на інференс у реальному часі. LibreYOLO постачає її лише для
  семантичної сегментації.
keywords:
  - PIDNet
  - семантична сегментація в реальному часі
  - сегментація з урахуванням меж
  - Cityscapes
  - щільне передбачення
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePIDNets-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # Ідентифікатори класів (H, W)

        print(mask.classes)      # Відсортовані ідентифікатори класів на
        зображенні
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibrePIDNets-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## Встановлення

PIDNet не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

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

Доступні три розміри, усі з фіксованим входом 1024 px. Опубліковані контрольні
точки є перетвореннями офіційних ваг PIDNet для Cityscapes із 19 класами.

LibreYOLO не навчає PIDNet: `train()` спричиняє `NotImplementedError` для
цього сімейства, яке [рівень підтримки](/docs/models) вище позначає як
призначене лише для інференсу.

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

## Цитування

<citation-block />
