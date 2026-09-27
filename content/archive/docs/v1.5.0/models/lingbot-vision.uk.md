---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: семантична сегментація в LibreYOLO'
description: >-
  Використовуйте LingBot-Vision у LibreYOLO для семантичної сегментації на
  бекбоні ViT за Apache-2.0. Установлення, передбачення, навчання, валідація й
  експорт, розміри s/b/l.
lead: >-
  LingBot-Vision, це сімейство бекбонів-трансформерів зору із самоконтрольованим
  навчанням через орієнтоване на межі масковане моделювання для щільного
  просторового сприйняття, випущене Robbyant. LibreYOLO поєднує бекбон зі
  щільною головою й підтримує його для однієї задачі, семантичної сегментації.
keywords:
  - LingBot-Vision
  - семантична сегментація
  - трансформер зору
  - самоконтрольоване попереднє навчання
  - моделювання меж
  - Robbyant
  - щільне передбачення
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (лінійне дослідження)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Бекбон стандартно заморожений відповідно до протоколу оцінювання
        # upstream: навчається лише щільна голова 1x1.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Повне донавчання
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Використання експортованого файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика вибирає маршрут за суфіксом файла, тому експортований артефакт

        # завантажується як контрольна точка й повертає той самий об'єкт
        Results.

        model = LibreYOLO("LibreLingBotVisions-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Встановлення

LingBot-Vision не потребує необов'язкових залежностей. Усе, що вона імпортує,
є в базовому встановленні.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

`result.semantic_mask` містить щільну карту класів: `.data`, це тензор
`(H, W)` ідентифікаторів класів у розмірі вихідного зображення, а
`.classes` перелічує фактично наявні класи. `result.boxes` дорівнює
`None`, оскільки виявлень окремих екземплярів немає. `conf` і `iou`
приймаються для узгодженості API, але не змінюють результат, адже модель
повертає один клас на піксель, а не виявлення для фільтрування. Джерела,
потокову обробку й роботу з результатами описано в розділі
[передбачення](/docs/predict).

## Варіанти

Опубліковано три розміри, s, b і l, дистильовані з учителя ViT-g/16 на 1,1
млрд параметрів. Сам учитель розміру `g` завантажується й донавчається в
LibreYOLO, але LibreYOLO не розміщує власну контрольну точку `g`.

<checkpoint-table />

## Навчання

`train()` донавчає опубліковану контрольну точку. Стандартний рецепт, це
лінійне дослідження зі звіту upstream: бекбон ViT заморожений, і навчається
лише щільна голова 1x1, як під час створення наведених вище ваг на хостингу
LibreYOLO. Передайте `freeze_backbone=False`, щоб натомість донавчати всю
мережу, і відповідно зменште `lr0`.

<code-tabs name="train" />

Датасети, аугментацію, кілька GPU й логери описано в розділі
[навчання](/docs/train).

## Валідація

`val()` повертає словник ключів `metrics/`: mIoU та піксельну точність,
виміряні на будь-якому датасеті у форматі навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт завантажується через `LibreYOLO()` за суфіксом
файла, тому файл `.onnx` або `.engine` поводиться як контрольна точка й
повертає той самий `Results`. Аргументи кожного формату перелічено в
розділі [експорт](/docs/export).

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

Випуск upstream описує свій ViT як побудований на архітектурі DINOv2/DINOv3,
опублікованій Meta AI. Robbyant розповсюджує власну реалізацію за Apache-2.0,
і це перенесення LibreYOLO створено лише з репозиторію Robbyant, без читання
чи копіювання коду DINOv2 або DINOv3 від Meta.

</provenance-box>

## Цитування

<citation-block />

