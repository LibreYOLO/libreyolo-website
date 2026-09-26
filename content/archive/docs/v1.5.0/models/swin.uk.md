---
title: Swin Transformer
families:
  - swin
seo_title: 'Swin Transformer: класифікація зображень за допомогою LibreSwin у LibreYOLO'
description: >-
  Виконуйте передбачення, валідацію та експорт класифікаторів Swin Transformer
  за допомогою LibreYOLO. Ваги під ліцензією MIT; донавчання ще не
  підтримується.
lead: >-
  Swin Transformer V1 є ієрархічним візуальним трансформером, який обчислює
  увагу всередині зміщених локальних вікон, а не на всьому зображенні. LibreYOLO
  постачає чотири розміри для класифікації зображень.
keywords:
  - Swin Transformer
  - ієрархічний візуальний трансформер
  - увага у зміщених вікнах
  - класифікація зображень
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data є кореневим каталогом із поділом на папки класів train/ і val/
        # (структура ImageFolder), а не YAML датасету.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreSwint-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## Встановлення

Swin не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Класифікатор повертає `result.probs` замість `result.boxes`: `top1` і
`top5` містять індекси класів, а `top1conf` і `top5conf` містять їхні
оцінки впевненості. Усі розміри мають фіксований вхід 224px, оскільки
завершальний етап уваги побудовано для цієї роздільної здатності. Передбачення,
валідація та експорт спричиняють помилку, якщо передати інше значення `imgsz`.
Типи джерел, потокове передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступні чотири розміри від tiny до large, побудовані з однієї вежі зміщених
вікон і відмінні шириною ембедингу та глибиною етапів. Large попередньо
навчено на ImageNet-22k і донавчено на ImageNet-1k; інші три навчено
безпосередньо на ImageNet-1k. LibreYOLO постачає це сімейство лише для
інференсу: підтримуються передбачення, валідація top-1/top-5 у стилі ImageNet
та експорт, а початковий рецепт навчання ImageNet не реалізовано.

## Валідація

`val()` працює з поділом у стилі ImageFolder (каталогом із підкаталогами
`train/` і `val/`, що містять по одній папці на клас) та повертає
правильність top-1 і top-5.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results`. У розділі
[експорту](/docs/export) наведено аргументи, які приймає кожен формат, і
додаткові параметри деяких форматів.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
