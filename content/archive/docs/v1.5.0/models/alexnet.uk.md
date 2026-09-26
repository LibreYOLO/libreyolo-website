---
title: AlexNet
families:
  - alexnet
seo_title: 'AlexNet: запуск класичного класифікатора ImageNet у LibreYOLO'
description: >-
  Виконуйте передбачення, валідацію та експорт AlexNet за допомогою LibreYOLO.
  Ваги torchvision під ліцензією BSD-3-Clause; донавчання ще не підтримується.
lead: >-
  AlexNet є згортковою мережею, яка перемогла в ILSVRC 2012 і допомогла
  започаткувати еру глибокого навчання в комп'ютерному зорі. LibreYOLO постачає
  пізнішу одногілкову редакцію архітектури для класифікації зображень.
keywords:
  - AlexNet
  - ImageNet
  - згорткова нейронна мережа
  - класифікація зображень
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data є кореневим каталогом із поділом на папки класів train/ і val/
        # (структура ImageFolder), а не YAML датасету.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreAlexNetb-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## Встановлення

AlexNet не потребує додаткових залежностей. Усе, що вона імпортує, входить до
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
оцінки впевненості. Типи джерел, потокове передбачення та обробку результатів
описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступний один розмір. Постачаний граф є пізнішою одногілковою редакцією від
torchvision із 64 фільтрами першого шару й без локальної нормалізації відгуку,
а не початковою архітектурою 2012 року для двох GPU. LibreYOLO постачає це
сімейство лише для інференсу: підтримуються передбачення, валідація top-1/top-5
у стилі ImageNet та експорт, а донавчання не реалізовано.

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
