---
title: VGG
families:
  - vgg
seo_title: 'VGG: запуск класифікаторів зображень VGG-16/19 у LibreYOLO'
description: >-
  Виконуйте передбачення, валідацію та експорт класифікаторів VGG за допомогою
  LibreYOLO. Ваги torchvision під ліцензією BSD-3-Clause; донавчання ще не
  підтримується.
lead: >-
  VGG є згортковим класифікатором зображень, побудованим з однорідних стеків
  малих згорток 3x3 замість більших фільтрів. LibreYOLO постачає 16- і 19-шарові
  розміри у звичайних варіантах і з пакетною нормалізацією для класифікації
  зображень.
keywords:
  - VGG
  - VGG-16
  - VGG-19
  - згорткова нейронна мережа
  - класифікація зображень
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data є кореневим каталогом із поділом на папки класів train/ і val/
        # (структура ImageFolder), а не YAML датасету.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreVGG16-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## Встановлення

VGG не потребує додаткових залежностей. Усе, що вона імпортує, входить до
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
оцінки впевненості. Передбачення виконується з фіксованим входом 224px і
спричиняє помилку, якщо передати інше значення `imgsz`. Типи джерел,
потокове передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступні чотири розміри: 16 і 19 згорткових шарів, кожен у звичайному варіанті
та з пакетною нормалізацією. Постачані ваги отримано під час пізнішого навчання
torchvision на ImageNet з нуля, а не шляхом перетворення початкового релізу
групи Oxford для Caffe 2014 року. LibreYOLO постачає це сімейство лише для
інференсу: підтримуються передбачення, валідація top-1/top-5 у стилі ImageNet
та експорт, а донавчання не реалізовано.

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
