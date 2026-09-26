---
title: ViT
families:
  - vit
seo_title: 'ViT: запуск класичних класифікаторів Vision Transformer у LibreYOLO'
description: >-
  Виконуйте передбачення, валідацію та експорт класифікаторів ViT за допомогою
  LibreYOLO. Ваги AugReg під ліцензією Apache-2.0; донавчання ще не
  підтримується.
lead: >-
  Класичний Vision Transformer є чистим трансформером, застосованим до патчів
  зображення фіксованого розміру, із навченим токеном класу та без згорток.
  LibreYOLO постачає чотири попередньо навчені за AugReg розміри для
  класифікації зображень.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - класифікація зображень
  - класифікатор transformer
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # data є кореневим каталогом із поділом на папки класів train/ і val/
        # (структура ImageFolder), а не YAML датасету.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreViTti-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Встановлення

ViT не потребує додаткових залежностей. Усе, що вона імпортує, входить до
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
оцінки впевненості. Попередня обробка змінює розмір і виконує центральне
кадрування до фіксованого входу 224px за рецептом оцінювання AugReg із
бібліотеки timm: бікубічна інтерполяція з часткою кадрування 0.9. Типи джерел,
потокове передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступні чотири розміри від tiny до large. Усі мають один фіксований граф
224px із патчем 16 і відрізняються шириною ембедингу та глибиною трансформера.
LibreYOLO постачає це сімейство лише для інференсу: підтримуються передбачення,
валідація top-1/top-5 у стилі ImageNet та експорт, а рецепт донавчання AugReg
не реалізовано.

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
