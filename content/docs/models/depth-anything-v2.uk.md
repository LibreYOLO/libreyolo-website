---
title: Depth Anything V2
families:
  - depth_anything
seo_title: 'Depth Anything V2: передбачення та валідація монокулярної глибини'
description: >-
  Використовуйте Depth Anything V2 у LibreYOLO для монокулярного оцінювання
  глибини. Установлюйте, виконуйте передбачення та валідацію; Small має ліцензію
  Apache-2.0, Base і Large мають CC-BY-NC-4.0.
lead: >-
  Depth Anything V2 є енкодером DINOv2 із декодером DPT, який передбачає щільну
  карту відносної оберненої глибини з одного зображення. LibreYOLO підтримує її
  для завдання глибини: передбачення та валідація без прикладів без шляху
  навчання.
keywords:
  - Depth Anything V2
  - монокулярне оцінювання глибини
  - DPT
  - DINOv2
  - відносна глибина
  - карта глибини
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Читання карти глибини
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: щільна (H, W), більше означає
        ближче

        raw = depth.data                # Тензор без метричної одиниці чи
        масштабу між зображеннями

        normalized = depth.normalized() # Масштабовано до [0, 1] для
        візуалізації
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## Встановлення

Depth Anything V2 не потребує додаткових залежностей. Усе, що вона імпортує,
входить до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

`result.depth_map` містить щільну карту відносної оберненої глибини: більші
значення означають ближче до камери, а значення не мають метричної одиниці чи
спільного масштабу між зображеннями. `save=True` записує на диск візуалізацію
цієї карти з кольоровою шкалою; `Results.plot()` не охоплює це сімейство,
оскільки визначений лише для нормалей поверхні та країв. Вхідна роздільна
здатність має ділитися на 14, розмір сітки патчів DINOv2, на якій побудовано
голову DPT. LibreYOLO перевіряє це до запуску й спричиняє помилку за
невідповідності. Типи джерел, потокове передбачення та обробку результатів
описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступні чотири розміри енкодера s/b/l/g, що відповідають ViT-S/B/L/G. У
таблиці контрольних точок нижче наведено лише s, b і l; контрольну точку Giant
не опубліковано. Усі чотири мають однакову вхідну роздільну здатність, тому
вибір розміру змінює місткість енкодера, а не розмір зображення. Ліцензія також
впливає на вибір: контрольна точка Small має Apache-2.0, а Base і Large мають
CC-BY-NC-4.0, див. розділ «Ліцензування» нижче.

Навчання й донавчання для цього сімейства не пропонуються.
`LibreDepthAnythingV2.train()` завжди спричиняє `NotImplementedError`;
натомість перетворіть сумісну початкову контрольну точку за допомогою
`weights/convert_depth_anything_v2_weights.py`.

## Валідація

`val()` запускає спільний валідатор глибини: він вирівнює кожне передбачення
з еталонними даними за масштабом і зсувом методом найменших квадратів для
кожного зображення, а потім повідомляє стандартні метрики відносної глибини
без навчальних прикладів AbsRel, RMSE і три пороги delta.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results` із `depth_map` замість рамок.
У розділі [експорту](/docs/export) наведено аргументи, які приймає кожен
формат.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
