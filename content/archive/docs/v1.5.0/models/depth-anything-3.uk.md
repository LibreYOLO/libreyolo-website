---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: 'Depth Anything 3: передбачення монокулярної глибини в LibreYOLO'
description: >-
  Використовуйте Depth Anything 3 у LibreYOLO для монокулярного оцінювання
  глибини. Установлюйте, виконуйте передбачення, валідацію та експорт
  контрольної точки DA3MONO-LARGE під ліцензією Apache-2.0.
lead: >-
  Depth Anything 3 є звичайним трансформером DINOv2, навченим передбачати
  глибину й геометрію камери з одного або кількох ракурсів без архітектурної
  спеціалізації. LibreYOLO переносить її контрольну точку DA3MONO-LARGE для
  завдання глибини: передбачення та валідація без прикладів без шляху навчання.
keywords:
  - Depth Anything 3
  - DA3
  - монокулярне оцінювання глибини
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

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Читання карти глибини
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnything3l-depth.pt")

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

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---

## Встановлення

Depth Anything 3 не потребує додаткових залежностей. Усе, що вона імпортує,
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
спільного масштабу між зображеннями. Початкова контрольна точка виводить
додатну відносну глибину; мережева обгортка LibreYOLO інвертує її та відтворює
офіційну обробку неба, щоб вихід відповідав спільному контракту глибини
LibreYOLO. `save=True` записує на диск візуалізацію цієї карти з кольоровою
шкалою; `Results.plot()` не охоплює це сімейство, оскільки визначений лише
для нормалей поверхні та країв. Типи джерел, потокове передбачення та обробку
результатів описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступний один розмір `l` із фіксованою вхідною роздільною здатністю.
Початковий проєкт DA3 також публікує контрольні точки Small і Base для
довільної кількості ракурсів, контрольну точку метричної глибини та контрольні
точки Nested і Giant; LibreYOLO не відкриває жодної з них. Метрична глибина
потребує іншого публічного контракту, ніж завдання відносної оберненої глибини
LibreYOLO, а контрольні точки довільних ракурсів і Nested потребують API
камери з кількома зображеннями, якого LibreYOLO не пропонує. Контрольні точки
Large і Giant для довільних ракурсів також мають CC-BY-NC-4.0 і не згадуються
жодним шляхом завантаження LibreYOLO.

Навчання для цього сімейства не пропонується. `LibreDepthAnything3.train()`
завжди спричиняє `NotImplementedError`; навчіть модель засобами початкового
проєкту й перетворіть сумісну контрольну точку DA3MONO-LARGE за допомогою
`weights/convert_depth_anything3_weights.py`.

## Валідація

`val()` запускає спільний валідатор глибини: він вирівнює кожне передбачення
з еталонними даними за масштабом і зсувом методом найменших квадратів для
кожного зображення, а потім повідомляє стандартні метрики відносної глибини
без навчальних прикладів AbsRel, RMSE і три пороги delta.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експорт для цього сімейства обмежено п'ятьма форматами: ONNX, TorchScript,
ExecuTorch, TensorRT і OpenVINO. Запит будь-якого іншого формату спричиняє
`NotImplementedError` замість спроби неперевіреного перетворення.
Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results` із `depth_map` замість рамок.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
