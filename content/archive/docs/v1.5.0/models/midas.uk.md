---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: монокулярне оцінювання глибини в LibreYOLO'
description: >-
  Використовуйте MiDaS у LibreYOLO для монокулярного оцінювання глибини.
  Установлюйте, виконуйте передбачення, валідацію та експорт двох варіантів під
  ліцензією MIT, завантажених з isl-org.
lead: >-
  MiDaS виконує монокулярне оцінювання відносної глибини та навчається зі сталою
  до масштабу й зсуву функцією втрат на змішаних датасетах. Ця лінія досліджень
  започаткувала протокол перенесення глибини без навчальних прикладів, який
  повторно використовують пізніші сімейства. LibreYOLO підтримує її для завдання
  глибини: передбачення та валідація без прикладів без шляху навчання.
keywords:
  - MiDaS
  - монокулярне оцінювання глибини
  - DPT
  - відносна глибина
  - карта глибини
  - zero-shot depth
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Якщо файлу ще немає на диску, LibreYOLO завантажує його з офіційного
        # релізу isl-org/MiDaS на GitHub і перевіряє зафіксований SHA-256.
        model = LibreYOLO("LibreMiDaSl-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Малий варіант
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Енкодер EfficientNet-Lite3, менший і швидший за розмір DPT-Large l.
        model = LibreYOLO("LibreMiDaSs-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreMiDaSl-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: ce2fbf3ae43e9be4
---

## Встановлення

MiDaS не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

MiDaS є єдиним сімейством глибини, яке LibreYOLO не перевидає у власній
організації Hugging Face. Запит контрольної точки за назвою файлу LibreYOLO
завантажує відповідний офіційний ресурс безпосередньо з релізів
`isl-org/MiDaS` на GitHub, перевіряє за зафіксованим SHA-256 і перед першим
використанням обгортає метаданими контрольної точки LibreYOLO. Наступні
запуски повторно використовують кешований локальний файл. Причину пояснено в
розділі «Ліцензування».

<code-tabs name="predict" />

`result.depth_map` містить щільну карту відносної оберненої глибини: більші
значення означають ближче до камери, а значення не мають метричної одиниці чи
спільного масштабу між зображеннями. `save=True` записує на диск візуалізацію
цієї карти з кольоровою шкалою; `Results.plot()` не охоплює це сімейство,
оскільки визначений лише для нормалей поверхні та країв. Типи джерел, потокове
передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступні два варіанти з різними енкодерами, а не просто різними масштабами
одного. `s` є MiDaS v2.1 Small з енкодером EfficientNet-Lite3. `l` є
DPT-Large з енкодером ViT-L/16 і декодером DPT, який MiDaS запровадила для
щільного передбачення. Вони також мають різну попередню обробку: `s`
використовує зміну розміру зі збереженням пропорцій і верхньою межею та
нормалізацію середнім/стандартним відхиленням ImageNet; `l` використовує
мінімальну зміну розміру зі збереженням пропорцій і середнє та стандартне
відхилення 0.5. Вибирайте `s` для легшої CNN, а `l` для правильності
трансформерного декодера.

Навчання для цього сімейства не пропонується. `LibreMiDaS.train()` завжди
спричиняє `NotImplementedError`.

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

<code-tabs name="export" />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
