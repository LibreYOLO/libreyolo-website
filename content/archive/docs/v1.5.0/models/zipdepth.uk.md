---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: легке монокулярне оцінювання глибини в LibreYOLO'
description: >-
  Використовуйте ZipDepth у LibreYOLO для легкого монокулярного оцінювання
  глибини. Установлюйте, виконуйте передбачення, валідацію та експорт двох
  контрольних точок під ліцензією MIT.
lead: >-
  ZipDepth є компактною репараметризованою CNN, дистильованою з Depth Anything
  V2 Large, яка передбачає щільну карту відносної оберненої глибини. LibreYOLO
  підтримує її для завдання глибини: передбачення та валідація без прикладів без
  шляху навчання.
keywords:
  - ZipDepth
  - монокулярне оцінювання глибини
  - edge-модель глибини
  - відносна глибина
  - карта глибини
  - репараметризована CNN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Контрольна точка NPU/edge
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Той самий енкодер із головою підвищення роздільної здатності без
        unfold

        # для компіляторів без підтримки gather/unfold. Вихід візуально
        еквівалентний b.

        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreZipDepthb-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Встановлення

ZipDepth не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

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
оскільки визначений лише для нормалей поверхні та країв. Типи джерел, потокове
передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступні дві контрольні точки з однаковою місткістю енкодера, які
відрізняються лише навченою головою підвищення роздільної здатності. `b`
використовує опукле підвищення роздільної здатності та працює на GPU або CPU.
`bnpu` замінює його декодером без unfold для NPU та edge-компіляторів, які
не підтримують gather/unfold; його вихід задокументовано як візуально
еквівалентний `b`. Вибирайте `bnpu` для обмеженого середовища виконання,
до якого експортується модель, і `b` в інших випадках.

Обидві контрольні точки дистильовано з псевдоміток Depth Anything V2 Large,
тому це сімейство є компактним, орієнтованим на edge-пристрої рівнем завдання
глибини LibreYOLO поряд із більшими енкодерами Depth Anything V2.

Навчання для цього сімейства не пропонується. `LibreZipDepth.train()` завжди
спричиняє `NotImplementedError`: початковий рецепт дистилює псевдомітки на
великому наборі зображень, що не відтворюється як запуск навчання LibreYOLO.
Навчіть модель засобами початкового проєкту
[fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth) і перетворіть
результат за допомогою `weights/convert_zipdepth_weights.py`.

## Валідація

`val()` запускає спільний валідатор глибини: він вирівнює кожне передбачення
з еталонними даними за масштабом і зсувом методом найменших квадратів для
кожного зображення, а потім повідомляє стандартні метрики відносної глибини
без навчальних прикладів AbsRel, RMSE і три пороги delta.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експорт використовує щільний контракт із фіксованою роздільною здатністю:
початкове зображення розтягується до експортованого полотна, а повернена карта
глибини потім масштабується до початкового полотна. Експортований артефакт
знову завантажується через `LibreYOLO()` відповідно до суфікса файлу, тому
файл `.onnx` або `.ncnn` поводиться як контрольна точка й повертає той
самий об'єкт `Results` із `depth_map` замість рамок.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
