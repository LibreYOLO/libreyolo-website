---
title: TEED
families:
  - teed
seo_title: 'TEED: виявлення країв із власною контрольною точкою'
description: >-
  Використовуйте TEED у LibreYOLO для щільного передбачення ймовірності країв.
  Перетворіть ліцензовану контрольну точку, а потім виконуйте передбачення,
  валідацію та експорт.
lead: >-
  TEED (Tiny and Efficient Edge Detector) є малою згортковою мережею, яка
  передбачає щільну карту ймовірності країв з одного зображення RGB. LibreYOLO
  обгортає її архітектуру лише для виявлення країв; контрольна точка не
  постачається з бібліотекою.
keywords:
  - TEED
  - Tiny and Efficient Edge Detector
  - детекція країв
  - виявлення контурів
  - BIPED
  - щільне передбачення
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)        # (H, W) float32 у діапазоні [0, 1]

        print(edges.binary(0.5).sum())  # Кількість пікселів країв після
        порогування
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        metrics = model.val(data="my-dataset.yaml", imgsz=352)


        print(metrics["metrics/ODS"])   # F-міра в оптимальному масштабі
        датасету

        print(metrics["metrics/OIS"])   # F-міра в оптимальному масштабі
        зображення
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352

        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Використання експортованого файлу
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## Встановлення

TEED не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

LibreYOLO не постачає контрольної точки TEED. Офіційно випущені ваги навчено
на BIPED, опубліковані умови датасету якого обмежують використання
некомерційними цілями, тому LibreYOLO не створює їхнього дзеркала. Перетворіть
контрольну точку, на використання якої маєте ліцензію, за допомогою
`weights/convert_teed_weights.py`. Скрипт перевіряє ключі тензорів відносно
архітектури середовища виконання, перш ніж записати файл, який LibreYOLO може
завантажити безпосередньо:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` містить результат: масив float32 `(H, W)` у діапазоні
`[0, 1]`; `.binary(threshold)` повертає булеву маску країв. Рамок немає,
тому `conf`, `iou` і `max_det` не впливають на результат. Типи джерел,
потокове передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

LibreYOLO постачає TEED в одному розмірі. Засоби бенчмарків LibreYOLO не
вимірювали це сімейство, тому опублікованих показників для порівняння немає.

## Валідація

`val()` повідомляє F-міри ODS і OIS у стилі BSDS для парного датасету країв:
зображення поруч із картами країв з однаковою основою назви й необов'язковою
маскою дійсності, щоб доповнені пікселі ніколи не враховувалися. `imgsz` має
ділитися на крок зменшення мережі, а в іншому разі LibreYOLO показує зрозумілу
помилку.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експорт країв використовує контракт середовища виконання з фіксованою
роздільною здатністю та розміром батча 1: `dynamic` і `batch` зі значенням,
відмінним від 1, відхиляються, а експортований граф виводить одну об'єднану
карту ймовірностей. Експортований артефакт знову завантажується через
`LibreYOLO()` відповідно до суфікса файлу, тому файл `.onnx` поводиться як
контрольна точка й повертає той самий об'єкт `Results`.

<code-tabs name="export" />

## Ліцензування

<provenance-box>

LibreYOLO не публікує контрольної точки TEED. Організація LibreYOLO не
розміщує жодного дзеркала; натомість перетворіть контрольну точку, на яку
маєте ліцензію, за допомогою `weights/convert_teed_weights.py`.

</provenance-box>

## Цитування

<citation-block />
