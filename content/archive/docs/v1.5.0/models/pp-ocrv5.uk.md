---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5: виявлення та розпізнавання тексту в LibreYOLO'
description: >-
  Використовуйте PP-OCRv5 у LibreYOLO для багатомовного OCR тексту на сценах.
  Установлюйте, виконуйте передбачення та валідацію контрольних точок t і l під
  ліцензією Apache-2.0.
lead: >-
  PP-OCRv5 є пайплайном PaddleOCR для виявлення та розпізнавання тексту:
  детектор із диференційовною бінаризацією знаходить чотирикутники тексту, а
  розпізнавач SVTR/CTC читає їх. LibreYOLO переносить його до PyTorch у двох
  рівнях.
keywords:
  - PP-OCRv5
  - PaddleOCR
  - OCR
  - розпізнавання тексту
  - детекція тексту
  - текст на зображенні
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Чотирикутники
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRl-ocr.pt")

        result = model(SAMPLE_IMAGE)


        # Полігони (N, 4, 2) у порядку читання: верхній лівий, верхній правий,

        # нижній правий, нижній лівий. Чотирикутники виявлення є справжніми

        # полігонами (повернутий текст), тому вони заповнюють result.ocr, а не
        result.boxes.

        print(result.ocr.data.shape)

        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # Основна метрика
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## Встановлення

Для PP-OCRv5 не потрібні додаткові залежності понад базовий пакет.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Кожна контрольна точка об'єднує обидва етапи, виявлення та розпізнавання, в
одному файлі `.pt`. Набір символів розпізнавання й типові параметри пайплайна
зберігаються в метаданих контрольної точки. Один словник дає розпізнавачу
змогу читати спрощену й традиційну китайську, англійську, японську та піньїнь.
`result.ocr` є корисним навантаженням `OCRRegions`: `.data` містить
чотириточкові полігони, `.texts` містить транскрипції, `.conf` містить
оцінку розпізнавання для кожної області, а `.det_conf` містить оцінку
виявлення. Джерела з кількох зображень обробляються послідовно: двостадійний
пайплайн не формує батч із різних зображень. Типи джерел, потокове
передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступні два рівні: `t` використовує легші бекбони
PP-LCNetV3/PP-OCRv5_mobile для CPU, а `l` використовує серверні бекбони
PP-HGNetV2 для вищої правильності. Обидва рівні виконують виявлення з
фіксованим обмеженням довгої сторони й розпізнають кропи батчами; `rec_batch`
задає кількість кропів, що проходять через розпізнавач за один прямий прохід.

## Валідація

`val()` вимірює пайплайн на каталозі зображень і файлі
`labels/<split>.jsonl` або еквівалентному YAML датасету. Кожна мітка містить
полігони текстових областей зображення та їхні транскрипції. Метод повідомляє
гармонійне середнє виявлення (точність/повнота/F1 зі зіставленням за IoU),
наскрізну F1 (гармонійне середнє разом із точним збігом нормалізованої
транскрипції, метрика придатності контрольної точки) і 1-NED, середню
нормалізовану відстань редагування для зіставлених пар.

<code-tabs name="val" />

## Експорт

<export-matrix />

PP-OCRv5 є пайплайном із двох мереж, у якому виявлення та розпізнавання
переміщуються разом, а не одним трасованим графом. Експорт для нього не
реалізовано, і жоден формат поки не підтримується. Якщо потрібна контрольна
точка поза цим форматом, донавчіть початковий код навчання під ліцензією
Apache-2.0 та перетворіть результат за допомогою
`weights/convert_ppocr_weights.py`.

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
