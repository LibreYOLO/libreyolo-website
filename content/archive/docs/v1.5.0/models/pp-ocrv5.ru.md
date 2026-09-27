---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5: детекция и распознавание текста в LibreYOLO'
description: >-
  Используйте PP-OCRv5 в LibreYOLO для многоязычного OCR по тексту в сцене.
  Установка, предсказание и валидация чекпойнтов t и l, лицензия Apache-2.0.
lead: >-
  PP-OCRv5 — пайплайн детекции и распознавания текста из PaddleOCR: детектор с
  дифференцируемой бинаризацией находит четырёхугольники с текстом, а
  распознаватель SVTR/CTC их читает. LibreYOLO портирует его на PyTorch в двух
  уровнях.
keywords:
  - PP-OCRv5
  - PaddleOCR
  - ocr python
  - распознавание текста на изображении
  - детекция текста
  - распознать текст с фото нейросетью
  - ocr для сканов и чеков
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
    - label: Четырёхугольники
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # полигоны (N, 4, 2) в порядке чтения: верхний левый, верхний правый,
        # нижний правый, нижний левый. Четырёхугольники детекции — настоящие
        # полигоны (повёрнутый текст), поэтому они попадают в result.ocr,
        # а не в result.boxes.
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
        print(metrics["metrics/e2e_f1"])       # главная метрика
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## Установка

PP-OCRv5 не требует ничего сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Каждый чекпойнт упаковывает обе стадии, детекцию и распознавание, в один файл
`.pt`, а набор символов распознавателя и значения пайплайна по умолчанию лежат
в метаданных чекпойнта. Распознаватель читает упрощённый и традиционный
китайский, английский, японский и пиньинь по одному словарю. `result.ocr` — это
структура `OCRRegions`: в `.data` лежат четырёхточечные полигоны, в `.texts` —
распознанный текст, в `.conf` — оценка уверенности распознавания по каждой
области, а в `.det_conf` — оценка детекции. Источники из нескольких изображений
обрабатываются последовательно: двухстадийный пайплайн не собирает батчи из
разных изображений. Об источниках, стриминге и обработке результатов — в разделе
[предсказание](/docs/predict).

## Варианты

Два уровня: `t` на более лёгких бэкбонах PP-LCNetV3/PP-OCRv5_mobile для работы
на CPU и `l` на серверных бэкбонах PP-HGNetV2 для более высокой точности. Оба
уровня запускают детекцию с фиксированным ограничением по длинной стороне и
распознают вырезанные фрагменты батчами; `rec_batch` задаёт, сколько фрагментов
проходит через распознаватель за один прямой проход.

## Валидация

`val()` прогоняет пайплайн по каталогу изображений и файлу
`labels/<split>.jsonl` либо по эквивалентному YAML датасета, где каждая метка
перечисляет полигоны текстовых областей изображения и их текст. Считаются hmean
детекции (точность/полнота/F1 при сопоставлении по IoU), сквозная F1 (тот же
hmean плюс точное совпадение текста после нормализации — метрика fitness
чекпойнта) и 1-NED, среднее нормализованное расстояние редактирования по
сопоставленным парам.

<code-tabs name="val" />

## Экспорт

<export-matrix />

PP-OCRv5 — пайплайн из двух сетей, детекция и распознавание работают вместе, а
не как один трассируемый граф, и экспорт для него не реализован: пока не
поддерживается ни один формат. Если вам нужен чекпойнт вне этого формата,
дообучите напрямую оригинальный код обучения под Apache-2.0 и сконвертируйте
результат скриптом `weights/convert_ppocr_weights.py`.

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
