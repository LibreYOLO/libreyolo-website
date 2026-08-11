---
title: Swin Transformer
families: [swin]
seo_title: "Swin Transformer: классификация изображений с LibreSwin в LibreYOLO"
description: "Предсказание, валидация и экспорт классификаторов Swin Transformer в LibreYOLO. Веса под MIT; дообучение пока не поддерживается."
lead: "Swin Transformer V1 — иерархический vision transformer, который считает внимание внутри сдвинутых локальных окон, а не по всему изображению. LibreYOLO поставляет четыре размера для классификации изображений."
keywords: [Swin Transformer, иерархический vision transformer, внимание в сдвинутых окнах, классификация изображений python, классификатор ImageNet]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSwint-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data — это корень каталога со сплитами train/ и val/, разложенными
        # по папкам классов (структура ImageFolder), а не YAML датасета.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как обычный чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreSwint-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Установка

Для Swin не нужны дополнительные extra-зависимости. Всё, что он импортирует,
входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Классификатор возвращает `result.probs`, а не `result.boxes`: `top1` и `top5`
дают индексы классов, а `top1conf` и `top5conf` — их уверенности. У каждого
размера вход жёстко зафиксирован на 224px, потому что последняя стадия
внимания рассчитана именно на это разрешение; предсказание, валидация и
экспорт выбрасывают ошибку, если передать другой `imgsz`. Про источники,
стриминг и обработку результатов см. [предсказание](/docs/predict).

## Варианты

Четыре размера, от tiny до large, собраны из одной и той же башни со
сдвинутыми окнами и различаются шириной эмбеддинга и глубиной стадий. Размер
large предобучен на ImageNet-22k и дообучен на ImageNet-1k; остальные три
обучены сразу на ImageNet-1k. LibreYOLO поставляет это семейство только для
инференса: предсказание, валидация top-1/top-5 в стиле ImageNet и экспорт
поддерживаются, а оригинальный рецепт обучения на ImageNet не реализован.

## Валидация

`val()` работает по разбиению в стиле ImageFolder (каталог с подпапками
`train/` и `val/`, по одной папке на класс) и возвращает top-1 и top-5
accuracy.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. В разделе [экспорт](/docs/export) перечислены аргументы,
которые принимает каждый формат, и дополнительные, которые добавляют некоторые
из них.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
