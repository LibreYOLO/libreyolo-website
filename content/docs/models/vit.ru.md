---
title: ViT
families:
  - vit
seo_title: 'ViT: запуск классических классификаторов Vision Transformer в LibreYOLO'
description: >-
  Предсказание, валидация и экспорт классификаторов ViT в LibreYOLO. Веса AugReg
  под лицензией Apache-2.0; дообучение пока не поддерживается.
lead: >-
  Классический Vision Transformer: чистый трансформер, применённый к патчам
  изображения фиксированного размера, с обучаемым class-токеном и без свёрток.
  LibreYOLO поставляет четыре размера с предобучением AugReg для классификации
  изображений.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - классификация изображений python
  - классификатор на трансформере
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

        # data — это корневой каталог со сплитами train/ и val/ в папках по
        # классам (формат ImageFolder), а не YAML датасета.
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
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный

        # артефакт загружается как любой чекпойнт и возвращает тот же объект
        Results.

        model = LibreYOLO("LibreViTti-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Установка

ViT не требует опциональных extra. Всё, что он импортирует, входит в базовую
установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Классификатор возвращает `result.probs`, а не `result.boxes`: `top1`
и `top5` дают индексы классов, а `top1conf` и `top5conf` — их оценки
уверенности. Предобработка масштабирует изображение и вырезает центральную
область до фиксированного входа 224px по рецепту оценки AugReg из timm:
бикубическая интерполяция с долей кропа 0.9. Про источники, стриминг и
обработку результатов см. [предсказание](/docs/predict).

## Варианты

Четыре размера, от tiny до large, разделяющих один и тот же граф —
фиксированные 224px, patch-16 — и различающихся шириной эмбеддинга и глубиной
трансформера. LibreYOLO поставляет это семейство только для инференса:
предсказание, валидация top-1/top-5 в стиле ImageNet и экспорт
поддерживаются, а рецепт дообучения AugReg не реализован.

## Валидация

`val()` работает со сплитом в формате ImageFolder (каталог с подпапками `train/`
и `val/`, по одной папке на класс) и возвращает точность top-1 и top-5.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. [Экспорт](/docs/export) перечисляет аргументы, которые
принимает каждый формат, и те дополнительные, что добавляют некоторые из них.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
