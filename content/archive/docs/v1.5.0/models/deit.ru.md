---
title: DeiT
families:
  - deit
seo_title: 'Классификатор изображений DeiT: предсказание, валидация, экспорт'
description: >-
  Запуск классификаторов изображений DeiT в LibreYOLO: замороженное музейное
  семейство только для инференса, размеры tiny, small и base, лицензия
  Apache-2.0.
lead: >-
  DeiT (Data-efficient image Transformer) — классификатор на обычном Vision
  Transformer, обученный только на ImageNet-1k, без дополнительных данных для
  предобучения. LibreYOLO поставляет размеры tiny, small и base с патчами 16 как
  замороженный экспонат только для инференса.
keywords:
  - DeiT
  - Vision Transformer
  - ViT
  - классификация изображений python
  - ImageNet
  - предобученный трансформер для классификации
  - музейные модели libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает загрузчик по расширению файла, поэтому
        экспортированный

        # артефакт загружается как любой чекпойнт и возвращает тот же объект
        Results.

        model = LibreYOLO("LibreDeiTb-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## Установка

DeiT не требует ничего сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Это семейство доступно только для инференса: `train()` выбрасывает
`NotImplementedError`, поэтому на этой странице нет раздела про обучение.
Предсказание, валидация и экспорт поддерживаются. Веса скачиваются с
Hugging Face при первом запуске и кэшируются локально. Суффикс `-cls` в имени
файла обязателен и выбирает задачу классификации.

<code-tabs name="predict" />

Возвращаемый объект `Results` несёт тензор `probs`, а не `boxes`; `top1` и
`top5` индексируют 1000 классов ImageNet-1k, а `top1conf` — это softmax-оценка
для лучшего предсказания. У каждого размера своё фиксированное входное
разрешение, заданное позиционным эмбеддингом: препроцессинг масштабирует
изображение и обрезает его по центру до этого разрешения, а другое значение
`imgsz` приводит к ошибке, а не к молчаливому пересэмплированию. Про источники,
стриминг и обработку результатов см. [предсказание](/docs/predict).

## Валидация

`val()` возвращает словарь с точностью top-1 и top-5, измеренной на датасете,
разложенном по обычной структуре папок `train/<class>/` и `val/<class>/`.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запуск графа в чистой среде выполнения, без установленной
LibreYOLO, тоже поддерживается, но тогда препроцессинг и постпроцессинг
придётся писать самостоятельно.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
