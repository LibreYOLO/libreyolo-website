---
title: ConvNeXt
families:
  - convnext
seo_title: 'ConvNeXt: обучение, валидация и экспорт под Apache-2.0'
description: >-
  Используйте ConvNeXt в LibreYOLO для классификации изображений. Установка,
  предсказание, дообучение с LoRA, валидация и экспорт LibreConvNeXt
  tiny/small/base.
lead: >-
  ConvNeXt — классификатор изображений, целиком построенный на обычных свёртках:
  его блок за блоком модернизировали от ResNet в сторону проектных решений
  vision transformer. LibreYOLO поддерживает его для одной задачи —
  классификации.
keywords:
  - ConvNeXt
  - ConvNeXt tiny
  - классификация изображений python
  - чисто свёрточная сеть
  - классификатор ImageNet
last_verified: 1.6.0

snippets:
  predict:
  - label: Python
    language: python
    code: |
      from libreyolo import LibreYOLO, SAMPLE_IMAGE

      model = LibreYOLO("LibreConvNeXtt-cls.pt")
      result = model(SAMPLE_IMAGE, save=True)

      print(result.probs.top1, result.probs.top1conf)
      print(result.probs.top5)
  - label: CLI
    language: bash
    code: |
      libreyolo predict model=LibreConvNeXtt-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  train:
  - label: Python
    language: python
    code: |
      from libreyolo import LibreYOLO

      model = LibreYOLO("LibreConvNeXtt-cls.pt")
      model.train(data="imagenette160", epochs=5)
  - label: CLI
    language: bash
    code: |
      libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
  - label: LoRA
    language: python
    code: |
      from libreyolo import LibreYOLO

      model = LibreYOLO("LibreConvNeXtt-cls.pt")
      model.train(data="imagenette160", epochs=5, lora=True)
  - label: Multi-GPU
    language: bash
    code: |
      libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
        epochs=50 device=0,1 batch=-1
  val:
  - label: Python
    language: python
    code: |
      from libreyolo import LibreYOLO

      model = LibreYOLO("LibreConvNeXtt-cls.pt")
      metrics = model.val(data="imagenette160")

      print(metrics["metrics/accuracy_top1"])
      print(metrics["metrics/accuracy_top5"])
  - label: CLI
    language: bash
    code: |
      libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
  - label: Python
    language: python
    code: |
      from libreyolo import LibreYOLO

      model = LibreYOLO("LibreConvNeXtt-cls.pt")
      model.export(format="onnx")
      model.export(format="tensorrt", half=True)
  - label: CLI
    language: bash
    code: |
      libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
      libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
  - label: Использование экспортированного файла
    language: python
    code: |
      from libreyolo import LibreYOLO, SAMPLE_IMAGE

      # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
      # артефакт загружается как обычный чекпойнт и возвращает тот же Results.
      model = LibreYOLO("LibreConvNeXtt-cls.onnx")
      result = model(SAMPLE_IMAGE)

      print(result.probs.top1)

source_hash: 8f33a40e4d3a8f29
---

## Установка

Для ConvNeXt не нужны дополнительные extra-зависимости. Всё, что он
импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

Исключение — дообучение адаптерами с `lora=True`: ему нужен extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство,
поэтому переход на другую модель сводится к правке в одну строку. У
классификатора нет ни рамок, ни масок: `result.probs` содержит предсказание
для всего изображения, с полями `top1`, `top5`, `top1conf` и `top5conf`.
Аргументы `conf`, `iou` и `max_det` принимаются ради единообразия API, но ни
на что не влияют: в одном векторе вероятностей нечего отсекать по порогу и
нечего подавлять. Про источники, стриминг и обработку результатов —
[предсказание](/docs/predict).

## Варианты

Три размера, tiny/small/base, все обучены и оценены одинаково, поэтому выбор
сводится к прямому размену числа параметров на точность. Задача
зафиксирована: каждый размер закрывает только классификацию. Имя файла весов
у всех размеров заканчивается на `-cls.pt`, и именно по этому суффиксу
фабрика определяет семейство; аргумент `task=` не нужен.

## Обучение

Дообучение стартует с опубликованного бэкбона ImageNet и автоматически
перестраивает последний слой классификатора под число классов целевого
датасета.

<code-tabs name="train" />

Если ничего не менять, обучение идёт 100 эпох с `lr0=1e-3` и AdamW, размером
батча 64 и ранней остановкой после 50 эпох без улучшений. `data` принимает
корень датасета (`train/` и `val/`, по одной папке на класс), известное
короткое имя вроде `imagenette160` или URL на `.zip`. В блоках ConvNeXt есть
MLP на `nn.Linear`, которые нужны LoRA, поэтому `lora=True` здесь
поддерживается: адаптеры внедряются в MLP блоков, а не дообучается весь
бэкбон.

Про датасеты, аугментацию, обучение на нескольких GPU и логгеры —
[обучение](/docs/train).

`cls_pw=0` отключает взвешивание функции потерь; значения до 1 используют веса, обратные частоте классов и нормализованные к среднему 1. `class_weights=True` использует обратные частоты, нормализованные по числу примеров, и не сочетается с `cls_pw>0`. При возобновлении эти настройки должны совпадать. См. [классификацию](/docs/tasks/image-classification).

## Валидация

`val()` возвращает словарь с ключами `metrics/`. Для классификации это top-1
и top-5 accuracy на валидационной выборке.

<code-tabs name="val" />

Валидация и калибровка INT8 используют преобразование семейства для оценки. Метаданные экспорта записывают `norm_mean`, `norm_std` и `resize_mode`; старые артефакты используют значения семейства. Предобработчики калибровки возвращают требуемые массив CHW и коэффициент масштаба.

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и
возвращает те же `Results`. В разделе [экспорт](/docs/export) перечислены
аргументы, которые принимает каждый формат, и дополнительные, которые
добавляют некоторые из них.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

Эта страница посвящена ConvNeXt V1. [ConvNeXt V2](/docs/models/convnextv2) представляет отдельное семейство; его официальные предобученные веса сохраняют лицензию CC-BY-NC-4.0.

</provenance-box>

## Цитирование

<citation-block />
