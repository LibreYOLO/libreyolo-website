---
title: MobileNetV4
families:
  - mobilenetv4
seo_title: 'MobileNetV4: обучение, валидация и экспорт под Apache-2.0'
description: >-
  Используйте MobileNetV4 в LibreYOLO для классификации изображений. Установка,
  предсказание, дообучение, валидация и экспорт LibreMobileNetV4
  small/medium/large.
lead: >-
  MobileNetV4 — классификатор изображений, рассчитанный на мобильное и
  edge-железо: блок Universal Inverted Bottleneck сводит несколько прежних
  вариантов мобильных блоков в одну структуру, по которой можно вести поиск
  архитектуры. LibreYOLO поддерживает его для одной задачи — классификации.
keywords:
  - MobileNetV4
  - MobileNetV4 conv
  - классификация изображений python
  - инференс на мобильных устройствах
  - классификатор для edge
  - классификатор ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMobileNetV4s-cls.pt source=cat.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160
        epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMobileNetV4s-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMobileNetV4s-cls.pt format=onnx

        libreyolo export model=LibreMobileNetV4s-cls.pt format=tensorrt
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как обычный чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreMobileNetV4s-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 4a9a1b392ffb136d
---

## Установка

Для MobileNetV4 не нужны дополнительные extra-зависимости. Всё, что он
импортирует, входит в базовую установку.

```bash
pip install libreyolo
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

Три размера, small/medium/large, все только свёрточные: гибридные варианты с
вниманием Mobile MQA в это семейство не входят. Выбор размера — прямой размен
числа параметров на точность. Задача зафиксирована: каждый размер закрывает
только классификацию. Имя файла весов у всех размеров заканчивается на
`-cls.pt`, и именно по этому суффиксу фабрика определяет семейство; аргумент
`task=` не нужен.

## Обучение

Дообучение стартует с опубликованного бэкбона ImageNet и автоматически
перестраивает последний слой классификатора под число классов целевого
датасета.

<code-tabs name="train" />

Если ничего не менять, обучение идёт 100 эпох с `lr0=1e-3` и AdamW, размером
батча 64 и ранней остановкой после 50 эпох без улучшений. `data` принимает
корень датасета (`train/` и `val/`, по одной папке на класс), известное
короткое имя вроде `imagenette160` или URL на `.zip`. `lora=True` здесь не
поддерживается: при передаче этого аргумента возникает ошибка, потому что LoRA
в LibreYOLO работает с трансформерными компонентами со слоями `nn.Linear`, а в
блоках UIB этого семейства их нет.

Про датасеты, аугментацию, обучение на нескольких GPU и логгеры —
[обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`. Для классификации это top-1
и top-5 accuracy на валидационной выборке.

<code-tabs name="val" />

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

<provenance-box></provenance-box>
