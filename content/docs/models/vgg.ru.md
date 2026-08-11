---
title: VGG
families: [vgg]
seo_title: "VGG: запуск классификаторов изображений VGG-16/19 в LibreYOLO"
description: "Предсказание, валидация и экспорт классификаторов VGG в LibreYOLO. Веса torchvision под лицензией BSD-3-Clause; дообучение пока не поддерживается."
lead: "VGG — свёрточный классификатор изображений, собранный из однородных стопок небольших свёрток 3x3 вместо фильтров большего размера. LibreYOLO поставляет размеры на 16 и 19 слоёв, обычные и с батч-нормализацией, для классификации изображений."
keywords: [VGG, VGG-16, VGG-19, свёрточная нейросеть, классификация изображений python]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreVGG16-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data — это корневой каталог со сплитами train/ и val/ в папках по
        # классам (формат ImageFolder), а не YAML датасета.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreVGG16-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Установка

VGG не требует опциональных extra. Всё, что он импортирует, входит в базовую
установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Классификатор возвращает `result.probs`, а не `result.boxes`: `top1`
и `top5` дают индексы классов, а `top1conf` и `top5conf` — их оценки
уверенности. Предсказание работает с фиксированным входом 224px и завершается
ошибкой, если передать другой `imgsz`. Про источники, стриминг и обработку
результатов см. [предсказание](/docs/predict).

## Варианты

Четыре размера: 16 и 19 свёрточных слоёв, каждый в обычном варианте и в
варианте с батч-нормализацией. Поставляемые веса — это более позднее обучение
с нуля на ImageNet от torchvision, а не конвертация исходного релиза
оксфордской группы 2014 года на Caffe. LibreYOLO поставляет это семейство
только для инференса: предсказание, валидация top-1/top-5 в стиле ImageNet и
экспорт поддерживаются, а дообучение не реализовано.

## Валидация

`val()` работает со сплитом в формате ImageFolder (каталог с подпапками `train/`
и `val/`, по одной папке на класс) и возвращает точность top-1 и top-5.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. [Экспорт](/docs/export) перечисляет аргументы, которые
принимает каждый формат, и те дополнительные, что добавляют некоторые из них.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>
