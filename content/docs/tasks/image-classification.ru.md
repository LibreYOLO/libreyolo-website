---
title: Классификация изображений
seo_title: Классификация изображений в LibreYOLO
description: >-
  Присвоить метку изображению целиком в LibreYOLO: семейства, которые закрывают
  задачу, структура датасета ImageFolder и вызовы предсказания, обучения,
  валидации и экспорта.
lead: >-
  Классификация изображений присваивает одно распределение меток всему
  изображению и ничего внутри него не локализует. Ключ задачи — classify.
keywords:
  - классификация изображений python
  - обучить классификатор изображений
  - датасет ImageFolder
  - top-1 accuracy
  - zero-shot классификация
  - библиотека классификации изображений mit
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суффикс -cls в имени файла выбирает задачу, поэтому аргумент
        # task не нужен.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Полное распределение
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)

        probs = result.probs


        # .data — полный вектор (C,); top5/top5conf — упорядоченные
        представления.

        print(probs.data.shape)

        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 'Zero-shot, без обучения'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP сопоставляет изображение с текстовыми промптами, поэтому набор

        # меток задаётся в момент вызова, а не зашит в чекпойнт.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # imagenette160 — известное имя датасета, оно скачивается при первом
        # использовании. Для своих данных передайте каталог со сплитом train/.
        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() возвращает обычный словарь, а не объект.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как чекпойнт и возвращает тот же объект Results.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Определение

Классификация изображений выдаёт по одной оценке на класс для всего изображения
и никаких координат не даёт. Она отвечает на вопрос «что на картинке», но не
«где», и именно этим отличается от
[детекции объектов](/docs/tasks/object-detection).

`classify` — канонический ключ задачи, а выбирает его суффикс `-cls` в имени
файла чекпойнта. Для семейств классификации этот суффикс обязателен, а не
опционален, поэтому `LibreResNet50.pt` классификатором не считается, а
`LibreResNet50-cls.pt` — считается.

`predict()` заполняет `result.probs` и оставляет `boxes` пустым. `.data` —
полный вектор оценок, `.top1` — индекс наибольшей оценки, а `.top1conf` — её
значение, `.top5` — индексы пяти наибольших оценок по убыванию, а `.top5conf` —
соответствующие оценки. Индексы указывают на элементы `result.names`. Срез
объекта `Results` никогда не обрезает `probs`, потому что вектор относится ко
всему изображению, а не к одной строке.

## Модели

Пять семейств поддерживают и обучение, и предсказание: [ResNet](/docs/models/resnet),
[ConvNeXt](/docs/models/convnext), [MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) и
[DINOv2](/docs/models/dinov2). Первым четырём достаточно базового пакета, и для
них опубликованы веса. Для DINOv2 нужен `pip install "libreyolo[rfdetr]"`, и
чекпойнта, размещённого LibreYOLO, для него нет: он загружает исходный бэкбон со
случайно инициализированной линейной головой, поэтому это отправная точка для
дообучения, а не готовая для предсказаний модель.

Ещё пять поддерживают предсказание, валидацию и экспорт, но их `train()`
выбрасывает `NotImplementedError`: [ViT](/docs/models/vit),
[Swin](/docs/models/swin), [VGG](/docs/models/vgg),
[AlexNet](/docs/models/alexnet) и [DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) и [SigLIP2](/docs/models/siglip2) классифицируют без
фиксированного набора меток. Они сопоставляют изображение с текстовыми
промптами, поэтому классы задаёт `set_classes()` в момент вызова, и для нового
набора меток обучение не нужно вообще. Оба также закрывают задачу `embed`.

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`conf`, `iou` и `max_det` здесь ни на что не влияют: отсекать по порогу и
подавлять нечего, кандидатов нет — есть одно распределение. Источники, стриминг
и обработка результатов описаны в разделе [предсказание](/docs/predict).

## Формат датасета

Классификация использует дерево каталогов, а не файлы разметки и не YAML.
`data` — это корень датасета.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` обязателен для обучения и задаёт соответствие классов индексам по
отсортированным именам папок, так что первая по алфавиту папка становится
классом 0. `val/` обязателен для валидации. Сплит `test/` может присутствовать, и
команды обучения и валидации по умолчанию его не используют. Любой сплит, кроме
`train`, обязан содержать те же имена папок классов, что и ожидаемый набор
классов, — именно поэтому расхождение падает с явной ошибкой, а не засчитывается
как неверное предсказание. Допустимые расширения изображений — `.jpg`, `.jpeg`,
`.png`, `.bmp`, `.webp`, `.tif` и `.tiff`.

`data` принимает три вещи: путь к каталогу со сплитом `train/`, URL на `.zip`
или одно из известных имён датасетов, `imagenette160` и `smoke10`, которые
скачиваются и кэшируются при первом использовании.

Канонический загрузчик — `libreyolo.data.classify_dataset`.

## Обучение

<code-tabs name="train" />

Объявлять `nc` не нужно: количество классов берётся из имён папок внутри
`train/`, а последний линейный слой пересобирается под него, тогда как бэкбон
переносится без изменений. Датасеты, аугментация, обучение на нескольких GPU и
логгеры описаны в разделе [обучение](/docs/train).

## Валидация

`val()` возвращает обычный словарь с ключами `metrics/`, посчитанный по сплиту
`val/` в корне датасета.

<code-tabs name="val" />

`metrics/accuracy_top1` — доля изображений, у которых класс с наибольшей оценкой
оказался верным; это главное число, именно по нему обучение выбирает лучшую
эпоху. `metrics/accuracy_top5` — доля изображений, у которых верный класс попал
хоть куда-то в пятёрку классов с наибольшими оценками, и чем меньше классов в
датасете, тем меньше смысла в этой метрике. В словаре есть и `fitness` — копия
значения top-1.

## Экспорт

<code-tabs name="export" />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Набор поддерживаемых форматов различается по семействам;
матрица на странице каждой модели генерируется из проверенного набора, а не
набивается руками. Форматы, их дополнительные зависимости и ограничения описаны
в разделе [экспорт и развёртывание](/docs/export).
