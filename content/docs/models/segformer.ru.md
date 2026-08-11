---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: семантическая сегментация в LibreYOLO'
description: >-
  Используйте SegFormer в LibreYOLO для семантической сегментации ADE20K в
  размерах b0–b5. Установка, предсказание, обучение и экспорт; предобученные
  веса — некоммерческие.
lead: >-
  SegFormer — трансформер для семантической сегментации, который сочетает
  иерархический энкодер Mix Transformer (MiT) с лёгкой декодирующей головой из
  одних только MLP и обходится без тяжёлых декодеров и фиксированных позиционных
  кодировок, нужных более ранним трансформерам для сегментации. В LibreYOLO он
  поддержан для одной задачи — семантической сегментации — в шести размерах.
keywords:
  - SegFormer
  - семантическая сегментация python
  - сегментация изображений трансформер
  - Mix Transformer
  - MiT
  - ADE20K
  - обучить segformer на своём датасете
  - segformer onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (дообучение)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: С нуля
      language: python
      code: |
        from libreyolo.models.segformer.model import LibreSegformer

        # Без model_path: случайная инициализация, ничего не скачивается.
        # Единственный способ получить веса без некоммерческого условия
        # предобученных чекпойнтов.
        model = LibreSegformer(size="b0", nb_classes=150)
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Установка

SegFormer не требует опциональных extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`result.semantic_mask` хранит плотную карту классов: `.data` — это тензор
`(H, W)` с id классов в размере исходного изображения, а `.classes`
перечисляет id классов, которые реально присутствуют. `result.boxes` равен
`None`, потому что отдельных детекций по экземплярам здесь нет. `conf` и `iou`
принимаются ради совместимости API, но на вывод не влияют: модель возвращает
по одному классу на пиксель, а не детекции экземпляров, которые нужно
фильтровать или чистить от дублей. Про источники, стриминг и обработку
результатов см. [предсказание](/docs/predict).

## Варианты

Шесть размеров, от b0 до b5: на каждом шаге энкодер Mix Transformer становится
шире и глубже, а устройство декодирующей головы из одних MLP остаётся тем же.

<checkpoint-table />

## Обучение

По умолчанию `train()` дообучает опубликованный чекпойнт. Если же не передавать
`model_path` в `LibreSegformer(...)`, модель собирается со случайно
инициализированными энкодером и головой и обучается с нуля — это единственный
способ получить веса, на которые не распространяется некоммерческое
ограничение предобученных чекпойнтов (см. [Лицензирование](#licensing)).

<code-tabs name="train" />

Если ничего не менять, обучение идёт по рецепту ADE20K из статьи про
SegFormer: AdamW с базовой скоростью обучения для бэкбона и в 10 раз большей
для декодирующей головы, weight decay везде, кроме LayerNorm и позиционной
свёртки Mix-FFN, и линейно затухающий планировщик с прогревом. Сходимость для
больших размеров, от b3 до b5, не проверялась от начала до конца.

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`: mIoU и pixel accuracy,
измеренные на любом датасете в том формате, в котором вы обучали.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. [Экспорт](/docs/export) перечисляет аргументы, которые
принимает каждый формат.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

Энкодер и декодирующая голова LibreSegformer — это PyTorch-порт реализации
SegFormer из Hugging Face Transformers под Apache-2.0, а не из
NVlabs/SegFormer: исходный репозиторий NVIDIA не читали и не копировали, он
упомянут здесь только ради указания авторства статьи. Некоммерческое
ограничение NVIDIA распространяется только на предобученные чекпойнты выше;
сама архитектура и собственный код LibreYOLO остаются под MIT.

</provenance-box>

## Цитирование

<citation-block />
