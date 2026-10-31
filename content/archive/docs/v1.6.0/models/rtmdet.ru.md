---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet в LibreYOLO: предсказание, обучение и экспорт'
description: >-
  Использование RTMDet в LibreYOLO для детекции объектов и сегментации
  экземпляров RTMDet-Ins. Установка, предсказание, обучение, валидация и экспорт
  под лицензией Apache-2.0.
lead: >-
  RTMDet — одностадийный детектор, который предсказывает из одного точечного
  приора на каждую позицию сетки, без якорей, через голову, свёртки которой
  общие для всех уровней признаков. LibreYOLO поддерживает его для детекции и
  сегментации экземпляров RTMDet-Ins.
keywords:
  - RTMDet
  - детекция объектов python
  - сегментация экземпляров
  - RTMDet-Ins
  - детекция без якорей anchor-free
  - обучить rtmdet на своём датасете
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Сегментация экземпляров
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суффикс -seg в имени файла выбирает голову масок RTMDet-Ins,
        # поэтому аргумент task здесь не нужен.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Сегментация экземпляров
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # маски
        print(metrics["metrics/mAP50-95(B)"])   # рамки
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreRTMDets.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Установка

RTMDet не требует никаких extra сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена на другой детектор занимает одну строку. Имя файла с `-seg` само по себе
разрешается в задачу RTMDet-Ins, и тогда `result.masks` несёт маски экземпляров
рядом с рамками. `conf` задаёт порог уверенности, а `iou` — порог NMS. Про
источники, стриминг и обработку результатов см. [предсказание](/docs/predict).

## Варианты

Пять размеров, от `t` до `x`, используют одну архитектуру на общем входном
разрешении. Таблицы бенчмарков у этого семейства здесь нет: размеры сравнивайте
по размеру файла чекпойнта в таблице ниже.

## Обучение

<code-tabs name="train" />

Детекция обучается через `train()`. Компоненты QualityFocalLoss, GIoU и
DynamicSoftLabelAssigner портированы из апстрима mmdetection, прямой проход и
экспорт в ONNX побитово совпадают с ним, а постобработка сходится с выводом
mmdet в пределах 0.001 mAP на подмножествах val2017.

Что не проверялось, согласно докстрингу самого `train()`: сходимость
дообучения на маленьких датасетах, соответствие статье при обучении с нуля,
поведение на multi-GPU, пропускная способность кэшированных Mosaic и MixUp,
строгое апстримовое переключение двухстадийного пайплайна и попараметрические
переопределения weight decay, которые обнуляют его для параметров
нормализации и смещений.

У RTMDet-Ins нет пути обучения. Вызов `train()` на `-seg`-чекпойнте или с
`task="segment"` выбрасывает `NotImplementedError`; сегментация экземпляров
поддерживает только инференс и валидацию.

`train()` также принимает аргумент `pretrained`, но внутри метода его значение
никогда не читается: обучение всегда продолжается с тех весов, с которыми была
создана модель, поэтому `pretrained=False` не переинициализирует сеть.

Если ничего не менять, обучение идёт 300 эпох с AdamW при `lr0=0.004` и
`weight_decay=0.05`, с прогревом длиной в одну эпоху по косинусному расписанию
и с отключёнными Mosaic и MixUp на последних 20 эпохах.

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность,
полноту, mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, на
котором вы обучались.

<code-tabs name="val" />

Для `-seg`-чекпойнта обычный ключ `metrics/mAP50-95` содержит оценку по маскам,
и тот же запуск дополнительно сообщает рамки под `(B)` и маски под `(M)`, так
что оба значения доступны за один проход.

## Экспорт

<export-matrix />

Детекция экспортируется в большинство форматов; сегментация экземпляров сейчас
не экспортируется ни в один из них; матрица выше отражает это разделение.
Экспортированный артефакт детекции загружается обратно через `LibreYOLO()` по
расширению файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и
возвращает тот же `Results`. Запуск графа в голой среде выполнения, без
установленной LibreYOLO, тоже поддерживается, но тогда предобработку и
постобработку вам придётся писать самостоятельно.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
