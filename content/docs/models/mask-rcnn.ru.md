---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: 'Mask R-CNN в LibreYOLO: предсказание, валидация и экспорт'
description: >-
  Запуск Mask R-CNN в LibreYOLO для детекции объектов и сегментации экземпляров.
  Установка, предсказание, валидация и экспорт порта из torchvision под
  лицензией BSD-3-Clause.
lead: >-
  Mask R-CNN добавляет к Faster R-CNN ветку масок по регионам: для каждой
  найденной рамки она предсказывает маску сегментации. LibreYOLO портирует
  реализацию из torchvision для детекции и сегментации экземпляров.
keywords:
  - Mask R-CNN
  - сегментация экземпляров python
  - детекция объектов python
  - Faster R-CNN
  - torchvision
  - двухстадийный детектор
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Только рамки
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" пропускает голову масок и возвращает рамки из того же
        # чекпойнта, без масок в результате.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # маски
        print(metrics["metrics/mAP50-95(B)"])   # рамки
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает загрузчик по расширению файла, поэтому
        экспортированный

        # артефакт загружается как любой чекпойнт и возвращает тот же объект
        Results.

        model = LibreYOLO("LibreMaskRCNNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## Установка

Mask R-CNN не требует опциональных extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена одного детектора на другой сводится к одной строке. Если загрузить
чекпойнт без аргумента `task`, возвращаются маски экземпляров, потому что
сегментация — задача этого семейства по умолчанию; тогда `result.masks` несёт
их вместе с рамками. С `task="detect"` загружаются те же веса, но без головы
масок, и возвращаются только рамки. `conf` и `iou` задают пороги уверенности и
NMS; в отличие от детектора на запросах, Mask R-CNN сохраняет шаг NMS из
оригинальной реализации. Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

## Варианты

Один бэкбон: ResNet-50 с пирамидой признаков, через сборщик Mask R-CNN версии
v2 из torchvision. Опубликованный чекпойнт распространяется под лицензией
BSD-3-Clause и обслуживает обе задачи этого семейства, так что выбирать между
размерами не приходится.

## Валидация

`val()` возвращает словарь с ключами `metrics/`. Для задачи сегментации,
которая у этого чекпойнта стоит по умолчанию, обычный ключ
`metrics/mAP50-95` содержит оценку по маскам, а рамки тот же запуск отдаёт под
суффиксом `(B)`, так что оба значения доступны за один проход.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Mask R-CNN экспортируется только в ONNX и только с размером батча 1.
Экспортированный граф содержит внутри шаги изменения размера и наложения масок
из оригинальной реализации, поэтому LibreYOLO принудительно ставит
`dynamic=True` независимо от того, что передано, чтобы граф оставался
корректным для неквадратных источников. Экспортированный файл `.onnx`
загружается обратно через `LibreYOLO()` по расширению файла и возвращает тот же
`Results`.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства. Единственный чекпойнт ниже
указан в разделе detect, но тот же файл загружается и для сегментации: не
передавайте аргумент `task` — и он по умолчанию вернёт маски.

<checkpoint-table />

## Лицензирование

<provenance-box>

Mask R-CNN сделан как подкласс обёртки Faster R-CNN в LibreYOLO: у него тот же
исходник из torchvision и та же лицензия BSD-3-Clause, а сверху добавлены
предсказатель масок и голова масок RoI из того же портированного коммита.

</provenance-box>
