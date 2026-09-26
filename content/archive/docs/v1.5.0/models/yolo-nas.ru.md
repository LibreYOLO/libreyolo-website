---
title: YOLO-NAS
families:
  - yolonas
seo_title: 'YOLO-NAS: предсказание, обучение и экспорт в LibreYOLO'
description: >-
  Использование YOLO-NAS в LibreYOLO для детекции и оценки позы. Веса Deci.AI
  проприетарны и разрешены только для некоммерческого использования, и LibreYOLO
  их не публикует.
lead: >-
  Свёрточный детектор, бэкбон и neck которого получены поиском архитектуры в
  Deci.AI и собраны из блоков RepVGG, рассчитанных на квантизацию. Веса
  принадлежат Deci.AI, лицензированы только для некоммерческого использования, и
  LibreYOLO их не публикует.
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - детекция объектов python
  - оценка позы
  - детектор с поддержкой квантизации
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Имя, которого ещё нет на диске, скачивается с CDN компании Deci.

        # Перед скачиванием печатаются условия лицензии Deci; забирая файл, вы
        их принимаете.

        model = LibreYOLO("LibreYOLONASs.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Оценка позы
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суффикс -pose выбирает голову позы и её собственный набор весов.
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Обучение с нуля
      language: python
      code: >
        from libreyolo import LibreYOLONAS


        # Ни один чекпойнт Deci не задействован: модель стартует со случайных
        весов,

        # поэтому результат запуска получен только из ваших данных.

        model = LibreYOLONAS(None, size="s")

        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Валидация на COCO
      language: bash
      code: >
        # В поставляемом YAML для COCO есть встроенный скрипт скачивания,
        поэтому

        # нужно явное разрешение, если датасет ещё не лежит локально.

        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает путь по расширению файла, поэтому экспортированный
        артефакт

        # загружается как любой чекпойнт и возвращает тот же объект Results.

        model = LibreYOLO("LibreYOLONASs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## Установка

YOLO-NAS не требует ничего сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Имя чекпойнта, которого ещё нет на диске, скачивается с публичного CDN компании
Deci, а не из организации LibreYOLO: она не хранит эти веса. Перед началом
передачи библиотека один раз за процесс печатает условия лицензии Deci, а перед
открытием скачанного файла его SHA-256 сверяется с зафиксированным значением.
Что именно разрешают эти условия, описано в разделе
[лицензирование](#licensing).

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена на другой детектор занимает одну строку. `conf` задаёт порог
уверенности, а `iou` — порог NMS. Про источники, стриминг и обработку
результатов см. [предсказание](/docs/predict).

## Варианты

Детекция и оценка позы — одна и та же архитектура под разными головами, и они
принимают одни и те же аргументы. Размеры в таблице ниже относятся к детекции;
для оценки позы опубликованы они же и ещё один, меньший. Голова позы
предсказывает набор ключевых точек COCO.

<benchmark-table task="detect" />

<va-embed />

## Обучение

<code-tabs name="train" />

`epochs`, `lr0` и `amp` определяются под задачу, если вы их не задали, поэтому
запуск для оценки позы стартует с других значений по умолчанию, чем запуск для
детекции. Оптимизатор по умолчанию — AdamW. Число классов берётся из YAML
датасета, и перед первой эпохой голова пересобирается под него; на голове позы
число ключевых точек обрабатывается так же, поэтому чекпойнт позы COCO можно
дообучить под скелет другого размера.

Дообучение стартует с весов Deci — именно это и покрывает лицензия Deci.
Обучение модели со случайной инициализацией вообще не задействует чекпойнт
Deci, и это третий пример кода выше.

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность,
полноту, mAP 50 и mAP 50-95, измеренные на любом датасете в том же формате, в
котором вы обучали модель.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запуск графа в голой среде выполнения, без установленной
библиотеки LibreYOLO, тоже поддерживается, но тогда предобработку и
постобработку вам придётся писать самостоятельно. Каждый формат ставится через
свой extra и принимает несколько собственных аргументов. И то и другое описано
на странице этого формата.

Экспорт — это ещё одна копия тех же весов в другом контейнере. Экспорт
чекпойнта Deci не меняет ни того, откуда взялись веса, ни лицензии, которая на
них распространяется.

<code-tabs name="export" />

## Чекпойнты

Перечислять нечего. Лицензия Deci запрещает распространение, поэтому
организация LibreYOLO не публикует веса YOLO-NAS и скачивание уходит в другое
место: имя вида `LibreYOLONAS<size>.pt` или `LibreYOLONAS<size>-pose.pt` для
оценки позы отображается на соответствующий объект на публичном CDN компании
Deci.

Так скачиваются только те чекпойнты, чей SHA-256 зафиксирован в библиотеке. Всё
остальное завершается отказом, а не открытием непроверенного стороннего
pickle-файла, и такой файл придётся скачать вручную и указать путь к нему.
Файл, уже лежащий на диске, загружается по своему пути, без скачивания и без
проверки контрольной суммы. Это касается и файла `.pth` от Deci под его исходным именем:
загрузчик его распознаёт.

## Лицензирование

<provenance-box>

LibreYOLO не размещает эти веса у себя и не делает их зеркал: в организации
LibreYOLO на Hugging Face для этого семейства нет ничего. Каждое автоматическое
скачивание вместо этого идёт на публичный CDN компании Deci, один раз за
процесс печатает условия Deci перед началом и сверяется с зафиксированным
SHA-256 до открытия файла.

Альтернатива — обучение модели со случайной инициализацией. Архитектура в
апстриме под Apache-2.0, а здесь под MIT, поэтому модель, обученная так на
ваших данных, не происходит ни от одного чекпойнта Deci.

</provenance-box>

## Цитирование

YOLO-NAS вышел без статьи. Запись ниже — та, о которой просят его авторы; она
относится к SuperGradients, библиотеке, в составе которой он вышел.

<citation-block />
