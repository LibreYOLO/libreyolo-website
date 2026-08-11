---
title: YOLOv7
families:
  - yolo7
seo_title: 'YOLOv7 в LibreYOLO: предсказание, обучение и экспорт под MIT'
description: >-
  Запуск YOLOv7 в LibreYOLO для детекции объектов: установка, предсказание,
  обучение, валидация и экспорт, код и веса под лицензией MIT.
lead: >-
  YOLOv7 — одностадийный детектор на якорях, голова которого перед финальной
  свёрткой добавляет обучаемые сдвиги неявного знания (implicit knowledge).
  LibreYOLO поддерживает его единственный опубликованный размер для детекции.
keywords:
  - YOLOv7
  - yolov7 python
  - детекция объектов python
  - обучить yolov7 на своём датасете
  - детектор на якорях
  - детекция объектов в реальном времени
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Тёплый старт новой модели
      language: python
      code: |
        from libreyolo import LibreYOLO7

        # pretrained=True всегда загружает опубликованный чекпойнт
        # LibreYOLO7b.pt, независимо от того, с чем был создан этот экземпляр.
        # Если создавать класс напрямую, а не через LibreYOLO(), то веса не
        # загружаются вообще.
        model = LibreYOLO7(None, size="b")
        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreYOLO7b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## Установка

Для YOLOv7 не нужны дополнительные зависимости сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращают все семейства, поэтому
замена на другой детектор — правка в одну строку. `conf` задаёт порог
уверенности, а `iou` — порог NMS, который применяется после декодирования головы
на якорях. Про источники, стриминг и обработку результатов — в разделе
[предсказание](/docs/predict).

## Варианты

LibreYOLO поставляет один размер — `b`. В апстриме опубликована единственная
модель YOLOv7, поэтому выбирать размер не из чего.

## Обучение

<code-tabs name="train" />

`pretrained` здесь действительно читается, в отличие от одноимённой заглушки у
некоторых других семейств: передайте `True`, чтобы сделать тёплый старт с
опубликованного чекпойнта `LibreYOLO7b.pt` (скачивается автоматически), либо
путь или имя, если нужен любой другой. Этот опубликованный чекпойнт — COCO на 80
классов, поэтому если запросить его для модели, уже перестроенной под другое
число классов, она сначала перестраивается обратно под 80, чекпойнт загружается,
а затем каждый совпадающий по форме тензор переносится в голову с целевым числом
классов — как только прочитано число классов датасета. `resume=True` нельзя
сочетать с `pretrained`. Если оставить значение по умолчанию `None`, обучение
продолжается с тех весов, с которыми была создана модель, или со случайной
инициализации, если ничего не загружалось.

Если больше ничего не задавать, обучение идёт 300 эпох с `lr0=0.01`, SGD с
моментом 0.937, прогревом на 3 эпохи, тем же назначением SimOTA и той же
финальной фазой из 15 эпох без аугментаций, что и в YOLOX, адаптированными под
голову на якорях. Единственное отличие: YOLOX добавляет в эти финальные эпохи
уточнение регрессии рамок по L1, а v7 его пропускает, потому что в функции
потерь SimOTA у v7 нет L1-ветки по сырым сдвигам, которую можно было бы
уточнять.

Про датасеты, аугментации, несколько GPU и логгеры — в разделе
[обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность, полноту,
mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в котором вы
обучали модель.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запуск графа в чистой среде выполнения, без установленного
LibreYOLO, тоже поддерживается, но тогда предобработку и постобработку придётся
писать самостоятельно.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
