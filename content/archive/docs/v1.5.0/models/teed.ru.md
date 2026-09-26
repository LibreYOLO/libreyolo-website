---
title: TEED
families:
  - teed
seo_title: 'TEED: выделение границ со своим чекпойнтом'
description: >-
  Используйте TEED в LibreYOLO для плотного предсказания вероятности границ.
  Конвертируйте лицензированный чекпойнт, а затем запускайте предсказание,
  валидацию и экспорт.
lead: >-
  TEED (Tiny and Efficient Edge Detector) — небольшая свёрточная сеть, которая
  по одному RGB-изображению предсказывает плотную карту вероятности границ.
  LibreYOLO использует её архитектуру только для выделения границ; чекпойнт с
  библиотекой не поставляется.
keywords:
  - TEED
  - Tiny and Efficient Edge Detector
  - выделение границ python
  - лёгкий детектор границ
  - карта границ нейросеть
  - BIPED
  - плотное предсказание
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32 в [0, 1]
        print(edges.binary(0.5).sum())  # число пикселей границ после порога
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        metrics = model.val(data="my-dataset.yaml", imgsz=352)


        print(metrics["metrics/ODS"])   # F-мера при оптимальном пороге для
        датасета

        print(metrics["metrics/OIS"])   # F-мера при оптимальном пороге для
        изображения
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352

        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## Установка

TEED не требует установки дополнительных extra-пакетов. Всё, что он
импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

LibreYOLO не поставляет чекпойнт TEED. Официально выпущенные веса обучены
на BIPED, а опубликованные условия использования этого датасета ограничивают
его некоммерческими целями, поэтому LibreYOLO не размещает у себя их копию.
Конвертируйте чекпойнт, на который у вас есть лицензия, скриптом
`weights/convert_teed_weights.py`: он сверяет ключи тензоров с архитектурой
среды выполнения и только после этого записывает файл, который LibreYOLO
загружает напрямую:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

В `result.edges` лежит результат — массив `(H, W)` типа float32 со значениями
в `[0, 1]`, а `.binary(threshold)` возвращает булеву маску границ. Рамок здесь
нет, поэтому `conf`, `iou` и `max_det` ни на что не влияют. Об источниках,
стриминге и обработке результатов — в разделе [предсказание](/docs/predict).

## Варианты

В LibreYOLO у TEED один размер. Бенчмарк-стенд LibreYOLO это семейство не
измерял, поэтому опубликованных цифр для сравнения нет.

## Валидация

`val()` считает F-меры ODS и OIS в стиле BSDS по парному датасету границ:
изображения и лежащие рядом одноимённые карты границ, плюс необязательная
маска валидности, чтобы пиксели дополнения никогда не учитывались. `imgsz`
должен делиться на суммарный шаг понижения разрешения сети, и LibreYOLO
выдаёт понятную ошибку, если это не так.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспорт границ работает по контракту среды выполнения с фиксированным
разрешением и batch=1: `dynamic` и `batch`, отличный от 1, отклоняются, а
экспортированный граф выдаёт одну объединённую карту вероятности.
Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` ведёт себя как чекпойнт и возвращает тот же
`Results`.

<code-tabs name="export" />

## Лицензирование

<provenance-box>

LibreYOLO не публикует чекпойнт TEED. В организации LibreYOLO не размещено
ни одной копии; вместо этого конвертируйте скриптом
`weights/convert_teed_weights.py` чекпойнт, на который у вас есть лицензия.

</provenance-box>

## Цитирование

<citation-block />
