---
title: SSD
families: [ssd]
seo_title: "SSD (SSD300): детекция объектов в LibreYOLO"
description: "Запуск SSD300 в LibreYOLO: single-shot детектор на VGG16 для предсказания, валидации и экспорта в ONNX под лицензией BSD-3-Clause. Пути обучения нет."
lead: "SSD (Single Shot MultiBox Detector) предсказывает все рамки и оценки классов из плотной сетки опорных рамок за один прямой проход, без отдельной стадии генерации регионов-кандидатов. LibreYOLO поставляет чекпойнт SSD300 на базе VGG16 как детектор только для инференса."
keywords: [SSD, SSD300, Single Shot MultiBox Detector, детекция объектов python, ssd300 инференс, VGG16, детектор на якорях]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSSD300.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")

        # imgsz здесь опущен намеренно: SSD300 трассируется на родном холсте
        # своего чекпойнта, а любое другое значение вызовет ошибку до начала экспорта.
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный артефакт
        # загружается как любой чекпойнт и возвращает тот же объект Results.
        model = LibreYOLO("LibreSSD300.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Установка

SSD не нужны опциональные extra. Всё, что он импортирует, входит в базовую
установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращают все семейства, поэтому
замена на другой детектор — правка в одну строку. SSD декодирует свою сетку
опорных рамок с оценками по классам, а затем запускает non-maximum suppression,
поэтому `conf`, `iou` и `max_det` здесь реально влияют на результат — в отличие
от детекторов на запросах в этой библиотеке. Про источники, стриминг и обработку
результатов — в разделе [предсказание](/docs/predict).

## Варианты

В SSD один чекпойнт: сеть SSD300 на базе VGG16 на своём фиксированном родном
холсте. Выбора размера или масштаба в этом семействе нет; предсказание,
валидация и экспорт используют один и тот же граф.

Файл весов — `LibreSSD300.pt`: префикс семейства и следом единственный ключ
размера, `"300"`. За ним стоит класс `LibreSSD`, поэтому напрямую модель
создаётся как `LibreSSD(size="300")`, а не через класс, названный по имени
файла.

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность, полноту,
mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в котором вы
обучали модель.

<code-tabs name="val" />

## Экспорт

<export-matrix />

SSD экспортируется только в ONNX; все остальные форматы для этого семейства
сейчас закрыты. Экспорт всегда использует родной холст чекпойнта, а граф отдаёт
сырую упакованную голову SSD, а не выход со встроенным non-maximum suppression,
поэтому `nms=True` при экспорте не принимается. Собственные бэкенды LibreYOLO
выполняют шаг декодирования и подавления уже после обратной загрузки графа.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

Код SSD300 в LibreYOLO не портирован из Caffe-релиза самих авторов статьи; он
основан на реализации SSD300 из torchvision под BSD-3-Clause, и именно на этот
репозиторий ведёт ссылка выше как на исходный проект. Веса VGG16 в бэкбоне
восходят ещё дальше — к полностью свёрточной сокращённой сети VGGNet из
Оксфорда, выпущенной под CC BY 4.0 Karen Simonyan и Andrew Zisserman.

</provenance-box>

## Цитирование

<citation-block />
