---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: детекция объектов в LibreYOLO'
description: >-
  Запуск CenterNet (Objects as Points) в LibreYOLO с бэкбонами ResDCN-18 и
  DLA-34. Предсказание, валидация и экспорт в ONNX под лицензией MIT. Пути
  обучения нет.
lead: >-
  CenterNet моделирует объект как центральную точку его ограничивающей рамки и
  регрессирует все остальные свойства из пика тепловой карты, поэтому ему не
  нужны ни якоря, ни шаг non-maximum-suppression. В LibreYOLO он поставляется
  как детектор только для инференса.
keywords:
  - CenterNet
  - Objects as Points
  - детекция объектов python
  - детектор без якорей anchor-free
  - детекция по ключевым точкам
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCenterNetresdcn18.pt")


        # Для экспорта в ONNX нужен opset 16 или новее: стадия апсемплинга

        # на деформируемых свёртках разворачивается в GridSample, который
        появился в opset 16.

        model.export(format="onnx", opset=18)

        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный
        артефакт

        # загружается как любой чекпойнт и возвращает тот же объект Results.

        model = LibreYOLO("LibreCenterNetresdcn18.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Установка

CenterNet не нужны опциональные extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращают все семейства, поэтому
замена на другой детектор — правка в одну строку. `conf` и `max_det` фильтруют
отранжированные пики тепловой карты; `iou` принимается ради совместимости API,
но ни на что не влияет, потому что декодированию top-k пиков в CenterNet не
нужен шаг подавления по IoU рамок. Про источники, стриминг и обработку
результатов — в разделе [предсказание](/docs/predict).

## Варианты

Два бэкбона. `resdcn18` соединяет ствол ResNet-18 с апсемплингом на
деформируемых свёртках; `dla34` соединяет ствол DLA-34 с апсемплингом через
итеративную глубокую агрегацию. Оба питают одни и те же три плотные головы
(тепловая карта, ширина/высота, смещение) и один и тот же входной холст.

## Валидация

`val()` возвращает словарь с ключами `metrics/`, покрывающими точность,
полноту, mAP 50 и mAP 50-95, измеренные на любом датасете в том формате, в
котором вы обучали модель.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Для экспорта в ONNX нужен opset 16 или новее: стадия апсемплинга на
деформируемых свёртках в обоих бэкбонах разворачивается в ONNX-оператор
`GridSample`, который появился в opset 16. Запрос более старого opset вызывает
ошибку ещё до начала трассировки.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

Граф ResDCN-18 дополнительно ссылается на human-pose-estimation.pytorch от
Microsoft под лицензией MIT, а граф DLA-34 — на реализацию DLA от Fisher Yu под
BSD-3-Clause. LibreYOLO не поставляет в своём составе оригинальное расширение
DCNv2 из исходного проекта; нативное выполнение вместо него запускает
`deform_conv2d` из torchvision под BSD-3-Clause, а переносимая реализация
только для экспорта написана для LibreYOLO отдельно.

</provenance-box>

## Цитирование

<citation-block />
