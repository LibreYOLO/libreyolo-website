---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2: предсказание, валидация и экспорт нормалей поверхности'
description: >-
  Используйте MoGe-2 в LibreYOLO для плотного предсказания нормалей поверхности.
  Установка, предсказание, валидация и экспорт официальных чекпойнтов ViT-S,
  ViT-B и ViT-L.
lead: >-
  MoGe-2 — монокулярная геометрическая модель, которая за один прямой проход
  предсказывает плотное поле нормалей поверхности по одному RGB-изображению. В
  LibreYOLO она доступна только для оценки нормалей — через официальные
  чекпойнты ViT-S, ViT-B и ViT-L.
keywords:
  - MoGe-2
  - MoGe 2
  - оценка нормалей поверхности
  - карта нормалей из фото
  - монокулярная геометрия
  - surface normal estimation
  - плотное предсказание
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # (H, W, 3) float32, единичные векторы
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # градусы
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # процент пикселей
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## Установка

MoGe-2 не требует установки дополнительных extra-пакетов. Всё, что он импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются автоматически при первом запуске: LibreYOLO берёт нужный
размер напрямую из официальных чекпойнтов и кэширует его локально.

<code-tabs name="predict" />

MoGe-2 возвращает плотное поле, а не набор детекций, поэтому `result.boxes`
пуст, а `conf`, `iou` и `max_det` ни на что не влияют. Результат лежит в
`result.normal_map`: массив `(H, W, 3)` из единичных векторов в системе
координат камеры OpenCV, где `+x` — вправо, `+y` — вниз, `+z` — вглубь сцены, а
обращённая к камере поверхность даёт `(0, 0, -1)`. Предсказание по списку
изображений выполняет по одному прямому проходу на изображение; быстрого пути со
сборкой батча у этого семейства нет. Об источниках, стриминге и обработке
результатов — в разделе [предсказание](/docs/predict).

## Варианты

Три размера энкодера поставляются отдельными чекпойнтами: ViT-S, ViT-B и ViT-L,
все с одним и тем же входным разрешением. Бенчмарк-стенд LibreYOLO это семейство
не измерял, поэтому опубликованных чисел по точности, по которым их можно было бы
сравнить, нет; выбирайте размер под свой бюджет вычислений.

## Валидация

`val()` измеряет угловую ошибку на парном датасете карт нормалей: изображения
рядом с 16-битными PNG нормалей с тем же именем файла и необязательная маска
валидности, чтобы пиксели заполнения и некорректные пиксели никогда не
учитывались. Возвращаются средняя и медианная угловая ошибка в градусах, а также
процент пикселей, попавших в 11.25, 22.5 и 30 градусов.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспорт нормалей работает по контракту среды выполнения с фиксированным
разрешением и батчем 1: `dynamic` и `batch`, отличный от 1, отклоняются, а
`imgsz` должен нацело делиться на размер патча ViT-энкодера — LibreYOLO
проверяет это до старта запуска. Экспортированный артефакт загружается обратно
через `LibreYOLO()` по суффиксу файла, поэтому файл `.onnx` ведёт себя как
чекпойнт и возвращает тот же `Results`.

<code-tabs name="export" />

## Лицензирование

<provenance-box>

LibreYOLO не копирует эти чекпойнты в свою организацию.
`LibreYOLO("LibreMoGe2s-normal.pt")` скачивает нужный размер напрямую из
официальных репозиториев на Hugging Face на зафиксированной ревизии и перед
использованием сверяет файл с записанной контрольной суммой SHA-256.

</provenance-box>

## Цитирование

<citation-block />
