---
title: SwinIR
families:
  - swinir
seo_title: 'SwinIR: увеличение разрешения изображений в 4x в LibreYOLO'
description: >-
  Используйте SwinIR в LibreYOLO для увеличения разрешения изображений в 4x.
  Установка, предсказание, валидация и экспорт лёгкого, среднего и большого
  чекпойнтов.
lead: >-
  Сеть на основе Swin Transformer для восстановления изображений. В LibreYOLO
  есть инференс и валидация для её чекпойнтов с увеличением в 4x: официального
  лёгкого генератора, а также среднего и большого генераторов для реальных
  изображений.
keywords:
  - SwinIR
  - Swin Transformer
  - увеличить разрешение изображения нейросетью
  - восстановление изображений python
  - апскейл фото 4x
  - super resolution python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Потайловый инференс для больших изображений
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tile разбивает прямой проход на перекрывающиеся тайлы и сглаживает
        # стыки между ними; tile_pad — это поля вокруг каждого тайла, которые
        # потом обрезаются обратно. Оба параметра есть только как именованные
        # аргументы в Python, а не как флаги CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")

        # если imgsz не указан, берётся небольшой внутренний размер патча, а
        # не ваше рабочее разрешение, поэтому передавайте тот размер, который
        # реально подаётся модели при развёртывании.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreSwinIRm-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## Установка

SwinIR не требует установки дополнительных extra-пакетов. Всё, что он
импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

В результате восстановления нет рамок; `result.restored` — плотное
RGB-изображение `(H, W, 3)` типа uint8 на холсте, который по каждому измерению
в 4 раза больше входного. `save=True` записывает это изображение напрямую, а не
картинку с аннотациями. Вход не масштабируется, а дополняется до размера,
кратного 8, поэтому предсказание выполняется в собственном разрешении фотографии;
источник, который не помещается в память, можно разбить с помощью `tile` и
`tile_pad` — они же сглаживают стыки тайлов в итоговом изображении. Об
источниках, стриминге и обработке результатов — в разделе
[предсказание](/docs/predict).

## Варианты

Три размера, все с фиксированным увеличением в 4x. `s` — это официальный лёгкий
генератор с четырьмя стадиями residual Swin Transformer block (RSTB) и прямым
повышением разрешения через pixel shuffle. `m` и `l` — средний и большой
генераторы для реальных изображений, с шестью и девятью стадиями RSTB и
апсемплером на основе интерполяции по ближайшему соседу плюс свёртка,
рассчитанным на реальные искажения, а не только на бикубическое уменьшение.

## Валидация

`val()` считает PSNR и SSIM между восстановленным изображением и чистым
эталоном; обе метрики вычисляются в RGB на исходном холсте, без обрезки краёв и
без масштабирования. SSIM использует гауссово окно 11x11 с сигмой 1.5 и
усредняется по трём цветовым каналам.

<code-tabs name="val" />

Аргумент датасета — это YAML, который связывает каталог повреждённых входных
изображений с каталогом чистых эталонных изображений того же разрешения; точные
ключи описаны в разделе [форматы датасетов](/docs/reference/dataset-formats).

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. ExecuTorch и все форматы, отмеченные в матрице как
недоступные, для этого семейства не поддерживаются; ONNX, TorchScript,
TensorRT, OpenVINO и TFLite — поддерживаются. В разделе
[экспорт](/docs/export) перечислены аргументы, которые принимает каждый формат,
и дополнительные, которые добавляют некоторые из них.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
