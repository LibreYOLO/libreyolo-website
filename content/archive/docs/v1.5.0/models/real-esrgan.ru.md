---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: увеличение разрешения изображений в LibreYOLO'
description: >-
  Используйте Real-ESRGAN в LibreYOLO для практичного увеличения разрешения
  изображений в 4x, 2x и в быстром режиме 4x. Установка, предсказание, валидация
  и экспорт.
lead: >-
  Практичный апскейлер для слепого увеличения разрешения, обученный на
  синтетических искажениях, а не только на бикубическом уменьшении. В LibreYOLO
  есть инференс и валидация для чекпойнтов 4x, 2x и быстрого 4x.
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - увеличить разрешение изображения нейросетью
  - апскейл фото python
  - улучшить качество фото нейросетью
  - super resolution python
  - восстановление старых фотографий
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Потайловый инференс для больших изображений
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tile разбивает прямой проход на перекрывающиеся тайлы и сглаживает
        # стыки между ними; tile_pad — это поля вокруг каждого тайла, которые
        # потом обрезаются обратно. Оба параметра есть только как именованные
        # аргументы в Python, а не как флаги CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # если imgsz не указан, берётся небольшой внутренний размер патча, а
        # не ваше рабочее разрешение, поэтому передавайте тот размер, который
        # реально подаётся модели при развёртывании.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Установка

Real-ESRGAN не требует установки дополнительных extra-пакетов. Всё, что он
импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

В результате восстановления нет рамок; `result.restored` — плотное
RGB-изображение `(H, W, 3)` типа uint8 на холсте, который по каждому измерению
в `Results.restore_scale` раз больше входного. `save=True` записывает это
изображение напрямую, а не картинку с аннотациями. Вход преобразуется в RGB,
альфа-канал, если он есть, отбрасывается. Источник, который не помещается в
память, можно разбить с помощью `tile` и `tile_pad` — они же сглаживают стыки
тайлов в итоговом изображении. Об источниках, стриминге и обработке
результатов — в разделе [предсказание](/docs/predict).

## Варианты

Три чекпойнта, названные по коэффициенту увеличения. `x4` — это RRDBNet
(`RealESRGAN_x4plus`), 23 блока residual-in-residual dense, вариант по
умолчанию для качества при 4x. `x2` — та же архитектура RRDBNet при 2x. `x4t` —
это SRVGGNetCompact (`realesr-general-x4v3`), меньший и более быстрый
генератор, рассчитанный на видео и работу с меньшей задержкой при 4x. В
исходном проекте у универсальной модели есть ещё парная сеть силы
шумоподавления, которая подмешивается на инференсе; в этом порте такой
регулировки нет — он запускает базовый генератор `x4t`.

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
тот же `Results`. В разделе [экспорт](/docs/export) перечислены аргументы,
которые принимает каждый формат, и дополнительные, которые добавляют некоторые
из них.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
