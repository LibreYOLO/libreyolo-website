---
title: BiRefNet
families:
  - birefnet
seo_title: 'BiRefNet: удаление фона и маттинг в LibreYOLO'
description: >-
  Используйте BiRefNet в LibreYOLO для удаления фона и дихотомической
  сегментации изображений. Установка, предсказание, валидация и экспорт общего
  чекпойнта.
lead: >-
  Сеть с двусторонними опорными признаками (bilateral reference), которая
  предсказывает мягкую альфа-маску (alpha matte), отделяющую объект от фона. В
  LibreYOLO есть инференс и валидация для задачи matte из BiRefNet.
keywords:
  - BiRefNet
  - удаление фона с фото
  - убрать фон python
  - дихотомическая сегментация изображений
  - alpha matte
  - маттинг изображений
  - вырезать объект с прозрачным фоном
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Вырезание объекта
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: исходный RGB плюс альфа-маска как альфа-канал.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Вместо YAML датасета подойдёт и каталог с images/ и автоматически
        # найденным каталогом альфа-масок (mattes/, matte/, gt/, masks/,
        # mask/ или alpha/).
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## Установка

BiRefNet не требует установки дополнительных extra-пакетов. Всё, что он
импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

В результате задачи `matte` нет рамок; `result.matte` — плотный массив
`(H, W)` типа float32 со значениями в `[0, 1]`, где 1 — полностью передний
план, а 0 — полностью фон. В отличие от бинарной маски, мягкая альфа-маска
сохраняет сглаженные детали на краях: например, волосы и шерсть.
`result.cutout()` собирает исходное изображение с этим альфа-каналом в
RGBA-массив, а `result.save(path)` (или `save=True` в вызове предсказания)
сразу записывает его в PNG с прозрачным фоном. Модель работает на
фиксированном родном холсте 1024x1024; другое разрешение не поддерживается,
потому что к нему привязаны таблицы относительных позиций бэкбона Swin, и при
несовпадении они интерполируются некорректно, а не вызывают ошибку. Об
источниках, стриминге и обработке результатов — в разделе
[предсказание](/docs/predict).

## Варианты

Опубликован один чекпойнт, `l`, — модель BiRefNet-general уровня Swin-L и
вариант по умолчанию для качества в исходном проекте. Код семейства
поддерживает и облегчённый уровень Swin-T, `t`, но конвертация в LibreYOLO для
него пока не опубликована.

## Валидация

`val()` считает две метрики по папке с парами изображение/альфа-маска; обе
лежат в `[0, 1]` и не зависят от разрешения: MAE — средняя абсолютная ошибка
относительно эталонной альфа-маски (ground truth), чем меньше, тем лучше, и
S-measure (Fan et al., ICCV 2017) — структурное сходство, которое учитывает
сохранение формы объекта и отверстий в нём, чего попиксельный MAE сам по себе
не замечает (чем больше, тем лучше). Валидация идёт через собственный
`predict` модели, поэтому использует ровно ту предобработку, что принята в
семействе.

<code-tabs name="val" />

Валидация работает только на инференсе; дообучение (fine-tuning) —
задокументированное продолжение, а не готовая возможность (точное ограничение
по разрешению, которое унаследует любой будущий trainer, описано в разделе
«Предсказание»).

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` ведёт себя как чекпойнт и возвращает тот же
`Results`. Проверенный путь — TorchScript; конвертация в ONNX работает, но
такую же планку по совпадению результатов пока не прошла. В разделе
[экспорт](/docs/export) перечислены аргументы, которые принимает каждый
формат, и дополнительные, которые добавляют некоторые из них.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
