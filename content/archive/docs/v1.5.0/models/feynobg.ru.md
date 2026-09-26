---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: удаление фона в LibreYOLO'
description: >-
  Используйте FeyNobg в LibreYOLO для удаления фона и альфа-маттинга — это
  углублённый вариант BiRefNet от Feyn Inc. Установка, предсказание и валидация.
lead: >-
  Модель для удаления фона от Feyn Inc., которая углубляет архитектуру BiRefNet
  и переобучает её. В LibreYOLO есть инференс и валидация для задачи matte из
  FeyNobg.
keywords:
  - FeyNobg
  - удаление фона с фото
  - убрать фон python
  - дихотомическая сегментация изображений
  - alpha matte
  - маттинг изображений
  - вырезать объект с прозрачным фоном
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Вырезание объекта
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: исходный RGB плюс альфа-маска как альфа-канал.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFeyNobgl-matte.pt")

        # Вместо YAML датасета подойдёт и каталог с images/ и автоматически
        # найденным каталогом альфа-масок (mattes/, matte/, gt/, masks/,
        # mask/ или alpha/).
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Установка

FeyNobg не требует установки дополнительных extra-пакетов. Всё, что он
импортирует, входит в базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Чекпойнт скачивается из организации LibreYOLO на Hugging Face при первом
запуске и кэшируется локально, как и в любом другом семействе, хотя в таблице
чекпойнтов на этой странице он пока не указан.

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

Опубликован один размер, `l`, — бэкбон уровня Swin-L. FeyNobg берёт
архитектуру BiRefNet и углубляет её третью стадию Swin с 18 до 24 блоков, а
затем переобучает, поэтому порт в LibreYOLO переиспользует прямой проход,
предобработку и контракт вывода с одним логитом из BiRefNet; предсказание,
валидация и работа с чекпойнтами устроены так же, как в семействе `birefnet`.

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

Валидация работает только на инференсе. В исходной библиотеке `nobg` есть код
обучения под Apache-2.0; дообучение (fine-tuning) сегодня означает обучение
там и конвертацию результата собственным скриптом конвертации LibreYOLO, а не
вызов `train()` в этом семействе — он вызывает ошибку, а не запускает
частично готовый trainer.

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
