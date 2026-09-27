---
title: DINOv2
families:
  - dinov2
seo_title: 'DINOv2 в LibreYOLO: семантика, классификация и эмбеддинги'
description: >-
  Используйте DINOv2 в LibreYOLO для семантической сегментации, классификации и
  эмбеддинга всего изображения на бэкбоне DINOv2-with-Registers. Везде
  Apache-2.0.
lead: >-
  DINOv2 — self-supervised vision transformer, который в Meta AI обучили без
  разметки, чтобы получать универсальные признаки изображений. LibreYOLO
  оборачивает его бэкбон DINOv2-with-Registers для трёх задач: семантической
  сегментации, классификации и эмбеддинга всего изображения.
keywords:
  - DINOv2
  - DINOv2 with registers
  - self-supervised обучение без разметки
  - vision transformer
  - семантическая сегментация python
  - эмбеддинги изображений
  - извлечение признаков изображения
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Семантическая сегментация
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # В этом семействе нет чекпойнта, размещённого LibreYOLO: здесь
        # скачивается бэкбон DINOv2-with-Registers-small под Apache-2.0 из
        # Hugging Face-организации Meta. Плотная голова инициализируется
        # случайно, пока её не обучить (см. «Обучение» ниже).
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        result = model(SAMPLE_IMAGE)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: Классификация
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= — число классов вашего датасета; линейная голова
        # инициализируется случайно, пока её не обучить.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: Эмбеддинг
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Обходит все головы задач: достаточно одного бэкбона, поэтому
        # никакого дообучения здесь не нужно.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), L2-нормированные
    - label: Эмбеддинг батча
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Удобная обёртка: вызывает predict() и складывает все строки в один
        # тензор (N, D).
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Семантическая сегментация
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Классификация
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Семантическая сегментация
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Классификация
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Семантическая сегментация
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Классификация
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Эмбеддинг
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный

        # артефакт загружается как обычный чекпойнт и возвращает тот же объект

        # Results. Имя файла экспорт берёт из задачи, здесь —
        LibreDINOv2s-sem.onnx.

        model = LibreYOLO("LibreDINOv2s-sem.onnx")

        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## Установка

LibreDINOv2 регистрируется, только когда установлен `transformers` — та же
необязательная зависимость, которая нужна RF-DETR для его бэкбона DINOv2,
поэтому нужен тот же extra.

```bash
pip install "libreyolo[rfdetr]"
```

## Предсказание

LibreYOLO не публикует чекпойнт LibreDINOv2. Вместо загрузки файла создайте
обёртку напрямую: с `model_path=None` (значение по умолчанию) при первом
запуске с Hugging Face скачивается бэкбон Meta
`facebook/dinov2-with-registers-small` под Apache-2.0. Аргумент `task=`
выбирает, что работает поверх него.

<code-tabs name="predict" />

`task="semantic"` и `task="classify"` добавляют поверх бэкбона плотную или
линейную голову; эта голова инициализируется случайно и становится полезной
только после обучения (см. [Обучение](#train)). `task="embed"` пропускает все
головы и возвращает финальный нормированный CLS-токен бэкбона как одну строку
на всё изображение в `result.embeddings`, поэтому обучение ему вообще не
нужно. `result.boxes` всегда `None`: ни одна из трёх задач не выдаёт детекции
по экземплярам. Про источники, стриминг и обработку результатов —
[предсказание](/docs/predict).

## Варианты

`size` выбирает ширину проектора в стиле RF-DETR, надстроенного над бэкбоном,
а не сам бэкбон: у всех размеров один и тот же энкодер DINOv2-S (small).
Семантическая сегментация работает на родной квадратной сетке патчей DINOv2;
классификация и эмбеддинг — на меньшем разрешении классификации, на котором
обучали linear probe.

## Обучение

`task="semantic"` и `task="classify"` обучаются оба; у `task="embed"` нет
зависящей от классов головы, которую нужно подгонять, и `train()` для него
вызывает `NotImplementedError`.

<code-tabs name="train" />

Основные именованные аргументы здесь — `batch_size` и `lr`, а не `batch` и
`lr0`, которые используют большинство других семейств; `batch` и `lr0`
по-прежнему принимаются и отображаются на них, но передача обоих вызывает
ошибку конфликта. `output_dir=` (по умолчанию `"runs/train"`) заменяет
`project=`/`name=` как основной способ задать место запуска, хотя передать
`project=`/`name=` напрямую по-прежнему можно. Про датасеты, аугментацию,
обучение на нескольких GPU и логгеры — [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`: mIoU и попиксельную accuracy
для `task="semantic"`, top-1 и top-5 accuracy для `task="classify"`. У
`task="embed"` нет эталонной разметки (ground truth), с которой можно
сравнивать, и `val()` для него вызывает `NotImplementedError`.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Каждая задача поддерживает свой набор форматов, показанный выше.
Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
те же `Results`. В разделе [экспорт](/docs/export) перечислены аргументы,
которые принимает каждый формат.

<code-tabs name="export" />

## Лицензирование

<provenance-box>

Строка «Weights» выше называет лицензию, которая здесь действует, —
Apache-2.0, но для этого семейства в Hugging Face-организации LibreYOLO на
самом деле ничего не переопубликовано: собственного чекпойнта LibreDINOv2 у
LibreYOLO нет. `LibreDINOv2(model_path=None)` скачивает репозиторий
`facebook/dinov2-with-registers-small` самой Meta, без изменений.

</provenance-box>

## Цитирование

<citation-block />
