---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: семантическая сегментация в LibreYOLO'
description: >-
  Используйте LingBot-Vision в LibreYOLO для семантической сегментации на
  бэкбоне ViT под Apache-2.0. Установка, предсказание, обучение, валидация и
  экспорт, размеры s/b/l.
lead: >-
  LingBot-Vision — семейство self-supervised бэкбонов vision transformer,
  обученных маскированным моделированием с упором на границы объектов ради
  плотного пространственного восприятия; выпущено компанией Robbyant. LibreYOLO
  соединяет бэкбон с плотной головой и поддерживает его для одной задачи —
  семантической сегментации.
keywords:
  - LingBot-Vision
  - семантическая сегментация python
  - vision transformer сегментация
  - self-supervised предобучение
  - моделирование границ
  - Robbyant
  - плотное предсказание
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (linear probe)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Бэкбон по умолчанию заморожен — как в оригинальном протоколе
        # оценки: обучается только плотная голова 1x1.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Полное дообучение
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла: экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreLingBotVisions-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Установка

LingBot-Vision не требует опциональных extra. Всё, что он импортирует, входит в
базовую установку.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`result.semantic_mask` содержит плотную карту классов: `.data` — тензор
`(H, W)` с id классов в исходном размере изображения, а `.classes` перечисляет
id классов, которые действительно присутствуют. `result.boxes` равен `None`,
поскольку детекций по экземплярам здесь нет. `conf` и `iou` принимаются ради
совместимости API, но не меняют вывод: модель возвращает по одному классу на
пиксель, а не детекции, которые нужно фильтровать. Про источники, стриминг и
обработку результатов см. [предсказание](/docs/predict).

## Варианты

Три опубликованных размера — s, b и l, — дистиллированные из модели-учителя
ViT-g/16 на 1.1 млрд параметров. Сам учитель, размер `g`, загружается и
дообучается в LibreYOLO, но собственного чекпойнта `g` LibreYOLO не размещает.

<checkpoint-table />

## Обучение

`train()` дообучает опубликованный чекпойнт. Рецепт по умолчанию — linear probe
из оригинального отчёта: бэкбон ViT заморожен, обучается только плотная голова
1x1 — именно так получены размещённые в LibreYOLO веса выше. Чтобы вместо этого
дообучить всю сеть, передайте `freeze_backbone=False` и будьте готовы
соответственно понизить `lr0`.

<code-tabs name="train" />

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`: mIoU и точность по пикселям,
измеренные на любом датасете в том формате, в котором вы обучали.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. [Экспорт](/docs/export) перечисляет аргументы, которые
принимает каждый формат.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

В оригинальном релизе указано, что его ViT построен на архитектуре
DINOv2/DINOv3, опубликованной Meta AI. Robbyant распространяет свою реализацию
под Apache-2.0, а этот порт в LibreYOLO сделан только из репозитория Robbyant,
никогда — из кода DINOv2 или DINOv3 от Meta.

</provenance-box>

## Цитирование

<citation-block />
