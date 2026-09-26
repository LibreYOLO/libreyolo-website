---
title: Аугментации
seo_title: Аугментации при обучении в LibreYOLO
description: >-
  Параметры аугментации в TrainConfig, четыре формы пайплайна за ними и таблица
  по семействам: какие параметры используются, привязаны к мозаике или
  игнорируются.
lead: >-
  Аугментация настраивается параметрами TrainConfig, но каждое семейство моделей
  выполняет свой обучающий пайплайн, и пайплайн, в котором нет ветки мозаики,
  игнорирует mosaic_prob, а не приближает его.
keywords:
  - аугментация данных yolo
  - mosaic augmentation
  - mixup
  - hsv джиттер
  - случайное аффинное преобразование
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # В CLI mosaic_prob пишется как mosaic, а mixup_prob — как mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  support:
    - label: Чтение таблицы поддержки для семейства
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Только игнорируемые
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Набор для классификации
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 47461cd13aab580c
---

## Настройка параметров

Параметры аугментации — обычные аргументы `train()`.

<code-tabs name="train" />

У двух из них написание в CLI короче: `mosaic` соответствует `mosaic_prob`, а
`mixup` — `mixup_prob`. Все остальные параметры пишутся одинаково в обоих
местах.

## Три состояния, а не два

Делает ли параметр хоть что-нибудь, зависит от семейства. В библиотеке это
описано декларативной таблицей, и каждая запись в ней — одно из трёх состояний.

`used` значит, что параметр доходит до пайплайна и меняет сэмплы. `ignored`
значит, что до пайплайна он не доходит, поэтому задавать его бесполезно.
`gated_by_mosaic` значит, что он применяется только к сэмплам, прошедшим ветку
мозаики, поэтому при `mosaic_prob=0` он не срабатывает никогда, хотя и подключён.

Третье состояние — то самое, которое всех удивляет. В пайплайне в стиле YOLOX
аффинное преобразование выполняется на холсте мозаики, а MixUp подмешивает
мозаичный сэмпл, поэтому `mosaic_prob=0` молча отключает сразу `degrees`,
`translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` и
`mixup_scale`. Для случая с MixUp трейнер отдельно пишет предупреждение в лог:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI предупреждает и об игнорируемых параметрах, перечисляя только те, которые вы
задали сами:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Четыре формы пайплайна

Семейства группируются в четыре обучающих пайплайна, и почти все ответы
определяются пайплайном.

Мозаичный пайплайн в стиле YOLOX применяет HSV-джиттер и отражения к каждому
сэмплу, а аффинное преобразование и MixUp выполняет внутри ветки мозаики. К нему
относятся YOLOX, YOLOv7, YOLOv9 и его варианты E2E и P2, RTMDet, PicoDet,
RT-DETR, RT-DETRv2 и FOMO.

Сквозной пайплайн в стиле DETR — без мозаики и без аффинного преобразования.
Фотометрическое искажение, zoom-out и обрезка по IoU заданы в нём константами
рецепта, а не параметрами конфигурации, поэтому реально работают только
`flip_prob` и `no_aug_epochs`. К нему относятся D-FINE, Dome-DETR, DEIM, DEIMv2,
RT-DETRv4, EC и, с одним отличием, RF-DETR.

Пайплайн классификации на ImageFolder игнорирует все параметры детекции. Его
горизонтальное отражение зафиксировано на 0.5, и `flip_prob` до него не доходит.
Вместо этого у него есть свой набор параметров, описанный ниже.

YOLO-NAS — отдельная форма: мозаики нет вовсе, аффинное преобразование для
каждого сэмпла включено всегда, а MixUp применяется независимо, не привязан к
мозаике. Его значение `mosaic_scale` переиспользуется как диапазон масштаба для
аффинного преобразования.

У SegFormer и NAFNet каждый свой пайплайн под конкретную задачу, и случайность в
нём зашита в семействе, а не настраивается. У SegFormer работают атрибуты класса
`semantic_scale_jitter` и `semantic_hsv_prob`, а не `mosaic_scale` и `hsv_prob`.
Обрезка и отражения в NAFNet — связанные
операции над входом и целью с фиксированной вероятностью 0.5.

## Какое семейство учитывает какой параметр

Таблица ниже — поставляемая с библиотекой спецификация из
`libreyolo/data/augment/spec.py`, которую собственные тесты библиотеки сверяют с
реальной обвязкой пайплайнов. Читайте её оттуда, а не выводите из архитектуры.

<code-tabs name="support" />

Сводка по пайплайнам, для базовых параметров:

| Параметр | В стиле YOLOX | YOLO-NAS | В стиле DETR | Классификация |
|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored |
| `mixup_prob` | привязан к мозаике | used | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored |
| `flip_prob` | used | used | used | ignored |
| `flipud` | used | used | ignored | ignored |
| `degrees` | привязан к мозаике | used | ignored | ignored |
| `translate` | привязан к мозаике | used | ignored | ignored |
| `shear` | привязан к мозаике | used | ignored | ignored |
| `perspective` | привязан к мозаике | used | ignored | ignored |
| `mosaic_scale` | привязан к мозаике | used | ignored | ignored |
| `mixup_scale` | привязан к мозаике | used | ignored | ignored |
| `no_aug_epochs` | used | used | used | used |

Исключения внутри этих столбцов, все — в сторону сужения:

- У RTMDet, PicoDet, RT-DETR, RT-DETRv2 и FOMO нет вертикального отражения,
  поэтому `flipud` игнорируется. Обёртка мозаики в FOMO вдобавок собрана без
  perspective.
- В нативном пайплайне RF-DETR нет HSV-джиттера, поэтому `hsv_prob` игнорируется
  дополнительно к столбцу в стиле DETR.
- EC учитывает `hsv_prob`, `degrees` и `translate`, но только для `task="pose"` —
  их читает преобразование с учётом ключевых точек. В путях detect и segment
  применяются фиксированные фотометрические рецепты.
- DINOv2 следует столбцу в стиле DETR для задач detect и semantic, а для
  `task="classify"` добавляет набор для классификации.

`no_aug_epochs` везде имеет статус `used`, но смысл у него везде разный. В
мозаичных пайплайнах он отключает мозаику и MixUp на последних эпохах. В
пайплайнах в стиле DETR он останавливает фотометрические аугментации, zoom-out и
обрезку и формирует хвост расписания. В пайплайнах классификации и семантики он
формирует только хвост.

## Набор для классификации

Четыре параметра управляют пайплайном классификации и больше ничем. Семейства
детекции игнорируют все четыре.

<code-tabs name="classify" />

`auto_augment` принимает `"randaugment"`, `"autoaugment"`, `"augmix"` или `None`.
`erasing` — вероятность RandomErasing. `mixup` и `cutmix` — вероятности на уровне
батча, дающие мягкие метки; за батч срабатывает не больше одной, сначала MixUp,
поэтому они складываются и в сумме не должны превышать 1.

Все четыре по умолчанию выключены, поэтому обучение классификации не меняется,
пока вы сами этого не попросите.

Одно совпадение имён стоит проговорить прямо: в CLI `mixup` — алиас для
`mixup_prob` из детекции. У поля `mixup` из классификации собственного написания
в CLI нет, и добраться до него можно только через `model.train(mixup=...)` в
Python.

## Параметры, специфичные для семейства

Некоторые параметры живут в подклассе конфига семейства, а не в базовом классе,
поэтому они существуют только для этого семейства и флага в CLI у них нет.

| Семейство | Параметр | Эффект |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Вероятность copy-paste-аугментации экземпляров, только `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` переиспользует тот же сэмпл в отражении, `"mixup"` берёт второй сэмпл |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Вероятность случайного поворота на 90 градусов |
| YOLOv9 | `max_labels` | Ограничение на число объектов эталонной разметки на изображение в обучающих преобразованиях, по умолчанию 100 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste для `task="segment"`, только в режиме `"flip"` |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Вероятность случайной обрезки с изменением размера |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Вероятности джиттера в пути pose и аффинного преобразования с учётом ключевых точек |

`max_labels` — тот самый параметр, который молча теряет данные. Рамки сверх
ограничения отбрасываются без ошибки, поэтому для плотных изображений вроде
аэрофотосъёмки его нужно поднимать.

Мозаика и MixUp отключаются при обучении с повёрнутыми рамками независимо от
параметров, потому что аугментация с учётом углов для повёрнутых рамок не
реализована.

## Смотрите также

- [Гиперпараметры](/docs/train/hyperparameters) — про `no_aug_epochs` как
  аргумент расписания и про остальной `train()`.
- [Датасеты](/docs/train/datasets) — про форматы разметки, которые эти
  преобразования принимают.
