---
title: Дистилляция знаний
seo_title: Дистилляция знаний в LibreYOLO
description: >-
  Обучение маленького детектора под руководством более крупного учителя или
  замороженного бэкбона DINOv2: функции потерь MGD, CWD и feature-MSE, точки
  съёма и поддержка семейств.
lead: >-
  Дистилляция добавляет второе слагаемое к функции потерь, которое подтягивает
  промежуточные карты признаков ученика к картам замороженного учителя.
  LibreYOLO снимает признаки forward-хуками, поэтому голова и функция потерь
  самого учителя вообще не участвуют.
keywords:
  - дистилляция знаний
  - masked generative distillation
  - channel-wise distillation
  - дистилляция признаков
  - дистилляция из dinov2
  - обучение teacher student
  - mgd loss
  - cwd loss
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Более крупный чекпойнт того же семейства выступает учителем для
        маленького.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Замороженный self-supervised ViT выступает учителем для одной стадии
        бэкбона.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Настройка функции потерь
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # общий вес дистилляции
            distill_tau=1.0,   # температура softmax в CWD
        )
source_hash: 7210031328f6826f
---

## Дистилляция из более крупного чекпойнта

Дистилляция включается, если задать `distill_model`. Значение — чекпойнт учителя,
который загружается той же фабрикой, что и любая другая модель.

<code-tabs name="detector" />

Прямой проход учителя выполняется под `no_grad`, а при включённом AMP — ещё и под
autocast, поэтому замороженная модель не выполняет на каждом шаге вычисления в
полной точности. Forward-хуки снимают его карты признаков в именованных точках съёма,
функция потерь сравнивает их с картами ученика, а результат добавляется к
функции потерь обучения и выводится как компонент с именем `distill`.

## Дистилляция из замороженного foundation-бэкбона

Вместо этого учителем для одной стадии бэкбона ученика может выступать
self-supervised ViT. Признаки учителя берутся из его собственного экстрактора
признаков, а не из хуков, и функция потерь сама справляется с несоответствием
между сеткой патчей и свёрточным страйдом.

<code-tabs name="foundation" />

`distill_model` понимает `dinov2` — это DINOv2-base, — а также `dinov2_vits14`,
`dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`, `dinov2-base`, `dinov2-large`
и любой идентификатор хаба, начинающийся с `facebook/dinov2`. Всё остальное
трактуется как путь к чекпойнту учителя.

Этот путь использует `feat_mse` независимо от `distill_loss_type` и требует
установленного `transformers`. Если учитель загружается с недостающими ключами
весов, запуск прерывается, вместо того чтобы дистиллировать из частично
случайного бэкбона.

## Какие семейства

Поддержка дистилляции — это метод модели-ученика, и таких методов два.

`get_distill_config()` задаёт многомасштабные точки съёма, по которым
учитель-детектор обучает ученика. Его реализуют YOLOv9, YOLOX и RF-DETR.

`get_backbone_distill_config()` задаёт единственную стадию бэкбона, по которой
ученика обучает foundation-учитель. Его реализует YOLOv9 — и только это семейство.

Всё остальное вызывает ошибку, а не обучается без этой функции потерь:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Точки съёма

Точки съёма фиксированы для каждого семейства и каждой роли, поэтому учитель и
ученик не обязаны быть одной архитектурой; им нужны совпадающие страйды признаков.

| Семейство | Роль | Точки съёма | Страйды |
|---|---|---|---|
| YOLOv9 | учитель или ученик | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | foundation-ученик | `backbone.elan3` | 16 |
| YOLOX | учитель или ученик | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | учитель или ученик | `model.backbone.0.projector.stages.0` | определяются при настройке |

Несовпадающие страйды вызывают ошибку до начала обучения:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

Для foundation-учителей эта проверка пропускается: весь их смысл в том, что сетки
различаются.

## Три функции потерь

`distill_loss_type` выбирает функцию потерь по признакам для учителя-детектора.
Foundation-учитель всегда использует `feat_mse`.

`mgd`, masked generative distillation, маскирует часть пространственных позиций
ученика и обучает небольшой генератор из двух свёрток восстанавливать полную
карту признаков учителя по тому, что осталось. `distill_mask_ratio` задаёт долю
замаскированных позиций, по умолчанию 0.65.

`cwd`, channel-wise distillation, превращает пространственные активации каждого
канала в распределение вероятностей и минимизирует KL-дивергенцию канал за
каналом. `distill_tau` — температура softmax, по умолчанию 1.0.

`feat_mse` приводит каналы ученика к каналам учителя свёрткой 1x1, билинейно
масштабирует сетку учителя под сетку ученика и берёт среднеквадратичную ошибку.
`distill_normalize=True` сначала L2-нормализует обе карты признаков по канальной
размерности, так что сопоставление учитывает только угол и не зависит от
масштаба. По умолчанию — `False`.

`dis` — общий вес, который применяется поверх. Если его не задать, каждая функция
потерь берёт своё опубликованное значение по умолчанию: 2e-5 для MGD, 1.0 для CWD
и 1.0 для feature MSE. Они различаются на пять порядков, поэтому вес, подобранный
для одного типа потерь, для другого бессмыслен.

<code-tabs name="tuned" />

У `distill_mask_ratio`, `distill_tau` и `distill_normalize` нет флагов в CLI. Это
аргументы Python или YAML-ключи в `cfg=`. Для RF-DETR дистилляция целиком
доступна только из Python, потому что сопоставление CLI-аргументов для этого
семейства не переносит ключи дистилляции.

## Адаптеры, чекпойнты и несколько GPU

Каждая функция потерь строит небольшие обучаемые модули, которые живут вне
ученика: канальные адаптеры 1x1 и генератор MGD. Они получают собственную группу
параметров в оптимизаторе — с той же эффективной скоростью обучения, что и весь
запуск.

Эти модули записываются в чекпойнт под ключом `distiller` и восстанавливаются при
возобновлении, поэтому возобновлённый запуск не начинает свои проекторы с нуля.

При DDP адаптеры находятся вне обёрнутого ученика, а значит, редьюсер DDP не видит
их градиенты. Обучающий цикл на каждом шаге явно делает по ним all-reduce, поэтому
все ранги обучают одни и те же адаптеры.

Захват CUDA-графа при запуске с дистилляцией недоступен. Передача
`cuda_graph=True` пишет одну строку в лог, и обучение идёт в eager-режиме. См.
[Производительность обучения](/docs/train/performance).

## Связанные страницы

- [Заморозка слоёв](/docs/train/layer-freezing) и
  [дообучение с LoRA](/docs/train/lora) — ни то, ни другое не запрещено сочетать
  с дистилляцией.
- [Гиперпараметры](/docs/train/hyperparameters) — про остальные аргументы
  `train()`.
