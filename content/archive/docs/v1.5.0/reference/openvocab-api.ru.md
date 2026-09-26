---
title: API открытого словаря
seo_title: 'API LibreOpenVocab: алиасы и аргументы'
description: >-
  Фабрика LibreOpenVocab, её четыре семейства и все алиасы, set_classes,
  значения conf по умолчанию для каждого семейства, а также правила для
  text_threshold и iou.
lead: >-
  LibreOpenVocab — фабрика детекторов, обусловленных текстом. Список классов
  здесь — это промпт, а не фиксированная голова, поэтому словарь задаётся через
  set_classes, а модель возвращает по нему обычные Results детекции.
keywords:
  - LibreOpenVocab
  - детекция с открытым словарём python
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - найти объект по текстовому запросу без обучения
last_verified: 1.5.0
verification: >-
  Алиасы прочитаны из libreyolo/models/openvocab/__init__.py; репозитории,
  размеры и пороги — из grounding_dino.py, owlv2.py, omdet_turbo.py и
  ov_deim.py; правила вызова — из libreyolo/models/openvocab/base.py, всё на
  версии 1.5.0. Замысел архитектуры — из
  docs/adr/0008-open-vocab-detector-contract.md.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## Установка

Этому уровню нужен extra `openvocab`.

<code-tabs name="install" />

## Фабрика

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` — это алиас, а не путь. Перед поиском подчёркивания сворачиваются в
дефисы, поэтому имена с указанием семейства вроде `omdet_turbo-t` и
`grounding_dino-t`, которые печатает инвентарь CLI, загружаются как есть.
Неизвестный алиас вызывает `ValueError` со списком всех известных алиасов.

Конструктор принимает `size`, `nb_classes=80`, `names=None`,
`device="auto"`, `task=None` и `text_threshold=None`. Передать `names` — то же
самое, что вызвать `set_classes` сразу после загрузки. Передача
`text_threshold` семейству, которое его не поддерживает, вызывает `TypeError`.

<code-tabs name="usage" />

## Семейства и алиасы

| Семейство | Алиасы | Размеры | Веса |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

Алиас по умолчанию — `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` и `LibreOMDetTurbo` экспортируются на уровне
пакета, и их можно создать напрямую с `size=`. OV-DEIM доступен через алиасы
фабрики выше.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Задаёт словарь для всех последующих вызовов `predict()` и возвращает модель,
чтобы вызовы можно было соединять в цепочку. Список должен быть непустым,
содержать только строки, а его элементы должны быть уникальными при сравнении
без учёта регистра; пустые метки отклоняются. Передача голой строки вызывает
`TypeError`, потому что она развернулась бы в набор односимвольных классов.

После вызова `model.names` сопоставляет `0..N-1` с метками в заданном порядке, а
`model.nb_classes` равно `N`.

## Аргументы вызова

Этот уровень переиспользует стандартный набор аргументов predict с тремя
отличиями.

`conf` по умолчанию берёт собственное значение семейства, а не общее 0.25:

| Семейство | conf по умолчанию | Подавление |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Собственная постобработка, порог 0.5, учитывает `iou=` |
| OV-DEIM | 0.25 | Взаимно-однозначное сопоставление с отбором top-K, без подавления |

`iou=` имеет смысл только для семейства, которое выполняет подавление.
OMDet-Turbo принимает порог как аргумент и, если `iou=` не задан, берёт 0.5 по
умолчанию. Остальные три семейства ничего не подавляют, поэтому переданный им
`iou=` выдаёт предупреждение и игнорируется.

`text_threshold=` есть только у Grounding DINO, где по умолчанию равен 0.25.
Его можно передать в конструктор, чтобы значение сохранялось, или задавать при
каждом вызове. Значение, переданное при вызове, нельзя сочетать с
`stream=True`, потому что результаты потока генерируются лениво; в этом случае
задайте его в конструкторе. Для всех остальных семейств он вызывает
`TypeError`.

`imgsz=` вызывает `ValueError`: изменением размера на этом уровне занимается
пайплайн предобработки. `augment=True` тоже вызывает ошибку, потому что
аугментация на этапе теста сюда не входит. Размеры входа для каждого семейства
приводятся только для справки: Grounding DINO 800, OWLv2 960 и 1008,
OMDet-Turbo 640, OV-DEIM 640.

## Что не поддерживается

`train()`, `val()`, `track()` и `export()` вызывают
`NotImplementedError`. Дообучайте модель в апстриме и загружайте полученные веса;
вместо трекинга запускайте `predict()` для каждого кадра. Для валидации нужен
отдельный валидатор, потому что общий валидатор детекции вызывает модель с
тензорами изображений, а этому уровню нужны входы, обусловленные текстом.
