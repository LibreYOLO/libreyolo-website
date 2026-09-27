---
title: API відкритого словника
seo_title: 'API LibreOpenVocab: псевдоніми й аргументи'
description: >-
  Фабрика LibreOpenVocab, її чотири сімейства та всі псевдоніми, set_classes,
  стандартні значення conf для сімейств і правила text_threshold та iou.
lead: >-
  LibreOpenVocab, це фабрика детекторів з текстовими умовами. Список класів є
  підказкою, а не фіксованою головою, тому словник задає set_classes, а модель
  повертає для нього звичайні результати виявлення.
keywords:
  - LibreOpenVocab
  - виявлення з відкритим словником
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: 1.5.0
verification: >-
  Псевдоніми прочитано з libreyolo/models/openvocab/__init__.py; репозиторії,
  розміри й пороги взято з grounding_dino.py, owlv2.py, omdet_turbo.py та
  ov_deim.py; правила викликів взято з libreyolo/models/openvocab/base.py, усе у
  версії v1.5.0. Проєктний задум взято з
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

## Встановлення

Для цього рівня потрібна додаткова залежність `openvocab`.

<code-tabs name="install" />

## Фабрика

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model`, це псевдонім, а не шлях. Перед пошуком символи підкреслення
перетворюються на дефіси, тому імена з указаним сімейством, які показує
перелік CLI, як-от `omdet_turbo-t` і `grounding_dino-t`, завантажуються
без змін. Невідомий псевдонім спричиняє `ValueError` зі списком усіх
відомих псевдонімів.

Конструктор приймає `size`, `nb_classes=80`, `names=None`,
`device="auto"`, `task=None` і `text_threshold=None`. Передавання
`names` рівнозначне виклику `set_classes` відразу після завантаження.
Передавання `text_threshold` сімейству, яке його не підтримує, спричиняє
`TypeError`.

<code-tabs name="usage" />

## Сімейства та псевдоніми

| Сімейство | Псевдоніми | Розміри | Ваги |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

Стандартний псевдонім, `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2` і `LibreOMDetTurbo` експортуються
на рівні пакета, їх можна створювати безпосередньо з `size=`. OV-DEIM
доступна через наведені вище псевдоніми фабрики.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Задає словник для кожного наступного виклику `predict()` і повертає модель,
що дає змогу ланцюжити виклики. Список має бути непорожнім, містити лише
рядки, а його елементи мають бути унікальними без урахування регістру;
порожні мітки відхиляються. Передавання окремого рядка спричиняє `TypeError`,
оскільки інакше його було б розбито на односивольні класи.

Після виклику `model.names` зіставляє `0..N-1` з мітками в заданому
порядку, а `model.nb_classes` дорівнює `N`.

## Аргументи виклику

Цей рівень повторно використовує стандартний інтерфейс передбачення з трьома
відмінностями.

Стандартне значення `conf` береться з конкретного сімейства, а не зі
спільного значення 0.25:

| Сімейство | Стандартне conf | Пригнічення |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Власна постобробка, поріг 0.5, ураховує `iou=` |
| OV-DEIM | 0.25 | Взаємно однозначне зіставлення з вибором top-K, без пригнічення |

`iou=` має значення лише для сімейства, яке виконує пригнічення. OMDet-Turbo
приймає поріг як аргумент і використовує 0.5, якщо `iou=` не задано. Інші
три сімейства нічого не пригнічують, тому передавання `iou=` для них
виводить попередження й ігнорується.

`text_threshold=` підтримує лише Grounding DINO, стандартне значення для
неї становить 0.25. Його можна задати під час створення як постійне значення
або для окремого виклику. Значення окремого виклику не можна поєднувати з
`stream=True`, оскільки потокові результати генеруються ліниво; натомість
задайте його в конструкторі. Для всіх інших сімейств цей аргумент спричиняє
`TypeError`.

`imgsz=` спричиняє `ValueError`: зміною розміру керує конвеєр
попередньої обробки цього рівня. `augment=True` також спричиняє помилку,
оскільки аугментація під час тестування тут не підтримується. Розміри входу
наведено лише для довідки за сімействами: Grounding DINO 800, OWLv2 960 і
1008, OMDet-Turbo 640, OV-DEIM 640.

## Не підтримується

`train()`, `val()`, `track()` і `export()` спричиняють
`NotImplementedError`. Виконайте донавчання в upstream і завантажте
отримані ваги; замість відстеження запускайте `predict()` для кожного
кадру. Для валідації потрібен окремий валідатор, оскільки спільний валідатор
виявлення викликає модель із тензорами зображень, а цьому рівню потрібні
входи з текстовими умовами.

