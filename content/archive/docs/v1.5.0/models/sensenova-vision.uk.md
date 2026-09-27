---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'SenseNova-Vision у LibreYOLO: 7 завдань, одна контрольна точка'
description: >-
  Використовуйте SenseNova-Vision у LibreYOLO для виявлення, сегментації,
  паноптичного подання, пози, точок, глибини й OCR за допомогою однієї
  генеративної контрольної точки із запитами.
lead: >-
  SenseNova-Vision є уніфікованою мультимодальною моделлю, яка подає завдання
  зору як генерацію за запитами у спільному декодері: рамки, точки, ключові
  точки й слова OCR виходять як текст із тегами, а карти глибини, масок і
  паноптичної сегментації виходять як зображення, які відтворює декодер.
  LibreYOLO завантажує її через LibreVLM і підтримує сім завдань за допомогою
  однієї контрольної точки 7B.
keywords:
  - SenseNova-Vision
  - SenseTime
  - уніфікована мультимодальна модель
  - Bagel
  - виявлення за запитами
  - щільне сприйняття
  - сегментація за описом
  - паноптична сегментація
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() перемикає завдання в тій самій завантаженій моделі.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Сегментація за описом і паноптична сегментація
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # Сегментація виконується за описом: їй потрібна цільова фраза, а не
        список класів.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Без власного словника паноптичне завдання використовує категорії

        # паноптичної сегментації COCO, на яких донавчено контрольну точку.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Точки, поза й OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Без заданого словника завдання пози типово використовує "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Встановлення

SenseNova-Vision потребує власного набору додаткових залежностей, який
установлює `accelerate` для розподілу великої моделі, потрібного цій
контрольній точці, а на платформах, відмінних від macOS, також
`bitsandbytes` для 4-бітного завантаження.

```bash
pip install "libreyolo[sensenova]"
```

Контрольну точку дзеркально розміщено на Hugging Face у власній організації
LibreYOLO, і під час першого використання вона завантажується автоматично.
Вона має ліцензію CC BY-NC 4.0 лише для некомерційного використання, і
завантажувач показує це повідомлення перед кожним автоматичним завантаженням.
Докладніше див. розділ «Ліцензування» нижче.

## Передбачення

<code-tabs name="predict" />

Кожне передбачення є дифузійним декодуванням на спільному бекбоні Bagel-MoT,
тому це модель можливостей, а не реального часу: затримка на зображення буде
помітно більшою, ніж у спеціалізованого детектора чи сегментатора.
`dtype="auto"` (типове значення) завантажує bf16 на GPU з достатнім обсягом
пам'яті й переходить до 4-бітного квантування NF4 в інших випадках, для чого
потрібен `bitsandbytes`. Передайте `dtype="bf16"`, щоб примусово
використовувати повну точність на достатньо великому GPU. Параметр
`noise_seed=42` під час створення задає початкове число дифузійного семплера
для відтворюваних щільних виходів; передайте `noise_seed=None`, щоб вимкнути
його.

Сім завдань використовують одну завантажену контрольну точку: `set_task()`
перемикає їх без повторного завантаження. `set_classes()` задає активний
словник. Виявлення, точки, поза й паноптична сегментація приймають список
класів, а сегментація виконується за описом і потребує точної фрази для
ізоляції. Кожне завдання повертає стандартний об'єкт `Results` із різним
заповненим корисним навантаженням: `boxes` для detect, `points` для point,
`boxes` і `keypoints` для pose, `ocr` для OCR, `depth_map` для depth,
`masks` для segment та `panoptic` (із `segments_info`) для panoptic.
Типи джерел, потокове передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Контрольні точки

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
