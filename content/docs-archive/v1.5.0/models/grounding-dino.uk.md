---
title: Grounding DINO
families:
  - grounding_dino
seo_title: 'Grounding DINO у LibreYOLO: виявлення у відкритій множині'
description: >-
  Використовуйте Grounding DINO у LibreYOLO для виявлення будь-яких об'єктів за
  текстовим описом. Установіть додаткові залежності openvocab і виконуйте
  передбачення з довільним текстовим словником.
lead: >-
  Grounding DINO є детектором об'єктів у відкритій множині від IDEA Research,
  який оцінює зображення відносно довільного текстового запиту замість
  фіксованого списку класів. LibreYOLO надає його як сімейство лише для
  передбачення на рівні детекторів із відкритим словником.
keywords:
  - Grounding DINO
  - виявлення об'єктів із відкритим словником
  - open-set детекція
  - zero-shot детекція
  - детектор за текстовим описом
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Текстовий поріг
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("grounding-dino-b")

        model.set_classes(["remote control", "school bus"])


        # conf фільтрує за оцінкою рамки, а text_threshold за оцінкою токена

        # декодованої фрази. Якщо їх не задано, типове значення обох дорівнює
        0.25.

        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)

        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Встановлення

Grounding DINO завантажується через рівень детекторів LibreYOLO з відкритим
словником, для якого потрібен додатковий набір залежностей `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Цей набір установлює `transformers` і `timm`, бібліотеки Hugging Face, які
використовує цей рівень.

## Передбачення

Grounding DINO не є контрольною точкою, яку LibreYOLO завантажує через
`LibreYOLO()`. Вона завантажується через споріднену фабрику
`LibreOpenVocab`, яка під час першого використання отримує знімок із
Hugging Face і кешує його в `weights/`.

<code-tabs name="predict" />

`set_classes()` задає текстовий словник, який зберігається для наступних
викликів. Викличте метод знову, щоб замінити список, або не викликайте його,
щоб залишити типові мітки COCO-80. Grounding DINO декодує довільні фрази з
власного текстового виходу й самостійно зіставляє їх зі словником. Перевагу має
точний нормалізований збіг, також приймається збіг цілого токена, а
неоднозначна чи незнайдена фраза відкидається без здогадок. Тому `school bus`
ніколи не зіставляється лише з `bus` або `school`. Словник, довжина якого
перевищує ліміт токенів текстового енкодера, поділяється на кілька запитів.
Вони обробляються окремими прямими проходами й об'єднуються в один набір
виявлень із лімітом `max_det`.

`iou` приймається для сумісності API, але видає попередження й нічого не
робить, оскільки тут не виконується немаксимальне придушення. `imgsz` та
`augment=True` одразу відхиляються: за зміну розміру відповідає процесор
`transformers`, а аугментація під час тестування не належить до можливостей
цього рівня. `predict()` для одного зображення повертає один об'єкт
`Results`, а не список. Передайте каталог, список зображень або
`stream=True` для джерела відео, щоб отримати кілька результатів. Для цього
сімейства немає шляху через CLI: `libreyolo predict` завантажує лише
контрольні точки `.pt` через `LibreYOLO()`, тому сімейства
`LibreOpenVocab` запускаються з Python. Типи джерел і потокове передбачення
описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступні дві контрольні точки: `t` і `b`. Якщо розмір не вказано, типовим
для цього рівня є `t`. Обидві відтворюють офіційний реліз IDEA Research
через `GroundingDinoForObjectDetection` із бібліотеки `transformers` і
одноразово завантажуються у розміщений LibreYOLO знімок Hugging Face, який
зберігає початкові файли. Показники правильності та затримки для цього
сімейства ще не опубліковано.

Навчання, валідація на датасеті та експорт не належать до можливостей цього
рівня: `train()`, `val()` і `export()` завжди спричиняють
`NotImplementedError`. Це обгортка опублікованої контрольної точки лише для
передбачення.

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
