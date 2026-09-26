---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo у LibreYOLO: виявлення без прикладів у реальному часі'
description: >-
  Використовуйте OMDet-Turbo у LibreYOLO для виявлення з відкритим словником у
  реальному часі. Установіть додаткові залежності openvocab і виконуйте
  передбачення з довільним текстовим словником.
lead: >-
  OMDet-Turbo є детектором об'єктів із відкритим словником у реальному часі від
  Om AI Lab, який відокремлює ембединги класів від мовного запиту завдання.
  LibreYOLO надає його як сімейство лише для передбачення на рівні детекторів із
  відкритим словником.
keywords:
  - OMDet-Turbo
  - OmDet
  - виявлення об'єктів із відкритим словником
  - детекція в реальному часі
  - zero-shot детекція
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Власний поріг NMS
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["traffic light", "bicycle"])

        # OMDet-Turbo є єдиним сімейством цього рівня, яке враховує iou=:
        # його власна подальша обробка приймає поріг придушення як аргумент,
        # типовим значенням якого є 0.5, якщо iou= не задано.
        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)
        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Встановлення

OMDet-Turbo завантажується через рівень детекторів LibreYOLO з відкритим
словником, для якого потрібен додатковий набір залежностей `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Цей набір установлює `transformers` і `timm`, бібліотеки Hugging Face, які
використовує цей рівень. Бекбон Swin моделі OMDet-Turbo завантажується через
обгортку `TimmBackbone` із бібліотеки `transformers`.

## Передбачення

OMDet-Turbo не є контрольною точкою, яку LibreYOLO завантажує через
`LibreYOLO()`. Вона завантажується через споріднену фабрику
`LibreOpenVocab`, яка під час першого використання отримує знімок із
Hugging Face і кешує його в `weights/`.

<code-tabs name="predict" />

`set_classes()` задає текстовий словник, який зберігається для наступних
викликів. Викличте метод знову, щоб повністю замінити список, або не викликайте
його, щоб залишити типові мітки COCO-80. Порожній результат є допустимим і не
вважається помилкою. На відміну від Grounding DINO, OMDet-Turbo відокремлює
ембединги класів від мовного запиту завдання, тому подальша обробка
`transformers` повертає мітки, які без етапу усунення неоднозначності фраз
безпосередньо зіставляються із запитаним списком класів.

OMDet-Turbo не має порога для текстових токенів: виявлення фільтрує лише
`conf`, а передавання `text_threshold` спричиняє помилку. Це єдине
сімейство цього рівня, яке виконує власне немаксимальне придушення всередині
`post_process_grounded_object_detection`, тому `iou` тут ураховується без
попередження. `imgsz` та `augment=True` одразу відхиляються: за зміну
розміру відповідає процесор `transformers`, а аугментація під час тестування
не належить до можливостей цього рівня. `predict()` для одного зображення
повертає один об'єкт `Results`, а не список. Передайте каталог, список
зображень або `stream=True` для джерела відео, щоб отримати кілька
результатів. Для цього сімейства немає шляху через CLI: `libreyolo predict`
завантажує лише контрольні точки `.pt` через `LibreYOLO()`, тому сімейства
`LibreOpenVocab` запускаються з Python. Типи джерел і потокове передбачення
описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступна одна контрольна точка `t`, єдиний розмір рівня. Вона відтворює
`omlab/omdet-turbo-swin-tiny-hf` у зафіксованій початковій редакції через
`OmDetTurboForObjectDetection` із бібліотеки `transformers`. Дзеркальний
файл ваг побайтово ідентичний початковому знімку. Показники правильності та
затримки для цього сімейства ще не опубліковано.

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
