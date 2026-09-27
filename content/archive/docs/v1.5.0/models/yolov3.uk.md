---
title: YOLOv3
families:
  - yolo3
seo_title: 'YOLOv3 у LibreYOLO: передбачення, валідація, експорт'
description: >-
  Запускайте YOLOv3 у LibreYOLO: заморожене музейне сімейство лише для інференсу
  в розмірах tiny, base і SPP. Виконуйте передбачення, валідацію та експорт за
  умовами ліцензії суспільного надбання.
lead: >-
  YOLOv3 є детектором на основі Darknet-53, який додав до лінійки YOLO
  багатомасштабне передбачення та незалежні логістичні класифікатори. LibreYOLO
  зберігає її як заморожений експонат лише для інференсу в розмірах tiny, base і
  SPP.
keywords:
  - YOLOv3
  - Darknet
  - Darknet-53
  - детекція об'єктів
  - багатомасштабне виявлення
  - музейне сімейство моделей
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Розмір SPP
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Варіант SPP додає блок просторового пірамідального пулінгу перед
        # головами виявлення та працює з власним початковим розміром входу.
        model = LibreYOLO("LibreYOLO3spp.pt")
        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreYOLO3b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## Встановлення

Для YOLOv3 не потрібні додаткові залежності понад базовий пакет.

```bash
pip install libreyolo
```

## Передбачення

Це сімейство призначене лише для інференсу: `train()` спричиняє
`NotImplementedError`, тому на цій сторінці немає розділу «Навчання».
Передбачення, валідація та експорт підтримуються. Під час першого використання
ваги завантажуються з Hugging Face і кешуються локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. `conf` фільтрує за порогом упевненості,
а `iou` задає поріг NMS. Обидва параметри застосовуються окремо для кожного
масштабу до об'єднання рамок з усіх трьох голів. Типи джерел, потокове
передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому виконується
валідація.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results`. Граф також можна запускати
безпосередньо в середовищі виконання без установленої LibreYOLO, але тоді
попередню та подальшу обробку потрібно реалізувати самостійно.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>
