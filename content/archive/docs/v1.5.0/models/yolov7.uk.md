---
title: YOLOv7
families:
  - yolo7
seo_title: 'YOLOv7 у LibreYOLO: передбачення, навчання та експорт під ліцензією MIT'
description: >-
  Запускайте YOLOv7 у LibreYOLO для виявлення об'єктів: установлення,
  передбачення, навчання, валідація та експорт коду й ваг під ліцензією MIT.
lead: >-
  YOLOv7 є одностадійним детектором на основі якорів, голова якого додає навчені
  зсуви неявних знань перед завершальною згорткою. LibreYOLO підтримує єдиний
  опублікований розмір для виявлення.
keywords:
  - YOLOv7
  - детекція об'єктів
  - детекція на основі якорів
  - неявні знання
  - ImplicitA
  - виявлення об'єктів у реальному часі
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Теплий старт нової моделі
      language: python
      code: >
        from libreyolo import LibreYOLO7


        # pretrained=True завжди завантажує опубліковану контрольну точку
        LibreYOLO7b.pt

        # незалежно від того, з чим створено цей екземпляр. Безпосереднє
        створення

        # класу замість LibreYOLO() починається без жодних завантажених ваг.

        model = LibreYOLO7(None, size="b")

        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreYOLO7b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## Встановлення

Для YOLOv7 не потрібні додаткові залежності понад базовий пакет.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. `conf` задає поріг упевненості, а
`iou` задає поріг NMS після декодування голови на основі якорів. Типи джерел,
потокове передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

LibreYOLO постачає один розмір `b`. Початковий проєкт публікує одну модель
YOLOv7, тому вибору розміру немає.

## Навчання

<code-tabs name="train" />

`pretrained` читається, на відміну від однойменного аргументу, який нічого не
робить у деяких інших сімействах тут: передайте `True` для теплого старту з
опублікованої контрольної точки `LibreYOLO7b.pt` (завантажується
автоматично) або шлях чи назву для іншої. Опублікована контрольна точка має
80 класів COCO, тому її запит для моделі, уже перебудованої під іншу кількість
класів, спочатку перебудовує модель назад до 80, завантажує її, а після
читання кількості класів датасету переносить усі тензори відповідної форми до
цільової голови. `resume=True` не можна поєднувати з `pretrained`. З
типовим `None` навчання продовжується з ваг, з якими створено модель, або з
випадкової ініціалізації, якщо нічого не завантажено.

За інших типових параметрів тренер виконує 300 епох із `lr0=0.01`,
імпульсом SGD 0.937, прогріванням протягом 3 епох, тим самим призначенням
SimOTA та завершальним 15-епоховим етапом без аугментацій, який використовує
YOLOX, адаптованим до голови на основі якорів. Єдина відмінність: YOLOX додає
у завершальних епохах уточнення регресії рамок L1, яке v7 пропускає, оскільки
функція втрат SimOTA у v7 не має гілки L1 необроблених зсувів для уточнення.

Датасети, аугментацію, кілька GPU та логери описано в розділі
[навчання](/docs/train).

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому проводилося
навчання.

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

## Цитування

<citation-block />
