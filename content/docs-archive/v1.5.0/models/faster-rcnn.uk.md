---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: 'Faster R-CNN у LibreYOLO: передбачення, валідація та експорт'
description: >-
  Запускайте Faster R-CNN у LibreYOLO для виявлення об'єктів із чотирма
  бекбонами. Установлюйте, виконуйте передбачення, валідацію та експорт порту
  torchvision під ліцензією BSD-3-Clause.
lead: >-
  Faster R-CNN виявляє об'єкти за допомогою мережі пропозицій областей, яка
  подає дані до двостадійного класифікатора. Ця архітектура зробила пропозиції
  областей частиною тієї самої навченої мережі замість окремого етапу. LibreYOLO
  переносить реалізацію torchvision для виявлення.
keywords:
  - Faster R-CNN
  - детекція об'єктів
  - мережа пропозицій областей
  - двостадійний детектор
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreFasterRCNNl.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## Встановлення

Faster R-CNN не потребує додаткових залежностей. Усе, що вона імпортує,
входить до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. `conf` і `iou` задають пороги
впевненості та NMS. На відміну від детектора на основі запитів, Faster R-CNN
зберігає початковий етап NMS. Типи джерел, потокове передбачення та обробку
результатів описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступні чотири розміри, кожен з окремою конфігурацією torchvision, а не
масштабованою версією однієї конфігурації: `n` використовує
MobileNetV3-Large із входом 320 px, `s` має той самий бекбон із входом
800 px, `m` використовує ResNet-50 із пірамідою ознак, а `l` є редакцією
v2 з глибшою головою пропозицій областей і головою рамок із чотирьох згорток
замість голови `m`. Варіанти `n` і `s` поступаються правильністю заради
легшого бекбона.

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому проводилося
навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Faster R-CNN експортується лише до ONNX із розміром батча 1. Експортований граф
зберігає всередині початковий етап зміни розміру, тому LibreYOLO примусово
вмикає `dynamic=True` незалежно від переданого значення, щоб граф залишався
коректним для неквадратних джерел. Експортований файл `.onnx` знову
завантажується через `LibreYOLO()` за суфіксом і повертає той самий об'єкт
`Results`.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
