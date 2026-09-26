---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: передбачення, навчання та експорт під ліцензією Apache-2.0'
description: >-
  Використовуйте YOLOX у LibreYOLO для виявлення об'єктів: установлення,
  передбачення, навчання, валідація та експорт під ліцензією Apache-2.0.
lead: >-
  YOLOX є одностадійним детектором без якорів із відокремленою головою
  класифікації та регресії, навченим із призначенням міток SimOTA. LibreYOLO
  підтримує його для виявлення.
keywords:
  - YOLOX
  - детекція об'єктів
  - детекція без якорів
  - відокремлена голова
  - SimOTA
  - виявлення об'єктів у реальному часі
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: На COCO
      language: bash
      code: >
        # Комплектний YAML COCO містить вбудований скрипт завантаження, тому
        потрібен

        # явний дозвіл, якщо датасет ще не доступний локально.

        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreYOLOXs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Встановлення

Для YOLOX не потрібні додаткові залежності понад базовий пакет.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. `conf` задає поріг упевненості, а
`iou` задає поріг NMS, застосований до трьох відокремлених масштабів
передбачення. Типи джерел, потокове передбачення та обробку результатів
описано в розділі [передбачення](/docs/predict).

## Варіанти

Шість розмірів мають спільний бекбон CSP і neck PAFPN. Два найменші, `n` і
`t`, працюють із меншою фіксованою вхідною роздільною здатністю, ніж інші
чотири; у таблиці бенчмарків нижче наведено точне значення для кожного.

<benchmark-table task="detect" />

<va-embed />

## Навчання

<code-tabs name="train" />

Без додаткових налаштувань тренер виконує 300 епох із `lr0=0.01`,
імпульсом SGD 0.9, прогріванням протягом 5 епох і вимкненими аугментаціями
mosaic та mixup на останніх 15 епохах. `train()` також приймає аргумент
`pretrained`, але його значення ніколи не читається всередині методу:
навчання завжди продовжується з ваг, з якими створено модель, тому
`pretrained=False` не ініціалізує мережу повторно.

`imgsz` типово має фіксоване значення з базової конфігурації навчання, а не
початкову роздільну здатність завантаженої контрольної точки. Це особливо
впливає на контрольні точки `n` і `t`: продовження навчання будь-якої з
них без явного `imgsz` перемикає її на більший типовий розмір замість
меншого опублікованого.

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
попередню та подальшу обробку потрібно реалізувати самостійно. Експорт CoreML
може вбудувати NMS у граф за допомогою `nms=True`; наразі цей прапорець
приймають лише YOLOX і YOLOv9.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
