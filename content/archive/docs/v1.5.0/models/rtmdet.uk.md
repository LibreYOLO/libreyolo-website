---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet у LibreYOLO: передбачення, навчання та експорт'
description: >-
  Запускайте RTMDet у LibreYOLO для виявлення об'єктів і RTMDet-Ins для
  сегментації екземплярів. Установлення, передбачення, навчання, валідація та
  експорт під ліцензією Apache-2.0.
lead: >-
  RTMDet є одностадійним детектором, що робить передбачення на основі одного
  точкового пріора для кожної позиції сітки без якорів через голову зі спільними
  згортками для всіх рівнів ознак. LibreYOLO підтримує його для виявлення
  об'єктів і сегментації екземплярів RTMDet-Ins.
keywords:
  - RTMDet
  - виявлення об'єктів RTMDet
  - сегментація екземплярів RTMDet-Ins
  - детектор без якорів
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Сегментація екземплярів
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суфікс -seg у назві файла вибирає голову масок RTMDet-Ins,
        # тому аргумент task тут не потрібен.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Сегментація екземплярів
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # маски
        print(metrics["metrics/mAP50-95(B)"])   # рамки
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Використання експортованого файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика визначає маршрут за суфіксом файла, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreRTMDets.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Установлення

RTMDet не потребує нічого додаткового до базового пакета.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` однаковий для всіх сімейств, тому для переходу на
інший детектор достатньо змінити один рядок. Назва файла із суфіксом `-seg`
автоматично визначає завдання RTMDet-Ins, після чого `result.masks` містить маски
екземплярів разом із рамками. `conf` задає поріг упевненості, а `iou` задає
поріг NMS. Докладніше про джерела, потокове оброблення та роботу з результатами
дивіться в розділі [передбачення](/docs/predict).

## Варіанти

П'ять розмірів, від `t` до `x`, використовують одну архітектуру зі спільною
роздільною здатністю вхідних даних. Для цього сімейства тут немає таблиці
бенчмарків: порівнюйте розміри за розміром файла контрольної точки в таблиці нижче.

## Навчання

<code-tabs name="train" />

Виявлення об'єктів навчається через `train()`. Компоненти QualityFocalLoss, GIoU
і DynamicSoftLabelAssigner перенесено з оригінального mmdetection, а прямий
прохід та експорт ONNX побітово еквівалентні йому. Постоброблення збігається з
виводом mmdet у межах 0.001 mAP на підмножинах val2017.

Згідно з власним рядком документації `train()`, не перевірено: збіжність
донавчання на малих датасетах, відповідність статті під час навчання з нуля,
поведінку на кількох GPU, пропускну здатність кешованих Mosaic і MixUp, суворе
перемикання двостадійного пайплайна оригінальної реалізації та параметричні
перевизначення спаду ваг, які обнуляють спад для параметрів нормалізації та зміщення.

Для RTMDet-Ins немає шляху навчання. Виклик `train()` для контрольної точки
із суфіксом `-seg` або з `task="segment"` спричиняє `NotImplementedError`;
сегментація екземплярів підтримує лише інференс і валідацію.

`train()` також приймає аргумент `pretrained`, але його значення всередині методу
ніколи не зчитується: навчання завжди продовжується з ваг, з якими створено модель,
тому `pretrained=False` не ініціалізує мережу повторно.

Якщо інші параметри не змінювати, навчання триває 300 епох з AdamW за
`lr0=0.004` і `weight_decay=0.05`, містить прогрівання протягом 1 епохи за
косинусним розкладом, а Mosaic і MixUp вимикаються на останні 20 епох.

Докладніше про датасети, аугментацію, кілька GPU та засоби журналювання дивіться
в розділі [навчання](/docs/train).

## Валідація

`val()` повертає словник ключів `metrics/`, що охоплюють точність, повноту,
mAP 50 і mAP 50-95, виміряні на будь-якому датасеті у форматі, на якому
проводилося навчання.

<code-tabs name="val" />

Для контрольної точки із суфіксом `-seg` звичайний ключ `metrics/mAP50-95`
містить оцінку масок. Той самий запуск також повертає рамки під ключем `(B)`
і маски під ключем `(M)`, тому обидва результати доступні за один прохід.

## Експорт

<export-matrix />

Моделі виявлення експортуються до більшості форматів, а моделі сегментації
екземплярів зараз не експортуються до жодного з них. Наведена вище матриця
відображає цю відмінність. Експортований артефакт виявлення завантажується назад
через `LibreYOLO()` за суфіксом файла, тому файл `.onnx` або `.engine` поводиться
як контрольна точка й повертає той самий об'єкт `Results`. Також підтримується
запуск графа в чистому середовищі виконання без установленої LibreYOLO, але тоді
попереднє та подальше оброблення потрібно реалізувати самостійно.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
