---
title: D-FINE
families:
  - dfine
seo_title: 'D-FINE: донавчання, валідація та експорт за ліцензією MIT'
description: >-
  Використання D-FINE у LibreYOLO для виявлення об'єктів і сегментації
  екземплярів. Встановлення, передбачення, донавчання, валідація та експорт із
  кодом за ліцензією MIT.
lead: >-
  Трансформер виявлення, який переформульовує регресію рамок як розподіл
  імовірностей для кожного краю рамки з уточненням між шарами декодера.
  LibreYOLO підтримує його для виявлення та сегментації екземплярів.
keywords:
  - D-FINE
  - трансформер виявлення
  - виявлення об'єктів у реальному часі
  - сегментація екземплярів
  - fine-grained distribution refinement
  - DETR
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDFINEn.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Сегментація екземплярів
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суфікс -seg у назві файлу вибирає голову масок, тому аргумент
        # завдання тут не потрібен.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDFINEn.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Сегментація екземплярів
      language: bash
      code: >
        # Продовжує навчання з опублікованих ваг сегментації, зокрема голови
        масок.

        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Сегментація з ваг виявлення
      language: bash
      code: >
        # Ваги виявлення не містять голови масок, тому це явне перенесення:

        # голова починає без навчання й стає корисною лише після нього. Саме
        запит

        # task=segment тут дозволяє перенесення.

        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Сегментація екземплярів
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # маски
        print(metrics["metrics/mAP50-95(B)"])   # рамки
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640

        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640
        half=True
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreDFINEn.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 0216631a26185524
---

## Встановлення

D-FINE не потребує додаткових пакетів. Усі його імпорти входять до базового
встановлення.

```bash
pip install libreyolo
```

Винятком є донавчання адаптерів із `lora=True`, для якого потрібен додатковий
пакет `lora`.

```bash
pip install "libreyolo[lora]"
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face та кешуються
локально.

<code-tabs name="predict" />

Повертається той самий об'єкт `Results`, що й для кожного сімейства, тому для
заміни детектора достатньо змінити один рядок. Назва файлу із `-seg` самостійно
вибирає завдання сегментації, після чого `result.masks` містить маски екземплярів
разом із рамками. Параметри `conf` і `max_det` фільтрують вибір запитів; `iou`
приймається для сумісності API, але не впливає на результат, оскільки декодер є
множинним предиктором без етапу NMS. Джерела, потокове оброблення та роботу з
результатами описано на сторінці [передбачення](/docs/predict).

## Варіанти

Доступні п'ять розмірів. Усі вони працюють з однаковою роздільною здатністю
вхідних даних, тому таблиця розрізняє їх за кількістю параметрів і правильністю.

<benchmark-table task="detect" />

<va-embed />

Сегментація повторно використовує бекбон, кодер і декодер виявлення та додає
голову масок, тому контрольна точка `-seg` приймає ті самі аргументи, що й
відповідна контрольна точка виявлення. Сімейство RT-DETRv4 у LibreYOLO реалізовано
як підклас обгортки D-FINE: воно успадковує цю лінію декодера, а потім знову
обмежує перелік завдань виявленням, оскільки не містить голови масок.

## Навчання

Для обох завдань навчання починається з опублікованої контрольної точки.

<code-tabs name="train" />

Без додаткових параметрів тренер виконує 132 епохи з `lr0=2e-4`, `amp=False`,
батчем 16 і ранньою зупинкою після 50 епох без покращення. Ваги виявлення можна
використати як початкову точку для навчання сегментації, але лише як явне
перенесення, оскільки голова масок починає без навчання й інакше повертала б
беззмістовні маски. Саме передавання `task=segment` до CLI дозволяє це. Шлях
Python має суворіші обмеження: потрібно безпосередньо створити `LibreDFINE` з
`allow_detect_to_segment_transfer=True`, оскільки фабрика `LibreYOLO()` не
приймає такого аргументу, а пряме створення не завантажує файли, тому файл ваг
уже має бути на диску.

Параметр `lora=True` застосовується до виявлення. Навчання сегментації відхиляє
його та натомість указує на `freeze='backbone'`, оскільки голову масок не
перевірено з адаптерами. На Apple silicon тренер переносить весь запуск на CPU:
зворотний прохід бінованого матричного множення Integral спричиняє помилку
компіляції Metal. Інференс на MPS не зазнає впливу.

Датасети, аугментацію, кілька GPU та системи журналювання описано на сторінці
[навчання](/docs/train).

## Валідація

Метод `val()` повертає словник із ключами за назвами метрик і виводить результати
для кожного класу, якщо `verbose` залишається ввімкненим.

<code-tabs name="val" />

Для контрольної точки `-seg` звичайний ключ `metrics/mAP50-95` містить оцінку
масок, а той самий запуск також повідомляє рамки під `(B)` і маски під `(M)`,
тому обидва результати доступні за один прохід.

## Експорт

<export-matrix />

Експортований артефакт повторно завантажується через `LibreYOLO()` за суфіксом
файлу, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий `Results`. Шляхи OpenVINO, Paddle, MNN і Core AI експортують із
фіксованим полотном, а не динамічними формами. На сторінці
[Експорт](/docs/export) наведено аргументи, які приймає кожен формат, і додаткові
пакети, потрібні деяким із них.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

Ваги сегментації мають друге джерело: декодер масок, зіставлення масок і функцію
втрат масок взято з ArgoHA/D-FINE-seg, також за ліцензією Apache-2.0, а його
супровідник схвалив повторне використання із зазначенням авторства.

</provenance-box>

## Цитування

<citation-block />

