---
title: PicoDet
families:
  - picodet
seo_title: 'PicoDet у LibreYOLO: передбачення, навчання й експорт'
description: >-
  Запускайте PicoDet у LibreYOLO для мобільного виявлення об'єктів.
  Установлення, передбачення, навчання, валідація й експорт за Apache-2.0.
lead: >-
  PicoDet, це одноетапний детектор для мобільних і периферійних CPU: бекбон
  ESNet, шия CSP-PAN і спільна голова Generalized Focal Loss. LibreYOLO
  підтримує його для виявлення.
keywords:
  - PicoDet
  - PP-PicoDet
  - виявлення об'єктів
  - мобільне виявлення об'єктів
  - виявлення на периферії
  - ESNet
  - Generalized Focal Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz варто задати: стандартне значення CLI, 640, а рідне значення

        # контрольної точки s, 320.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: Використання експортованого файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика вибирає маршрут за суфіксом файла, тому експортований артефакт

        # завантажується як контрольна точка й повертає той самий об'єкт
        Results.

        model = LibreYOLO("LibrePICODETs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## Встановлення

PicoDet не потребує нічого понад базовий пакет.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повертається той самий об'єкт `Results`, що й у кожного сімейства, тому
заміна детектора потребує зміни одного рядка. `conf` задає поріг
упевненості, а `iou`, поріг NMS. Джерела, потокову обробку й роботу з
результатами описано в розділі [передбачення](/docs/predict).

## Варіанти

Три розміри, кожен із власною фіксованою роздільною здатністю входу: `s`
найменший, а `l` найбільший. Роздільна здатність зростає з розміром, тому
більші контрольні точки дорожчі на кожне зображення не лише через більшу
кількість параметрів.

<benchmark-table task="detect" />

<va-embed />

## Навчання

<code-tabs name="train" />

Складники втрат і призначувач відповідають рецепту upstream: VFL, DFL, GIoU
і SimOTA зі зважуванням за якістю класифікації та цілями VFL із динамічним
IoU. Інференс побітово відповідає upstream на тій самій контрольній точці.

Згідно з власною документацією `train()`, не перевірено збіжність на
повному датасеті, поведінку з кількома GPU та будь-яку аугментацію крім
горизонтального віддзеркалення. Контрольна точка `s` із рідним розміром 320
також не проходить стабільно поріг точності LibreYOLO на тестовому наборі з
30 зображень і двох класів для малих донавчань. Цей розмір краще підходить
для масштабу повного COCO.

`train()` також приймає аргумент `pretrained`, але його значення не
читається всередині методу: навчання завжди продовжується з ваг моделі, з
якими її створено, тому `pretrained=False` не ініціалізує мережу заново.
Якщо не задавати `imgsz` у Python, береться рідна роздільна здатність
завантаженої контрольної точки: 320 для `s`, 416 для `m` і 640 для
`l`. CLI завжди передає `imgsz` зі стандартним значенням 640, тому
задайте його там відповідно до контрольної точки.

За інших стандартних значень тренер виконує 300 епох SGD з `lr0=0.01`,
моментом 0.9, спадом ваг 4e-5, прогріванням протягом 1 епохи та косинусним
розкладом. Єдина аугментація, горизонтальне віддзеркалення.

Датасети, аугментацію, кілька GPU й логери описано в розділі
[навчання](/docs/train).

## Валідація

`val()` повертає словник ключів `metrics/` для precision, recall, mAP 50
і mAP 50-95, виміряних на будь-якому датасеті у форматі навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт завантажується через `LibreYOLO()` за суфіксом
файла, тому файл `.onnx` або `.engine` поводиться як контрольна точка й
повертає той самий `Results`. Запуск графа в окремому середовищі виконання
без установленої LibreYOLO також підтримується, але тоді попередню й
подальшу обробку потрібно написати самостійно.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

Перенесення LibreYOLO відповідає Bo396543018/Picodet_Pytorch, повторній
реалізації оригінального PP-PicoDet із PaddleDetection у PyTorch, з вилученим
mmcv й точним узгодженням кожної активації, щоб контрольні точки PaddlePaddle,
перетворені конвеєром Bo, завантажувалися без числового відхилення. Обидва
джерела мають ті самі умови Apache-2.0, що й автори статті.

</provenance-box>

## Цитування

<citation-block />

