---
title: YOLOv9
families:
  - yolo9
seo_title: 'YOLOv9: передбачення, навчання та експорт під ліцензією MIT'
description: >-
  Запускайте YOLOv9 у LibreYOLO, включно з наскрізною головою без NMS і головою
  з кроком 4 для малих об'єктів. Установлення, передбачення, навчання, валідація
  та експорт.
lead: >-
  Одностадійний згортковий детектор: один прохід оцінює щільну сітку рамок, а
  NMS усуває дублікати. LibreYOLO містить три його варіанти, один із яких не має
  етапу NMS.
keywords:
  - YOLOv9
  - YOLO9
  - виявлення об'єктів YOLOv9
  - детектор без NMS
  - наскрізна детекція об'єктів
  - виявлення малих об'єктів
  - programmable gradient information
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Без NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Той самий виклик, інша контрольна точка. Наскрізна голова повертає
        власні

        # передбачення з найвищими оцінками, тому NMS не запускається, а iou
        ігнорується.

        model = LibreYOLO("LibreYOLO9E2Es.pt")

        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)


        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Малі об'єкти
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # Варіант із кроком 4 не має власної контрольної точки COCO, тому
        вкажіть

        # базову контрольну точку виявлення: бекбон і neck завантажуються без
        змін,

        # а башта голови з кроком 4 починає з випадкової ініціалізації.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: На COCO
      language: bash
      code: |
        # Вбудований файл COCO yaml містить убудований скрипт завантаження, тому
        # потрібен явний дозвіл, якщо датасету ще немає локально.
        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: З NMS у графі
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Використання експортованого файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика визначає маршрут за суфіксом файла, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreYOLO9s.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## Установлення

YOLOv9 не потребує нічого додаткового до базового пакета.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` однаковий для всіх сімейств, тому для переходу на
інший детектор достатньо змінити один рядок. У базовій моделі та моделі з кроком
4 аргумент `conf` задає поріг упевненості, а `iou` задає поріг NMS. Наскрізна
модель не запускає NMS та ігнорує `iou`, тому її вивід визначають `conf` і
`max_det`. Докладніше про джерела, потокове оброблення та роботу з результатами
дивіться в розділі [передбачення](/docs/predict).

## Варіанти

Три варіанти мають спільний бекбон. Усі три призначені лише для виявлення
об'єктів і приймають однакові аргументи.

Базова модель робить передбачення на трьох масштабах ознак і усуває дублікати
рамок за допомогою NMS.

Наскрізна модель зберігає цю голову й додає поряд із нею гілку зіставлення
один-до-одного. Під час інференсу зчитується лише гілка один-до-одного, з якої
вибираються передбачення з найвищими оцінками, тому NMS не запускається.
Вибирайте цю модель, якщо середовище виконання для розгортання не має оператора NMS.

Модель із кроком 4 виводить на поверхню ще один рівень вище в бекбоні, розширює
neck до нього й робить передбачення на чотирьох масштабах замість трьох.
Додатковий масштаб призначений для об'єктів, що займають мало пікселів; єдину
опубліковану контрольну точку для нього навчено на аерофотознімках. До нього
можна переносити базові контрольні точки виявлення: бекбон і neck завантажуються
без змін, три попередньо навчені башти голови зміщуються на одну позицію вгору,
а башта з кроком 4 починає з випадкової ініціалізації.

<benchmark-table task="detect" />

<va-embed />

## Навчання

<code-tabs name="train" />

`pretrained` визначає початковий стан запуску. Передайте `True`, щоб завантажити
опубліковану контрольну точку для тієї самої моделі й розміру, або назву чи шлях
для будь-якої іншої. Тензори з невідповідною формою пропускаються, а не
відхиляються, і в журналі запуску зазначено кількість завантажених тензорів.
Тому контрольна точка, навчена з іншою кількістю класів, усе одно придатна як
відправна точка.

Модель із кроком 4 не має власної опублікованої контрольної точки COCO, тому
`True` для неї визначається як файл, якого не існує, і завантаження завершується
помилкою. Натомість укажіть базову контрольну точку виявлення.

Докладніше про датасети, аугментацію, кілька GPU та засоби журналювання дивіться
в розділі [навчання](/docs/train).

## Валідація

`val()` повертає словник ключів `metrics/`, що охоплюють точність, повноту,
mAP 50 і mAP 50-95, виміряні на будь-якому датасеті у форматі, на якому
проводилося навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Позначка стосується всіх трьох варіантів: якщо їхня підтримка відрізняється,
матриця показує найслабший рівень серед трьох.

Експортований артефакт завантажується назад через `LibreYOLO()` за суфіксом файла,
тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає той
самий об'єкт `Results`. Також підтримується запуск графа в чистому середовищі
виконання без установленої LibreYOLO, але тоді попереднє та подальше оброблення
потрібно реалізувати самостійно.

Для базової моделі виявлення частину постоброблення можна перемістити до графа.
`nms=True` під час експорту ONNX додає придушення всередину моделі, а першим
виводом стає фіксований тензор `(1, max_det, 6)`, рядки якого мають вигляд
`x1, y1, x2, y2, score, class` і після кількості виявлень доповнюються нулями.
Цей граф працює з батчем 1 і не містить динамічних осей. Наскрізна модель і
модель із кроком 4 не приймають цей прапорець.

Кожен формат установлює окремий додатковий пакет і приймає кілька власних
аргументів. Обидва аспекти описано на сторінці цього формату.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

Одна контрольна точка тут не має ліцензії MIT. Модель із кроком 4, навчена на
VisDrone2019-DET, успадковує умови CC BY-NC-SA 3.0 цього датасету: лише
некомерційне використання, поширення похідних матеріалів на тих самих умовах
і відсутність дії дозвільної ліцензії, під якою розповсюджується решта цього
сімейства. Вона передбачає класи повітряних знімків VisDrone, а не класи COCO.
Бібліотека виводить усю цю інформацію перед завантаженням файла.

</provenance-box>

## Цитування

<citation-block />
