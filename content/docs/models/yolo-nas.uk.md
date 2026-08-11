---
title: YOLO-NAS
families:
  - yolonas
seo_title: 'YOLO-NAS: передбачення, навчання та експорт у LibreYOLO'
description: >-
  Використовуйте YOLO-NAS у LibreYOLO для виявлення об'єктів та оцінювання пози.
  Ваги Deci.AI пропрієтарні й призначені лише для некомерційного використання,
  LibreYOLO не публікує жодних із них.
lead: >-
  Згортковий детектор, бекбон і neck якого створено пошуком архітектури Deci.AI
  та побудовано з блоків RepVGG, пристосованих до квантування. Ваги належать
  Deci.AI, ліцензовані лише для некомерційного використання, і LibreYOLO не
  публікує жодних із них.
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - виявлення об'єктів YOLO-NAS
  - оцінювання пози YOLO-NAS
  - детектор з урахуванням квантування
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Назва, якої ще немає на диску, завантажується з CDN Deci. Спочатку

        # виводяться умови ліцензії Deci; завантаження файла означає їх
        прийняття.

        model = LibreYOLO("LibreYOLONASs.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Поза
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суфікс -pose вибирає голову пози та її власний набір ваг.
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: З нуля
      language: python
      code: >
        from libreyolo import LibreYOLONAS


        # Контрольна точка Deci не використовується: модель починає з випадкових
        ваг,

        # тому результат запуску походить лише з ваших даних.

        model = LibreYOLONAS(None, size="s")

        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: На COCO
      language: bash
      code: |
        # Вбудований файл COCO yaml містить убудований скрипт завантаження, тому
        # потрібен явний дозвіл, якщо датасету ще немає локально.
        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Використання експортованого файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика визначає маршрут за суфіксом файла, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreYOLONASs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## Установлення

YOLO-NAS не потребує нічого додаткового до базового пакета.

```bash
pip install libreyolo
```

## Передбачення

Якщо контрольної точки із заданою назвою ще немає на диску, вона завантажується
з публічного CDN Deci, а не з організації LibreYOLO, де ці ваги не розміщено.
Перед початком передавання бібліотека один раз за процес виводить умови ліцензії
Deci, а перед відкриттям завантаженого файла перевіряє його SHA-256 за закріпленим
значенням. Дозволи за цими умовами описано в розділі [ліцензування](#licensing).

<code-tabs name="predict" />

Повернений об'єкт `Results` однаковий для всіх сімейств, тому для переходу на
інший детектор достатньо змінити один рядок. `conf` задає поріг упевненості,
а `iou` задає поріг NMS. Докладніше про джерела, потокове оброблення та роботу
з результатами дивіться в розділі [передбачення](/docs/predict).

## Варіанти

Виявлення об'єктів та оцінювання пози використовують ту саму архітектуру з
різними головами й приймають однакові аргументи. Розміри в таблиці нижче
стосуються виявлення; модель пози опубліковано в цих розмірах і ще в одному
меншому. Голова пози передбачає набір ключових точок COCO.

<benchmark-table task="detect" />

<va-embed />

## Навчання

<code-tabs name="train" />

Якщо `epochs`, `lr0` та `amp` не задано, їх значення визначаються окремо для
кожного завдання, тому запуск оцінювання пози починається з інших типових значень,
ніж запуск виявлення. Типово оптимізатором є AdamW. Кількість класів береться з
YAML датасету, а голову перебудовано відповідно до неї перед першою епохою. Для
голови пози кількість ключових точок обробляється так само, тому контрольну точку
пози COCO можна донавчити на скелеті іншого розміру.

Донавчання починається з ваг Deci, на які поширюється ліцензія Deci. Навчання
випадково ініціалізованої моделі взагалі не залучає контрольної точки Deci, як
показано в третьому фрагменті вище.

Докладніше про датасети, аугментацію, кілька GPU та засоби журналювання дивіться
в розділі [навчання](/docs/train).

## Валідація

`val()` повертає словник ключів `metrics/`, що охоплюють точність, повноту,
mAP 50 і mAP 50-95, виміряні на будь-якому датасеті у форматі, на якому
проводилося навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт завантажується назад через `LibreYOLO()` за суфіксом файла,
тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає той
самий об'єкт `Results`. Також підтримується запуск графа в чистому середовищі
виконання без установленої LibreYOLO, але тоді попереднє та подальше оброблення
потрібно реалізувати самостійно. Кожен формат установлює окремий додатковий пакет
і приймає кілька власних аргументів. Обидва аспекти описано на сторінці цього формату.

Експорт створює ще одну копію тих самих ваг в іншому контейнері. Експорт
контрольної точки Deci не змінює ні походження ваг, ні ліцензію, що на них поширюється.

<code-tabs name="export" />

## Контрольні точки

Перелік порожній. Ліцензія Deci забороняє повторне розповсюдження, тому організація
LibreYOLO не публікує ваг YOLO-NAS, а завантаження відбувається з іншого джерела:
назва у формі `LibreYOLONAS<size>.pt` або `LibreYOLONAS<size>-pose.pt` для пози
відповідає об'єкту в публічному CDN Deci.

Так можна отримати лише контрольні точки, SHA-256 яких закріплено в бібліотеці.
Для всього іншого операцію відхилено без відкриття неперевіреного стороннього
pickle. Такий файл потрібно завантажити вручну й передати як шлях. Файл, який уже
є на диску, завантажується за своїм шляхом без завантаження з мережі й перевірки
контрольної суми. Це також стосується файла Deci `.pth` під початковою назвою,
яку розпізнає завантажувач.

## Ліцензування

<provenance-box>

LibreYOLO не розміщує й не дзеркалює ці ваги: для цього сімейства в організації
LibreYOLO на Hugging Face нічого немає. Кожне автоматичне завантаження натомість
звертається до публічного CDN Deci, один раз за процес виводить умови Deci перед
початком і перевіряється за закріпленим SHA-256 до відкриття файла.

Альтернативою є навчання випадково ініціалізованої моделі. Оригінальна архітектура
має ліцензію Apache-2.0, а ця реалізація має ліцензію MIT, тому модель, навчена
таким способом на ваших даних, не походить від контрольної точки Deci.

</provenance-box>

## Цитування

YOLO-NAS випущено без статті. Нижче наведено запис, який просять використовувати
автори; він стосується SuperGradients, бібліотеки, у складі якої випущено модель.

<citation-block />
