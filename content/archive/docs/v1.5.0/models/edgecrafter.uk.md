---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: виявлення, оцінювання пози та сегментація в LibreYOLO'
description: >-
  Використання EdgeCrafter у LibreYOLO для виявлення, оцінювання пози та
  сегментації екземплярів. Встановлення, передбачення, валідація та експорт із
  кодом за ліцензією MIT.
lead: >-
  Компактний візуальний трансформер для щільного передбачення на
  edge-обладнанні, опублікований у першоджерелі як три споріднені моделі: ECDet,
  ECPose та ECSeg. LibreYOLO завантажує всі три як одне сімейство, а завдання
  визначається контрольною точкою.
keywords:
  - EdgeCrafter
  - ECDet
  - ECPose
  - ECSeg
  - компактний візуальний трансформер
  - виявлення об'єктів
  - оцінювання пози
  - сегментація екземплярів
  - інференс на edge-пристроях
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Поза
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суфікс -pose у назві файлу вибирає голову ключових точок, тому
        # аргумент завдання тут не потрібен.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Сегментація екземплярів
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: Поза
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Потрібен однокласовий датасет ключових точок, у data.yaml якого
        оголошено

        # kpt_shape, та imgsz із нативним розміром контрольної точки.

        model = LibreYOLO("LibreECs-pose.pt")

        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Сегментація екземплярів
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Потрібні полігональні мітки та imgsz із нативним розміром контрольної
        точки.

        model = LibreYOLO("LibreECs-seg.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Поза
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Сегментація екземплярів
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # маски
        print(metrics["metrics/mAP50-95(B)"])   # рамки
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreECs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## Встановлення

EdgeCrafter не потребує додаткових пакетів. Усі його імпорти входять до базового
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

Завдання визначається за назвою файлу, тому контрольна точка `-pose` або `-seg`
вибирає власну голову й не приймає аргументу завдання. Усі три повертають об'єкт
`Results`, який повертає кожне сімейство, із доданим `result.keypoints` для пози
та `result.masks` для сегментації. Поза охоплює один клас, людину, із 17 ключовими
точками COCO, а кількість фіксується під час побудови моделі. Вона не має голови
рамок, тому кожна рамка пози охоплює її власні ключові точки, а третій канал
ключової точки є сталою, а не оцінкою окремої точки.

Параметри `conf` і `max_det` фільтрують вибір запитів; `iou` приймається для
сумісності API, але не впливає на результат, оскільки всі три голови декодують
набір запитів без етапу NMS. Джерела, потокове оброблення та роботу з результатами
описано на сторінці [передбачення](/docs/predict).

## Варіанти

Доступні чотири розміри. Усі вони працюють з однаковою роздільною здатністю
вхідних даних, тому таблиця розрізняє їх за кількістю параметрів і правильністю.

<benchmark-table task="detect" />

<va-embed />

Першоджерело публікує ECDet, ECPose та ECSeg як три окремі моделі, а не одну
модель із трьома головами. Вони мають спільні бекбон ECViT і гібридний кодер та
відрізняються лише головою, тому LibreYOLO об'єднує їх в одне сімейство, а завдання
визначається назвою файлу контрольної точки. Отже, літера розміру означає однакові
бекбон і кодер для всіх трьох, а передбачення, валідація та експорт приймають
однакові аргументи незалежно від завантаженої моделі.

## Навчання

Усі три завдання навчаються через `train()`, який читає завдання із завантаженої
контрольної точки та вибирає відповідний тренер.

<code-tabs name="train" />

Для виявлення та сегментації перевірено еквівалентність інференсу з першоджерелом
до 1e-5, шар за шаром і для кожного розміру, а також виконання функції втрат і
одного кроку навчання на синтетичних вхідних даних. Згідно з документацією самого
`train()`, не перевірено збіжність повного донавчання, навчання на кількох GPU,
етап повторного завантаження найкращої моделі після припинення аугментації та
перенесення класів Objects365 у COCO. Шлях пози наслідує опублікований рецепт
DETRPose, тобто угорський алгоритм зіставлення за вартістю класу, L1 ключових
точок і OKS із контрастним знешумленням ключових точок; його збіжність також не
перевірено від початку до кінця.

Без додаткових параметрів тренер виконує 74 епохи з `lr0=5e-4` та ввімкненою
змішаною точністю за рецептом першоджерела: AdamW, плаский косинусний розклад,
EMA зі значенням 0.9999 і вхідні дані, нормалізовані за ImageNet. Поза й
сегментація потребують `imgsz` із нативним розміром контрольної точки, оскільки
їхня якірна сітка оцінювання будується під час створення моделі; інше значення
спричиняє помилку до початку запуску. Для пози також потрібен однокласовий датасет,
у `data.yaml` якого оголошено `kpt_shape`, а кількість ключових точок відповідає
голові.

Параметр `lora=True` застосовується лише до виявлення; для пози та сегментації
він спричиняє `ValueError`. На Apple silicon тренер залишає запуск на GPU та
надсилає на CPU одну операцію, зворотний прохід grid-sample у деформованій увазі,
яку PyTorch не реалізує в Metal.

Датасети, аугментацію, кілька GPU та системи журналювання описано на сторінці
[навчання](/docs/train).

## Валідація

Метод `val()` повертає словник із ключами за назвами метрик і виводить результати
для кожного класу, якщо `verbose` залишається ввімкненим.

<code-tabs name="val" />

Поза повідомляє метрики OKS ключових точок під `metrics/keypoints_*`. Сегментація
повідомляє маски під звичайним ключем `metrics/mAP50-95` і повторює обидва подання
за один прохід, рамки під `(B)` та маски під `(M)`.

## Експорт

<export-matrix />

Експортований артефакт повторно завантажується через `LibreYOLO()` за суфіксом
файлу, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий `Results`. Поза й сегментація експортуються з фіксованими вхідними
даними 640 на 640, а не динамічними формами; кілька цілей виявлення також мають
фіксоване полотно, зокрема OpenVINO, Paddle, MNN, ExecuTorch та Core AI. На
сторінці [Експорт](/docs/export) наведено аргументи, які приймає кожен формат,
і додаткові пакети, потрібні деяким із них.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />

