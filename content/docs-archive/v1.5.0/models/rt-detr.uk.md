---
title: RT-DETR
families:
  - rtdetr
seo_title: 'RT-DETR, RT-DETRv2 та RT-DETRv4 у LibreYOLO'
description: >-
  Використання RT-DETR, RT-DETRv2 та RT-DETRv4 у LibreYOLO для виявлення
  об'єктів, а також орієнтованих рамок у RT-DETRv2. Встановлення, передбачення,
  навчання, валідація й експорт із вагами за ліцензією Apache-2.0.
lead: >-
  Трансформер виявлення для інференсу в реальному часі: він декодує фіксований
  набір запитів замість щільної сітки, тому не виконує NMS. LibreYOLO містить
  три його версії, які розрізняються за завантаженою контрольною точкою, а
  версія 2 також підтримує орієнтовані рамки.
keywords:
  - RT-DETR
  - RT-DETRv2
  - RT-DETRv4
  - трансформер виявлення реального часу
  - DETR
  - виявлення об'єктів
  - виявлення орієнтованих обмежувальних рамок
  - OBB
  - DOTA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Відео
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Версія є частиною назви файлу, а фабрика виконує маршрутизацію за
        # контрольною точкою, тому всі три завантажуються однаково.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # Будь-яке джерело, яке приймає бібліотека: файл, каталог, URL-адреса,
        # індекс вебкамери, потік RTSP або список .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Орієнтовані рамки
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Лише версія 2. Суфікс -obb вибирає завдання, а орієнтованість
        контрольної точки

        # розпізнається за її власними тензорами, тому аргумент завдання

        # не потрібен. Ці ваги для DOTA v1.0, 15 класів аерофотознімків із 1024
        px.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        result = model("aerial.png", save=True)


        obb = result.obb

        print(obb.xywhr)     # (N, 5): cx, cy, w, h, радіани

        print(obb.xyxyxyxy)  # ті самі рядки у вигляді чотирьох кутових точок

        print(result.boxes.xyxy)  # осьові рамки, що їх охоплюють
    - label: 'Орієнтовані рамки, CLI'
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRTDETRr18.pt")


        # coco128.yaml завантажує вибірку зі 128 зображень під час першого
        використання. Для справжнього

        # запуску вкажіть у `data` YAML-файл власного датасету.

        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Потрібен додатковий пакет lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() повертає звичайний словник, а не об'єкт
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: На COCO
      language: bash
      code: >
        # coco-val-only.yaml отримує 5000 зображень val2017 і пропускає

        # навчальну вибірку. Він містить вбудований скрипт завантаження, тому
        потребує

        # явного дозволу, якщо датасету ще немає локально.

        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Орієнтовані рамки
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Орієнтована валідація виконує зіставлення за повернутим IoU, тому
        передбачення в

        # правильному місці з неправильним кутом вважається промахом.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        metrics = model.val(data="my-obb-dataset.yaml")


        print(metrics["metrics/mAP50-95(OBB)"])

        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Потрібен додатковий пакет onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Орієнтовані рамки
      language: bash
      code: >
        # ONNX і TorchScript є валідованими цілями для орієнтованого завдання

        # у FP32, з батчем 1 і на фіксованому полотні 1024 на 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreRTDETRr18.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## Встановлення

RT-DETR не потребує додаткових пакетів. Усі його імпорти входять до базового
встановлення, а додатковий пакет `rtdetr` є стабільною назвою, яка нічого не
додає.

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
заміни детектора достатньо змінити один рядок. Параметри `conf` і `max_det`
фільтрують декодування top-k за запитами та класами; етапу NMS для налаштування
немає, а `iou` приймається, але не використовується. Орієнтована контрольна точка
нативно заповнює `result.obb` і також заповнює `result.boxes` осьовими
прямокутниками, що їх охоплюють. Джерела, потокове оброблення та роботу з
результатами описано на сторінці [передбачення](/docs/predict).

## Варіанти

Доступні три версії та два завдання, а коди розміру не утворюють єдиного ряду.
Версія 1 називає розміри за бекбоном, ResNet або HGNetv2. Версія 2 повторно
використовує лише назви ResNet: версія 1 уже постачає два розміри HGNetv2, а
результати версії 2 для них були достатньо близькими, тому LibreYOLO не публікує
дубльованих ваг. Версія 4 використовує звичайний літерний ряд, який конфліктує
з назвами HGNetv2 версії 1, тому самого коду розміру недостатньо для ідентифікації
моделі. Версію записано в назві файлу контрольної точки.

<benchmark-table task="detect" />

<va-embed />

Версія 2 зберігає архітектуру та компонування словника стану версії 1 і змінює
спосіб вибірки деформованої уваги, тому вони розрізняються за метаданими
контрольної точки, а не формою. Версія 4 має інше походження: вона повторно
використовує архітектуру й тренер D-FINE, а її ваги отримано дистиляцією
фундаментальної візуальної моделі DINOv3 як учителя у модель HGNetv2 як учня.
У LibreYOLO `LibreRTDETRv4` є підкласом `LibreDFINE` із примусово вимкненою
головою масок, тому підтримує лише виявлення.

### Орієнтовані рамки у версії 2

Версія 2 є єдиною версією з другим завданням. Вона підтримує завдання `detect`
і `obb`, які не мають спільного графа чи ряду розмірів. Виявлення використовує
розміри ResNet із 640 px; орієнтоване виявлення використовує ряд HGNetv2, n, s,
m, l та x, із 1024 px, а розмір вхідних даних визначається для кожного завдання,
а не сімейства. Орієнтованість контрольної точки розпізнається за її власними
тензорами, п'ятикоординатними головами рамок і параметрами вибірки версії 2,
тому ваги `-obb` завантажуються в орієнтований граф без аргументу `task`, а
невідповідність спричиняє явну помилку замість непомітного переосмислення.

Опубліковано файли від `LibreRTDETRv2n-obb.pt` до `LibreRTDETRv2x-obb.pt`. Це
офіційні одномасштабні контрольні точки DOTA v1.0, перетворені у формат LibreYOLO,
із 15 класами аерофотознімків від літака та корабля до гавані й гелікоптера;
назви класів записано в контрольній точці. На відміну від виявлення, орієнтоване
завдання призначено лише для інференсу: передбачення, валідація й експорт працюють,
а `train()` для орієнтованої моделі спричиняє помилку. Відстеження та аугментація
під час тестування також не підтримують орієнтовані рамки. Завдання, формат міток
і метрики описано на сторінці
[Орієнтоване виявлення](/docs/tasks/oriented-detection).

## Навчання

Навчання починається з опублікованої контрольної точки. `pretrained` приймається,
а потім відкидається в усіх трьох версіях, тому `pretrained=False` не створює
випадково ініціалізованої моделі. Усе в цьому розділі стосується виявлення:
орієнтоване завдання версії 2 призначено лише для інференсу, і немає шляху
перенесення з ваг виявлення, оскільки два завдання використовують різні бекбони.

<code-tabs name="train" />

Швидкість навчання є ключовим аргументом, і кожна версія має власне типове
значення замість спільного для бібліотеки. Сигнатура методу `train()` у Python
читає його з навчальної конфігурації відповідної версії, а CLI визначає те саме
значення, якщо `lr0` не передано. Версії 1 і 2 також приймають `lr_backbone` та
типово встановлюють його в одну двадцяту `lr0` відповідно до оригінального
рецепта; версія 4 працює через тренер D-FINE, який масштабує групу параметрів
бекбона за допомогою `backbone_lr_mult`.

Залишайте `imgsz` із нативним розміром контрольної точки, якщо немає причини його
змінювати. Валідація й передбачення з іншими розмірами працюють з одним залишковим
ефектом: прямокутний розмір, кількість токенів якого відповідає нативному розміру,
усе одно повторно використовує ембединг, побудований для неправильного
співвідношення сторін.

Датасети, аугментацію, кілька GPU та системи журналювання описано на сторінці
[навчання](/docs/train).

## Валідація

Метод `val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50
і mAP 50-95, виміряними для будь-якого датасету у форматі, на якому виконувалося
навчання.

<code-tabs name="val" />

Рядки наведеної вище таблиці бенчмарків отримано за допомогою системи бенчмарків
LibreYOLO; у примітці під таблицею зазначено датасет і наведено посилання на
записи запусків.

Орієнтована валідація виконується тим самим викликом і повідомляє ті самі ключі,
а також чотири повторені ключі із суфіксом `(OBB)`. Для зіставлення
використовується повернутий IoU, а не IoU прямокутників, що охоплюють рамки,
тому помилка кута вважається промахом. Параметр `augment=True` відхиляється
для цього завдання.

## Експорт

<export-matrix />

Матриця охоплює все походження на одній сторінці: якщо підтримка формату в трьох
версіях відрізняється, у клітинці показано найслабшу, щоб можливості жодної
завантаженої версії не були перебільшені. Орієнтований рядок належить лише версії
2. ONNX і TorchScript валідовано для нього у FP32, з батчем 1 і фіксованим
полотном 1024 на 1024; OpenVINO, TensorRT та ExecuTorch перетворюються й повторно
завантажуються, але не досягли еквівалентності необробленого виходу для повного
набору запитів, тому найкращі рамки збігаються до частки пікселя, а хвіст
відхиляється.

Експортований артефакт повторно завантажується через `LibreYOLO()` за суфіксом
файлу, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий `Results`.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

Назва файлу містить версію, потім розмір, а потім завдання. Ваги виявлення мають
назви `LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` та
`LibreRTDETRv4<size>.pt`, усі для 640 px. Орієнтовані ваги існують лише для
версії 2 й додають суфікс завдання, від `LibreRTDETRv2n-obb.pt` до
`LibreRTDETRv2x-obb.pt`, усі для 1024 px і навчені на DOTA v1.0, а не COCO.

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />

У наведеному вище блоці подано цитування, яке автори публікують для виявлення
у версіях 1 і 2. Орієнтовані ваги версії 2 мають третє першоджерело, репозиторій
RiO-DETR за ліцензією Apache-2.0 на
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), звідки
походять контрольні точки DOTA; цитуйте цей проєкт, якщо використовували одну з
них. Версія 4 описана в окремій статті іншої групи й має власний блок цитування
на сторінці
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation);
цитуйте його, якщо використовували контрольну точку версії 4.

