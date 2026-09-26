---
title: DEIM
families:
  - deim
seo_title: DEIM та DEIMv2 у LibreYOLO
description: >-
  Використання DEIM і DEIMv2 у LibreYOLO для виявлення об'єктів. Встановлення,
  передбачення, навчання, валідація та експорт, починаючи з розміру в пів
  мільйона параметрів.
lead: >-
  Трансформер виявлення, навчений із щільним взаємно однозначним зіставленням,
  який збігається за значно меншу кількість епох, ніж рецепти DETR, на яких він
  ґрунтується. LibreYOLO містить дві його версії, які розрізняються за
  завантаженою контрольною точкою.
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - трансформер виявлення
  - DETR
  - виявлення об'єктів
  - виявлення в реальному часі
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Відео
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Версія є частиною назви файлу, а фабрика виконує маршрутизацію за
        # контрольною точкою, тому обидві завантажуються однаково.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # Будь-яке джерело, яке приймає бібліотека: файл, каталог, URL-адреса,
        # індекс вебкамери, потік RTSP або список .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDEIMn.pt")


        # coco128.yaml завантажує вибірку зі 128 зображень під час першого
        використання. Для справжнього

        # запуску вкажіть у `data` YAML-файл власного датасету.

        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Якщо не задавати epochs, batch, imgsz та lr0, їхні значення беруться з
        опублікованого

        # рецепта для завантаженого розміру.

        model = LibreYOLO("LibreDEIMv2pico.pt")

        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Потрібен додатковий пакет lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() повертає звичайний словник, а не об'єкт
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: На COCO
      language: bash
      code: >
        # coco-val-only.yaml отримує 5000 зображень val2017 і пропускає

        # навчальну вибірку. Він містить вбудований скрипт завантаження, тому
        потребує

        # явного дозволу, якщо датасету ще немає локально.

        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Потрібен додатковий пакет onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreDEIMn.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## Встановлення

Жодна версія не потребує додаткових пакетів. Усі їхні імпорти входять до базового
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
заміни детектора достатньо змінити один рядок. Параметри `conf` і `max_det`
фільтрують декодування top-k за запитами та класами; етапу NMS для налаштування
немає, а `iou` приймається, але не використовується. Джерела, потокове оброблення
та роботу з результатами описано на сторінці [передбачення](/docs/predict).

## Варіанти

Версія 1 має п'ять розмірів з однаковим розміром вхідних даних. Версія 2 зберігає
ці п'ять назв і додає три менші, `atto`, `femto` та `pico`, причому перші дві
нативно працюють із меншим розміром вхідних даних, ніж решта. Отже, п'ять кодів
розміру існують в обох версіях і позначають різні моделі; версію записано в назві
файлу контрольної точки.

<benchmark-table task="detect" />

<va-embed />

Версія 1 зберігає архітектуру D-FINE та замінює її цільову функцію класифікації
на функцію втрат з урахуванням можливості зіставлення зі щільного взаємно
однозначного рецепта. Тому два сімейства мають майже всі однакові ключі словника
стану й розрізняються за метаданими контрольної точки. Версія 2 зберігає цей
контракт навчання та поєднує бекбони: HGNetv2 для розмірів нижче `s` і візуальний
трансформер DINOv3 з адаптером просторового налаштування для `s` і вище. Саме
цей бекбон додає другу ліцензію до чотирьох контрольних точок, тому перед
розповсюдженням однієї з них прочитайте розділ [ліцензування](#licensing).

## Навчання

Навчання починається з опублікованої контрольної точки. `pretrained` ніколи не
потрапляє до тренера: версія 1 попереджає про невідомий ключ та ігнорує його,
а версія 2 вилучає. Жодна з них не надає випадково ініціалізовану модель.

<code-tabs name="train" />

Для версії 1 передавайте `lr0` самостійно. Сигнатура її методу `train()` у Python
має типове значення `4e-4`, взяте з опублікованого рецепта COCO, а навчальна
конфігурація сімейства містить `1e-4` як типове значення донавчання. Саме це
нижче значення вибирає CLI, коли аргумент відсутній. У конфігурації зафіксовано
відповідне вимірювання: для розмірів батча, які реально використовуються під час
донавчання на малих датасетах, швидкість COCO помітно погіршувала перенесення.

Версія 2 самостійно визначає ці типові значення. Якщо не задавати `epochs`,
`batch`, `imgsz` та `lr0`, кожне з них читається з опублікованого рецепта для
завантаженого розміру, тому малі розміри навчаються з власною роздільною
здатністю вхідних даних без явної вказівки, а передане значення перевизначає
рецепт. Обмеження стосується аргументу `imgsz`: він має бути додатним числом,
кратним 32, інакше версія 2 спричиняє помилку до початку запуску.

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

## Експорт

<export-matrix />

Матриця охоплює обидві версії на одній сторінці: якщо їхня підтримка формату
відрізняється, у клітинці показано слабшу з двох, щоб можливості жодної
завантаженої версії не були перебільшені.

Експортований артефакт повторно завантажується через `LibreYOLO()` за суфіксом
файлу, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий `Results`.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>
Чотири розміри DEIMv2 від S і вище використовують бекбон DINOv3, тому їхні
репозиторії ваг мають одночасно ліцензію Apache-2.0 та ліцензію DINOv3 компанії
Meta, а LibreYOLO розповсюджує початковий код бекбона DINOv3 за тією самою угодою.
Решта сімейства, зокрема всі розміри DEIMv2 нижче S, має лише Apache-2.0.
</provenance-box>

## Цитування

<citation-block />

DEIMv2 описано в окремій статті з власним блоком цитування на сторінці
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation);
цитуйте його, якщо використовували контрольну точку версії 2.

