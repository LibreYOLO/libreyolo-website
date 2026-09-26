---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: локалізація точок, навчання та експорт у LibreYOLO'
description: >-
  Запуск FOMO (Faster Objects, More Objects) у LibreYOLO: крихітного детектора з
  локалізацією точок для підрахунку багатьох малих об'єктів. Встановлення,
  передбачення, навчання та експорт.
lead: >-
  FOMO є сітковим локалізатором точок: кожна клітинка сітки низької роздільної
  здатності класифікується як фон або центр об'єкта без регресії обмежувальної
  рамки. LibreYOLO підтримує його для завдання точок.
keywords:
  - FOMO
  - Faster Objects More Objects
  - локалізація точок
  - виявлення центроїдів
  - виявлення крихітних об'єктів
  - edge AI
  - детекція на MCU
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Ваги LibreFOMO не завантажуються автоматично (див. розділ «Контрольні
        точки» нижче).

        # Укажіть тут контрольну точку, яку вже завантажено локально.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # Потрібно передати imgsz: типовим значенням CLI є 640, а контрольна
        точка s

        # приймає лише свій нативний розмір 96.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Встановлення

FOMO не потребує нічого, крім базового пакета.

```bash
pip install libreyolo
```

## Передбачення

На відміну від усіх інших сімейств на цьому сайті, ваги LibreFOMO не
завантажуються автоматично: `LibreYOLO("LibreFOMOs-point.pt")` шукає цей файл на
диску й спричиняє `ValueError` із його назвою замість завантаження з Hugging Face.
Спочатку завантажте контрольну точку з
[організації LibreYOLO](https://huggingface.co/LibreYOLO) та завантажте її за
локальним шляхом або навчіть власну (див. розділ «Навчання» нижче).

<code-tabs name="predict" />

Результат містить `points` замість `boxes`: кожен рядок має вигляд
`x, y, class, confidence` і доступний як `result.points.data` або через засоби
доступу `.xy`, `.xyn`, `.cls` і `.conf`. Поріг `iou` не задається, оскільки немає
рамок для пригнічення; `predict(..., nms_radius=1)` визначає, на скільки клітинок
сітки мають відстояти два виявлення, щоб збереглися обидва, а назва файлу повинна
містити суфікс завдання FOMO `-point`, щоб завантажувач його розпізнав. Джерела,
потокове оброблення та роботу з результатами описано на сторінці
[передбачення](/docs/predict).

## Варіанти

Три розміри, `s`, `m` та `l`, використовують дедалі ширші бекбони в стилі
MobileNetV2 із відповідно більшими фіксованими роздільними здатностями вхідних
даних, кожен з однією головою класифікації 1x1. Це сімейство не має тут таблиці
бенчмарків; розмір файлу контрольної точки в таблиці нижче наразі є
найзрозумілішим показником для кожного розміру.

## Навчання

<code-tabs name="train" />

`imgsz` не можна вибирати довільно: типовим є нативний розмір завантаженої
контрольної точки, а передавання іншого значення спричиняє `ValueError` із назвою
очікуваного розміру. Ці розміри дорівнюють 96 для `s`, 192 для `m` та 224 для
`l`. Типовим значенням `imgsz` у CLI є 640, тому команда `libreyolo train`
повинна явно задавати значення відповідно до контрольної точки.

Без інших параметрів тренер виконує 40 епох із батчем 32, Adam та `lr0=3e-4`,
без спаду ваг, а клас переднього плану отримує вагу у 100x більшу за фон у
перехресній ентропії кожної клітинки, оскільки майже кожна клітинка сітки у
типовій сцені є фоном. EMA та змішану точність типово вимкнено, а геометричні й
колірні аугментації, які використовуються в інших частинах LibreYOLO, не
застосовуються: mosaic, mixup, зміна HSV, віддзеркалення, обертання, перенесення
та зсув мають нульові значення.

Саме цим шляхом опубліковані контрольні точки LibreFOMO навчено з нуля на COCO.

Датасети й системи журналювання описано на сторінці [навчання](/docs/train).

## Валідація

Метод `val()` виконує диспетчеризацію до валідатора на рівні сітки, створеного
для цього сімейства. Разом зі спільними для інших завдань точок ключами
зіставлення точок `metrics/precision`, `metrics/recall` та `metrics/mAP@` він
перебирає пороги упевненості й значення `nms_radius` та публікує найкраще
поєднання F1 у `metrics/grid_F1`, `metrics/grid_precision`,
`metrics/grid_recall` і `metrics/grid_mean_distance`, а також поріг і радіус,
які його створили, у `decode/threshold` і `decode/nms_radius`.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт повторно завантажується через `LibreYOLO()` за суфіксом
файлу, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий `Results`. Запуск графа безпосередньо в середовищі виконання без
установленої LibreYOLO також підтримується, але тоді попередню та подальшу
обробку потрібно написати самостійно.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства. Жоден із них не завантажується
автоматично: отримайте потрібний файл із пов'язаної сторінки Hugging Face і
передайте його локальний шлях до `LibreYOLO()`.

<checkpoint-table />

## Ліцензування

<provenance-box>

Немає репозиторію початкового коду FOMO, на який можна послатися: Edge Impulse
описує метод у дописі блогу та документації продукту, але не випустила код
навчання чи інференсу FOMO. Наведені тут архітектура й навчання є власною
реалізацією LibreYOLO за опублікованим описом, а опубліковані контрольні точки
LibreFOMO навчено з нуля на COCO. Тому і код, і ці ваги є власністю LibreYOLO
за ліцензією MIT. Назва FOMO та описаний нею метод залишаються власністю
Edge Impulse.

</provenance-box>

