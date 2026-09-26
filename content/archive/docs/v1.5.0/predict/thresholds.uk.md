---
title: Пороги та фільтрування
seo_title: 'conf, iou та max_det у LibreYOLO'
description: >-
  Що насправді роблять conf, iou, max_det і classes під час передбачення, які
  сімейства ігнорують iou через відсутність NMS і чому agnostic_nms не виконує
  жодної дії.
lead: >-
  Чотири аргументи визначають, які передбачення залишаються: conf, iou, max_det
  і classes. Лише два з них застосовуються до кожного сімейства, оскільки
  предиктор множини декодує фіксований набір запитів і ніколи не запускає NMS.
keywords:
  - yolo поріг conf
  - поріг iou nms
  - max_det
  - фільтрування класів детектора python
  - agnostic nms
  - detr без nms
  - поріг впевненості детекції
  - фільтр класів під час інференсу
last_verified: 1.5.0
verification: >-
  Типові значення наведено за InferenceRunner.__call__ у
  libreyolo/models/base/inference.py. Поведінку NMS для сімейств перевірено за
  кожним модулем у libreyolo/postprocess/ і звірено з _is_nms_free_family у
  libreyolo/backends/base.py. Фільтрування класів взято з
  InferenceRunner._apply_classes_filter і _wrap_results. Стан agnostic_nms
  перевірено за NOOP_PREDICT_KWARGS у libreyolo/utils/predict_args.py.
  Оброблення відкритого словника взято з NMS_THRESHOLD у
  libreyolo/models/openvocab/base.py. Типові значення валідації взято з
  BaseModel.val.
snippets:
  basic:
    - label: Чотири аргументи
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # залишати передбачення з цією або вищою оцінкою
            iou=0.45,       # поріг перекриття NMS там, де виконується NMS
            max_det=300,    # обмеження на зображення
            classes=None,   # або список ідентифікаторів класів
        )
        print(len(result.boxes))
    - label: Перебирання conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Фільтрування за певними класами
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        # Ідентифікатори класів індексують model.names. У COCO 0 відповідає
        person.

        result = model(SAMPLE_IMAGE, classes=[0])


        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Пошук ідентифікатора за назвою
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou для сімейства без NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # RF-DETR декодує фіксований набір запитів, тому iou тут нічого не
        змінює.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # Кількість однакова в обох випадках. Діють лише conf і max_det.

        print(len(loose.boxes), len(tight.boxes))
source_hash: 0b978963c356027d
---

## Чотири аргументи

| Аргумент | Типове значення | Застосовується до |
|---|---|---|
| `conf` | `0.25` | Кожного сімейства |
| `iou` | `0.45` | Сімейств, що виконують немаксимальне придушення |
| `max_det` | `300` | Кожного сімейства |
| `classes` | `None` | Кожного сімейства |

<code-tabs name="basic" />

Два з них універсальні, а два ні. Це найважливіша відомість перед налаштуванням
будь-яких параметрів.

Для валідації навмисно використано інші типові значення: `val()` працює з
`conf=0.001` і `iou=0.6`, оскільки середня точність обчислюється за повною
кривою точність-повнота, яку поріг 0.25 обрізав би.

## conf

`conf` є оцінкою, нижче якої передбачення відкидається. Аргумент застосовується
до кожного сімейства, включно з тими, які ніколи не запускають NMS. Саме його
варто налаштовувати першим, коли виявлень забагато або замало.

Типове значення `0.25` підходить для перегляду зображень. Для передавання в
подальшу систему зазвичай потрібне вище значення, а для вимірювання правильності
потрібне набагато нижче.

## iou

`iou` є порогом перекриття, вище якого немаксимальне придушення вилучає рамку з
нижчою оцінкою з двох рамок одного класу. Він має значення, лише якщо сімейство
взагалі виконує придушення.

Предиктор множини декодує фіксовану кількість запитів і вибирає запити з
найвищими оцінками. Дублікати придушуються всередині архітектури під час
навчання, а не на етапі постоброблення, тому порога для налаштування немає.
Наведені сімейства приймають `iou` задля узгодженості API та ігнорують його:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR і наскрізна голова YOLOv9.
Варіанти, побудовані на цих декодерах, успадковують таку поведінку.

<code-tabs name="nmsfree" />

У рядках документації постоброблення більшості з них це зазначено, але під час
виконання попередження не виникає. Тому перебір `iou` для RF-DETR дає пласку
лінію, а не помилку. Faster R-CNN і Mask R-CNN трохи відрізняються: обидві моделі
вже виконали NMS усередині себе за фіксованим порогом оригінальної реалізації,
для зміни якого через `iou` немає підтримуваного способу.

Ці сімейства використовують аргумент: YOLOv1 до YOLOv4, YOLOv7, YOLOv9, YOLOX,
YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet і SSD.

Два параметри під час передбачення роблять `iou` значущим навіть для предиктора
множини, оскільки обидва зливають рамки після завершення роботи моделі:

- `tiling=True` узгоджує перекривні тайли за допомогою NMS окремо для класів із порогом `iou`
- `augment=True` зливає віддзеркалені представлення за допомогою NMS окремо для класів із порогом `iou`

Обидва описано в розділі [Продуктивність інференсу](/docs/predict/performance).

Для детекторів із відкритим словником діє окреме правило. Сімейство, процесор
якого запускає NMS, оголошує власний типовий поріг і враховує `iou`, як у випадку
OMDet-Turbo. Сімейства, що нічого не придушують, Grounding DINO, OWLv2 та OV-DEIM,
виводять попередження, коли передано `iou`. Це єдине таке попередження в бібліотеці.

## max_det

`max_det` обмежує кількість передбачень, що повертаються для одного зображення.
Він застосовується всюди, але через різні механізми: сімейство NMS обрізає
результат після придушення, а предиктор множини використовує його як розмір
вибірки top-k.

Деякі сімейства встановлюють нижчу межу за запитану, оскільки так передбачено
їхньою оригінальною конфігурацією. SSD обмежує кількість до 200, сегментація
екземплярів RTMDet до 100, а FCOS до власної межі виявлень на зображення.
Збільшення `max_det` понад ці значення не впливає на результат.

Єдине місце, де `max_det` застосовується централізовано, а не окремо в сімействі,
це тайловий інференс: об'єднаний список обрізається після узгодження тайлів.

## Фільтрування класів

<code-tabs name="classes" />

`classes` приймає список ідентифікаторів класів і залишає лише передбачення,
клас яких є в цьому списку. Ідентифікатори індексують `result.names`, і
найнадійніший спосіб отримати потрібний полягає в читанні `names` із результату,
а не в припущенні про порядок датасету.

Фільтрування виконується централізовано після постоброблення кожного сімейства
в єдиній точці, через яку проходить кожен шлях передбачення. Це має два важливі
наслідки. Функція працює для кожного сімейства, включно з сімействами без NMS.
Вона також фільтрує корисні дані, вирівняні з рамками, тому маски, ключові точки
й орієнтовані рамки скорочуються разом із ними та не втрачають відповідності.

У командному рядку `classes` приймає окреме ціле число, список або рядок зі
значеннями, розділеними комами:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Фільтрування не дає правильність без витрат. Модель усе одно витрачає ресурси
на передбачення класів, які потім відкидаються, а `max_det` застосовується
сімейством перед фільтром. Тому зображення з багатьма небажаними класами може
досягти межі до появи потрібного класу. Якщо це трапляється, зменште `conf` або
збільште `max_det`.

## agnostic_nms

`agnostic_nms` приймається й нічого не робить. Його передавання спричиняє
попередження про відсутність дії та сумісність із командним рядком, після чого
аргумент відкидається.

Режиму придушення без урахування класів немає. Кожен виклик NMS у бібліотеці
враховує класи, тому дві перекривні рамки різних класів залишаються за будь-якого
`iou`. Якщо це створює проблему, спочатку відфільтруйте через `classes` або
самостійно виконайте придушення між класами для `result.boxes`.

## Що відхиляє predict

Два аргументи спричиняють помилку замість попередження: `visualize` та `embed`
спричиняють `NotImplementedError`. Для ембедінгів завантажте модель із
`task="embed"` і викликайте `predict` або `embed` як звичайно.

Будь-який нерозпізнаний аргумент спричиняє `TypeError` із переліком підтримуваних
параметрів, тому помилка в написанні виявляється негайно, а не ігнорується.

Наведені аргументи приймаються, спричиняють попередження та відкидаються:
`agnostic_nms`, `boxes`, `dnn`, `half`, `line_width`, `retina_masks`,
`show_conf`, `show_labels` і `verbose`.
