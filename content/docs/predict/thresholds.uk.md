---
title: Пороги та фільтрування
seo_title: 'conf, iou та max_det у LibreYOLO'
description: >-
  Що роблять conf, iou, max_det, classes і agnostic_nms під час передбачення та
  які сімейства ігнорують iou, бо не запускають власного NMS.
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
last_verified: 1.6.0
verification: >-
  Типові значення наведено за InferenceRunner.__call__ у
  libreyolo/models/base/inference.py. Поведінку NMS для сімейств перевірено за
  кожним модулем у libreyolo/postprocess/ і звірено з _is_nms_free_family у
  libreyolo/backends/base.py. Фільтрування класів взято з
  InferenceRunner._apply_classes_filter і _wrap_results. agnostic_nms взято з
  check_agnostic_nms і postprocess_max_det у libreyolo/utils/predict_args.py та
  кроку agnostic в InferenceRunner._wrap_results. Оброблення відкритого словника
  взято з NMS_THRESHOLD у libreyolo/models/openvocab/base.py. Типові значення
  валідації взято з BaseModel.val.
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
  agnostic:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        per_class = model(SAMPLE_IMAGE)

        agnostic = model(SAMPLE_IMAGE, agnostic_nms=True)


        # Перекривні рамки різних класів зводяться до тієї, що має найвищу
        оцінку.

        print(len(per_class.boxes), len(agnostic.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt agnostic_nms=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: fa0e6cee02b3e336
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

Валідація виявлення надає `metrics/best_conf` і пороги за класами з F1 при IoU 0.50. Використовуйте їх як виміряні початкові значення порогів розгортання; див. [валідацію](/docs/train/validation). Сегментація не надає цих ключів.

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

Три параметри під час передбачення роблять `iou` значущим навіть для предиктора
множини, оскільки всі три діють на рамки після завершення роботи моделі:

- `tiling=True` узгоджує перекривні тайли за допомогою NMS окремо для класів із порогом `iou`
- `augment=True` зливає віддзеркалені представлення за допомогою NMS окремо для класів із порогом `iou`
- `agnostic_nms=True` придушує перекривні рамки різних класів із порогом `iou`,
  див. [нижче](#agnostic_nms)

Перші два описано в розділі [Продуктивність інференсу](/docs/predict/performance).

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
їхньою оригінальною конфігурацією. SSD обмежує кількість до 200, а FCOS до власної межі виявлень на зображення.
Збільшення `max_det` понад ці значення не впливає на результат.

`max_det` застосовується централізовано, а не окремо в сімействі, у двох
випадках: тайловий інференс, де об'єднаний список обрізається після узгодження
тайлів, і виклики з `classes` або `agnostic_nms=True`, де обрізання відбувається
після фільтра чи придушення.

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
на передбачення класів, які потім відкидаються. Коли задано `classes`, сімейство
тримає бюджет кандидатів у 300 рамок або `max_det`, якщо він більший, а
обрізання до `max_det` відбувається після фільтра, тому малий `max_det` не
витісняє ваш клас. Експортовані бекенди використовують той самий бюджет, зокрема
пакетні рушії TensorRT. Зображення, на якому понад 300 рамок небажаних класів
мають вищі оцінки, усе одно може заповнити бюджет до появи вашого класу;
якщо це трапляється, збільште `max_det` понад 300.

## agnostic_nms

Власний NMS кожного сімейства враховує класи, тому за замовчуванням дві
перекривні рамки різних класів залишаються за будь-якого `iou`.
`agnostic_nms=True` додає один крок без урахування класів: з рамок, що
перекриваються понад `iou`, лишається тільки та, що має найвищу оцінку,
незалежно від їхніх класів.

<code-tabs name="agnostic" />

Крок працює на готових виявленнях сімейства, після фільтра `classes` і перед
обрізанням до `max_det`. Тому він поводиться однаково на предикторах множини, які
не запускають власного NMS, на експортованих бекендах і з `augment=True` та
`tiling=True`. Маски й ключові точки відкидаються разом зі своєю рамкою, а
орієнтовані рамки порівнюються за повернутим IoU. Він охоплює завдання, що
повертають рамки: detect, segment, pose і obb.

Для цього кроку сімейство тримає бюджет кандидатів у 300 рамок або `max_det`,
якщо він більший. Якщо `max_det` менший за 300, місця, які звільняє придушення,
заповнюють кандидати з нижчими оцінками. За 300 і більше, зокрема за типового
значення, ці місця залишаються порожніми.

Де він не застосовується:

- На стандартному шляху передбачення завдання без рамок спричиняє `ValueError`.
- `LibreEnsemble` і [HRNet](/docs/models/hrnet) спричиняють `NotImplementedError`.
- Моделі `LibreSAM` та [3D-детектори](/docs/tasks/3d-object-detection),
  WildDet3D, 3D-MOOD, FCOS3D і DetAny3D, не приймають такого аргументу й
  спричиняють `TypeError`.
- [L2CS-Net](/docs/models/l2cs), [PP-OCRv5](/docs/models/pp-ocrv5),
  [SAM 3D Body](/docs/models/sam-3d-body) і
  [LibreFaceRec](/docs/models/librefacerec) запускають власний шлях передбачення
  й ігнорують його.

`val(agnostic_nms=True)` застосовує той самий крок до валідації detect і segment;
див. [Валідація та метрики](/docs/train/validation).

## Що відхиляє predict

Два аргументи спричиняють помилку замість попередження: `visualize` та `embed`
спричиняють `NotImplementedError`. Для ембедінгів завантажте модель із
`task="embed"` і викликайте `predict` або `embed` як звичайно.
`agnostic_nms=True` спричиняє помилку там, де не застосовується, як перелічено
вище.

Будь-який нерозпізнаний аргумент спричиняє `TypeError` із переліком підтримуваних
параметрів, тому помилка в написанні виявляється негайно, а не ігнорується.

Наведені аргументи приймаються, спричиняють попередження та відкидаються:
`boxes`, `dnn`, `half`, `line_width`, `retina_masks`, `show_conf`,
`show_labels` і `verbose`.
