---
title: Оцінювання пози
seo_title: Оцінювання пози в LibreYOLO
description: >-
  Передбачайте ключові точки для кожного екземпляра в LibreYOLO: сімейства для
  цієї задачі, формат міток і виклики передбачення, навчання, валідації та
  експорту.
lead: >-
  Оцінювання пози визначає розташування кожного екземпляра й повертає для нього
  впорядкований набір іменованих ключових точок, тому результат містить
  внутрішню структуру об'єкта, а не лише його межі. Ключ задачі: pose.
keywords:
  - оцінювання пози python
  - детекція ключових точок
  - модель пози людини
  - ключові точки COCO
  - OKS mAP
  - навчити модель pose
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Суфікс -pose у назві файлу вибирає голову ключових точок, тому
        аргумент

        # task не потрібний.

        model = LibreYOLO("LibreECs-pose.pt")

        result = model(SAMPLE_IMAGE, save=True)


        print(result.keypoints.xy.shape)   # координати в пікселях (N, K, 2)

        print(result.boxes.xyxy.shape)     # (N, 4), ті самі N екземплярів
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Лише видимі ключові точки
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)

        kpts = result.keypoints


        # .has_visible виводиться з третього стовпця ключових точок і має

        # значення true для всіх точок, коли контрольна точка передбачає лише
        (x, y).

        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Натомість top-down
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # HRNet використовує top-down: спочатку вона вирізає кожну людину. Якщо
        джерело

        # людей не задано, модель поєднується з детектором LibreYOLO9t і записує
        вибір у лог.

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yaml містить вбудований скрипт завантаження, тому потрібен
        # явний дозвіл, якщо даних ще немає локально.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Власний датасет
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml має оголошувати kpt_shape, а рядки міток повинні містити
        # точно 5 + K * D полів.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() повертає звичайний словник, а не об'єкт.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика визначає маршрут за суфіксом файлу, тому експортований
        артефакт

        # завантажується як контрольна точка й повертає той самий об'єкт
        Results.

        model = LibreYOLO("LibreECs-pose.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Визначення

Оцінювання пози повертає структуру, а не лише межі. Кожен екземпляр так само отримує
рамку, клас і оцінку, а також `K` ключових точок у фіксованому порядку, тому індекс 5
означає ту саму частину тіла для кожного екземпляра й зображення. Набір міток визначає
цей порядок; у результаті немає назв, за якими можна ідентифікувати ключові точки.

`pose` є канонічним ключем задачі, а суфікс `-pose` у назві файлу контрольної точки
вибирає її, тому під час завантаження опублікованих ваг параметр `task=` не потрібний.

Метод `predict()` заповнює `result.keypoints` разом із `result.boxes`. Поле `.data`
має форму `(N, K, 2)` або `(N, K, 3)` і вирівняне за рядками з рамками, тому екземпляр
`i` в одному об'єкті відповідає екземпляру `i` в іншому. `.xy` вибирає координати
в пікселях, а `.xyn` нормалізує їх за розміром початкового зображення. `.conf` містить
третій стовпець, якщо контрольна точка передбачає його, і `None`, якщо ні. `.has_visible`
є отриманою з нього булевою маскою, усі значення якої дорівнюють true, коли третього
стовпця немає.

До такого результату ведуть дві архітектури. Одностадійна модель передбачає рамки
й ключові точки за один прохід. Модель top-down спочатку запускає детектор, вирізає
кожен екземпляр і регресує ключові точки всередині вирізаного фрагмента, тому її
правильність залежить від установленого перед нею детектора.

## Моделі

Три сімейства підтримують і навчання, і передбачення:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) і
[YOLO-NAS](/docs/models/yolo-nas), усі одностадійні. Для RF-DETR потрібне власне
доповнення `pip install "libreyolo[rfdetr]"`. RF-DETR і EdgeCrafter постачаються
з опублікованими контрольними точками пози, а донавчати обидві моделі можна на
однокласових датасетах лише з людьми. Голова ключових точок EdgeCrafter фіксується
під час створення й відхиляє датасет з іншою кількістю точок, тоді як RF-DETR
повторно ініціалізує свою голову під одну людину. YOLO-NAS отримує ваги з власної
CDN Deci.AI за некомерційною ліцензією, а LibreYOLO не публікує жодних ваг цього
сімейства. Голова пози також перебудовується для нової кількості ключових точок.
Це єдине з трьох сімейств, у якому кількість класів не зафіксована на одному,
тому воно підходить для багатокласового чи нелюдського скелета, наприклад пози тварин.

[HRNet](/docs/models/hrnet) є варіантом top-down. Модель виконує передбачення,
валідацію та експорт, а її `train()` породжує `NotImplementedError`. Якщо джерело
людей не вказано, вона автоматично поєднується з детектором LibreYOLO9t;
`cropped=True` розглядає все зображення як один екземпляр, `person_boxes=` приймає
вже наявні рамки, а `person_detector=` визначає інший детектор.

[SenseNova-Vision](/docs/models/sensenova-vision) також повертає ключові точки.
Це генеративна модель, керована підказками, із власною фабрикою `LibreVLM` і власним
доповненням. Якщо словник не задано, `set_task("pose")` використовує категорію людини.
Її ваги дозволено використовувати лише в некомерційних цілях, а затримка на зображення
значно вища, ніж у спеціалізованої голови пози, бо кожне передбачення потребує
декодування дифузійною моделлю.

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються локально.

<code-tabs name="predict" />

Кількість і порядок ключових точок є властивостями контрольної точки, а не бібліотеки,
тому модель, навчена на іншому скелеті, повертає інше значення `K` та інше значення
кожного індексу. Вміст третього стовпця ключових точок також залежить від контрольної
точки: EdgeCrafter записує туди сталу величину, а не оцінку для окремої точки, і взагалі
не має голови рамок, тому кожна рамка пози цієї моделі є обмежувальним прямокутником
ключових точок відповідного екземпляра. Докладніше про джерела, потокову обробку та
роботу з результатами див. у розділі [передбачення](/docs/predict).

## Формат датасету

Структура відповідає структурі для виявлення: кожне зображення має один файл міток
`.txt`, який знаходять заміною `images` на `labels` у шляху до зображення та зміною
розширення.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Рядок виявлення доповнюється ключовими точками:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Кількість полів точно дорівнює `5 + K * D`, де `D` є другим значенням `kpt_shape`.
Координати рамок і ключових точок є нормалізованими числами з рухомою крапкою відносно
ширини й висоти початкового зображення. Видимість `v`, наявна лише за значення `D` 3,
дорівнює `0`, `1` або `2`.

YAML доповнює спільний контракт двома ключами:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

Ключ `kpt_shape` обов'язковий і має значення `[K, 2]` або `[K, 3]`. Необов'язковий
ключ `flip_idx` є перестановкою `0..K-1`, яка для кожної ключової точки задає її
індекс після горизонтального віддзеркалення, завдяки чому ліве зап'ястя залишається
лівим зап'ястям. Якщо його пропустити, горизонтальне віддзеркалення ключових точок
вимикається, щоб не застосовувати його з неправильним порядком індексів.

## Навчання

<code-tabs name="train" />

Навчання продовжується з опублікованої контрольної точки `-pose`, яка вже містить
голову ключових точок. Задача визначається із завантаженої контрольної точки, а не
з прапорця, переданого під час навчання, тому контрольна точка виявлення не перетворюється
на запуск для пози за запитом. Значення `kpt_shape` у YAML має точно відповідати голові
EdgeCrafter, оскільки її зафіксовано під час створення, тоді як RF-DETR і YOLO-NAS
змінюють розмір голови для іншої кількості точок. Відомості про датасети, аугментацію,
роботу з кількома GPU й логери див. у розділі [навчання](/docs/train).

## Валідація

Метод `val()` повертає звичайний словник із ключами `metrics/`. Оцінювання виконується
за правилами COCO для ключових точок на основі Object Keypoint Similarity, що зважує
похибку відстані кожної ключової точки за масштабом екземпляра й допуском для окремої
точки, тому для ключових точок ця метрика відіграє роль IoU для рамок. Для неї потрібен
пакет `pycocotools`, який входить до базового встановлення.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` є основним показником, середньою точністю, усередненою
за порогами OKS від 0.50 до 0.95, і саме його навчання використовує для вибору найкращої
епохи. `metrics/keypoints_mAP50` і `metrics/keypoints_mAP75` є варіантами для окремих
порогів, а `metrics/keypoints_mAP_M` і `metrics/keypoints_mAP_L` розділяють середнє
значення за площею екземпляра на середні й великі; оцінювання ключових точок COCO
не визначає категорії малих екземплярів. Відповідні значення середньої повноти мають
ключі `metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`,
`metrics/keypoints_AR75`, `metrics/keypoints_AR_M` і `metrics/keypoints_AR_L`.
Кожен ключ цієї задачі має префікс `keypoints_`, тому ключів `mAP` для рамок,
які повертає детектор, тут немає.

## Експорт

<code-tabs name="export" />

Експортований артефакт знову завантажується через `LibreYOLO()` за суфіксом файлу,
тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає той самий
об'єкт `Results`. Підтримка форматів відрізняється між сімействами; матриця на кожній
сторінці моделі генерується з перевіреного набору, а не вводиться вручну. Формати,
їхні доповнення й обмеження описано в розділі
[експорту й розгортання](/docs/export).
