---
title: Сегментація екземплярів
seo_title: Сегментація екземплярів у LibreYOLO
description: >-
  Сегментуйте окремі об'єкти в LibreYOLO: сімейства для цієї задачі, формат
  полігональних міток і виклики передбачення, навчання, валідації та експорту.
lead: >-
  Сегментація екземплярів локалізує кожен екземпляр об'єкта й повертає для нього
  попіксельну маску разом із рамкою, класом та оцінкою, які повертає детектор.
  Ключ задачі має назву segment.
keywords:
  - сегментація екземплярів python
  - передбачення маски об'єкта
  - навчання моделі сегментації
  - полігональні мітки
  - MIT бібліотека сегментації
  - mask mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суфікс -seg у назві файла вибирає голову масок, тому аргумент
        # task не потрібний.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), одна маска на виявлення
        print(result.boxes.xyxy.shape)   # (N, 4), ті самі N рядків
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Контури масок
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDFINEn-seg.pt")

        result = model(SAMPLE_IMAGE)


        # .xy є списком контурів (P, 2) у пікселях, .xyn містить ті самі
        нормалізовані контури.

        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'Інше сімейство, той самий виклик'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Продовжує з опублікованих ваг сегментації, включно з головою масок.

        # data має вказувати на датасет, мітки якого містять полігони.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: З ваг виявлення
      language: bash
      code: |
        # Ваги виявлення не мають голови масок, тому це явне перенесення:
        # голова починає ненавченою. Запит task=segment надає дозвіл на це.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # маски
        print(metrics["metrics/mAP50-95(M)"])    # маски, явно
        print(metrics["metrics/mAP50-95(B)"])    # рамки
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Використати експортований файл
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика маршрутизує за суфіксом файла, тому експортований артефакт
        # завантажується як контрольна точка й повертає той самий Results.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Визначення

Сегментація екземплярів поєднує виявлення з формою. Кожен екземпляр об'єкта
досі отримує рамку, клас та оцінку, а також бінарну маску пікселів, які йому
належать. Маски можуть перекриватися, а пікселі, що не належать жодному об'єкту,
залишаються непризначеними, чим ця задача відрізняється від
[семантичної сегментації](/docs/tasks/semantic-segmentation) і
[паноптичної сегментації](/docs/tasks/panoptic-segmentation).

`segment` є канонічним ключем задачі, а суфікс `-seg` у назві файла контрольної
точки вибирає її, тому під час завантаження опублікованих ваг `task=` не
потрібний.

`predict()` заповнює `result.masks` разом із `result.boxes`. `.data` є стеком
у формі `(N, H, W)` на полотні початкового зображення, вирівняним за рядками з
рамками, тому маска `i` належить рамці `i`. `.xy` перетворює кожну маску на її
найбільший зовнішній контур як піксельний масив `(P, 2)`, а `.xyn` повертає той
самий нормалізований контур.

## Моделі

Чотири сімейства підтримують і навчання, і передбачення масок:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[D-FINE](/docs/models/d-fine) і [RTMDet](/docs/models/rtmdet). RF-DETR потребує
власного набору залежностей `pip install "libreyolo[rfdetr]"`; інші три
працюють із базовим пакетом.

[Mask R-CNN](/docs/models/mask-rcnn) передбачає, валідує та експортує маски,
але його `train()` спричиняє `NotImplementedError`.

[EoMT](/docs/models/eomt) передбачає й валідує маски та також не підтримує
навчання, а його експорт ще вужчий: `export()` приймає лише семантичну задачу
й спричиняє `NotImplementedError` для `segment` і `panoptic`, оскільки потрібний
їм контракт середовища виконання для масок запитів не визначено.
Використовуйте EoMT для масок екземплярів у Python, а не через експортований
граф.

Окрема група сегментує за підказкою, а не за списком класів: клацання, рамка
або фраза вибирає об'єкт, а модель повертає його маску.
[SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2),
[SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam),
[EdgeTAM](/docs/models/edgetam) і [PicoSAM3](/docs/models/picosam3) працюють
саме так, як і [SenseNova-Vision](/docs/models/sensenova-vision), сегментація
якої є референційною й приймає фразу, що називає один об'єкт. Вони
завантажуються через власну фабрику та набори залежностей, а точний виклик
наведено на сторінці кожної моделі.

## Передбачення

Ваги завантажуються з Hugging Face під час першого використання та кешуються
локально.

<code-tabs name="predict" />

`conf` і `max_det` формують вихідні дані так само, як для виявлення, а маски
фільтруються разом із рамками, яким вони належать. Джерела, потокове оброблення
й роботу з результатами описано в розділі [передбачення](/docs/predict).

## Формат датасету

Структура збігається зі структурою виявлення: один файл міток `.txt` для
кожного зображення, знайдений шляхом заміни `images` на `labels` у шляху
зображення та зміни розширення.

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

Змінюється рядок. Сегмент є індексом класу, за яким іде плоский полігон:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

Потрібні щонайменше три точки, тому кількість координат після індексу класу
має бути парною й не меншою за шість, а полігон не повинен бути виродженим.
Координати є числами з рухомою комою в діапазоні `[0, 1]` відносно ширини й
висоти початкового зображення. Рядок виявлення з п'ятьма полями також
приймається в датасеті сегментації та читається як прямокутний сегмент, що дає
змогу завантажити датасет лише з рамками без попереднього перетворення.

YAML збігається з YAML виявлення:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

Нативний JSON COCO також працює: додайте відповідність `annotations` між назвою
розбиття та файлом JSON, а шлях розбиття задає кореневий каталог зображень.

## Навчання

<code-tabs name="train" />

Типово навчання продовжується з опублікованої контрольної точки `-seg`. Можна
почати з ваг виявлення, але це навмисне перенесення: ці ваги не містять голови
масок, тому вона починає ненавченою, а передавання `task=segment` надає дозвіл
на заміну. Датасети, аугментацію, кілька GPU й засоби журналювання описано в
розділі [навчання](/docs/train).

## Валідація

`val()` повертає звичайний словник ключів `metrics/`. Рамки й маски оцінюються
окремо, обидва за допомогою оцінювання COCO, а показники масок є основними.

<code-tabs name="val" />

Ключі без суфікса містять результати масок: `metrics/mAP50-95`,
`metrics/mAP50`, `metrics/mAP75`, потім `metrics/mAP_small`,
`metrics/mAP_medium` і `metrics/mAP_large` за площею об'єкта, а також
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium`, `metrics/AR_large` для середньої повноти.
`metrics/AR_max_det` і `metrics/max_det` записують обмеження кількості виявлень,
використане під час запуску.

Чотири показники також публікуються з явним суфіксом, `(M)` для маски й `(B)`
для рамки, щоб порівняння не залежало від того, яке число сімейство визначило
основним: `metrics/mAP50-95(M)` і `metrics/mAP50-95(B)`,
`metrics/mAP50(M)` і `metrics/mAP50(B)`, `metrics/precision(M)` і
`metrics/precision(B)`, `metrics/recall(M)` і `metrics/recall(B)`. У цій
задачі немає `metrics/precision` або `metrics/recall` без суфікса.

Уважно читайте ключі точності й повноти. Їх збережено для зворотної сумісності,
і вони є псевдонімами, а не робочою точкою: `metrics/precision(M)` містить те
саме значення, що й `metrics/mAP50-95(M)`, а `metrics/recall(M)` те саме,
що й AR масок за 100 виявлень; для `(B)` рамки поводяться так само. Графік їх
пари повідомляє одне число двічі.

## Експорт

<code-tabs name="export" />

Експортований артефакт завантажується назад через `LibreYOLO()` за суфіксом
файла, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий об'єкт `Results`. Покриття сегментації вужче за покриття виявлення
для того самого сімейства. Матриця на сторінці кожної моделі створюється з
валідованого набору й указує причину недоступності цілі. Формати, їхні додаткові
набори залежностей та обмеження описано в розділі
[експорту й розгортання](/docs/export).
