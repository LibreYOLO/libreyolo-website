---
title: Семантична сегментація
seo_title: Семантична сегментація в LibreYOLO
description: >-
  Призначайте клас кожному пікселю в LibreYOLO: сімейства для цієї задачі,
  формат щільної маски й виклики передбачення, навчання, валідації та експорту.
lead: >-
  Семантична сегментація призначає клас кожному пікселю зображення й не
  розрізняє екземпляри одного класу. Ключ задачі: semantic.
keywords:
  - семантична сегментація python
  - класифікація пікселів
  - щільне передбачення
  - навчити модель сегментації
  - mIoU
  - MIT бібліотека сегментації
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Суфікс -sem у назві файлу вибирає задачу, тому аргумент

        # task не потрібний.

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # ідентифікатори класів (H, W) на початковому
        полотні

        print(mask.classes)      # відсортовані наявні ідентифікатори класів без
        255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: По одному класу
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # булевий масив (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Інше сімейство, той самий виклик'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: На ADE20K
      language: bash
      code: >
        # ade20k.yaml містить вбудований скрипт завантаження архіву обсягом
        близько 1 GB,

        # тому потрібен явний дозвіл, якщо даних немає локально.

        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() повертає звичайний словник, а не об'єкт.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика визначає маршрут за суфіксом файлу, тому експортований
        артефакт

        # завантажується як контрольна точка й повертає той самий об'єкт
        Results.

        model = LibreYOLO("LibreSegformerb0-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Визначення

Семантична сегментація позначає пікселі, а не об'єкти. Кожен піксель отримує один
ідентифікатор класу, а два автомобілі, що торкаються на зображенні, утворюють одну
ділянку класу автомобілів без межі між ними. Підраховувати окремі екземпляри дає змогу
[сегментація екземплярів](/docs/tasks/instance-segmentation), а одночасно позначати
кожен піксель і розділяти екземпляри дає змогу
[паноптична сегментація](/docs/tasks/panoptic-segmentation).

`semantic` є канонічним ключем задачі, а суфікс `-sem` у назві файлу контрольної
точки вибирає її, тому під час завантаження опублікованих ваг параметр `task=` не потрібний.

Метод `predict()` заповнює `result.semantic_mask`. Поле `.data` є цілочисловою
картою класів `(H, W)` на полотні початкового зображення, `.classes` містить наявні
ідентифікатори у відсортованому порядку, а `.class_mask(id)` повертає булеву вибірку
`(H, W)` для одного класу. Значення `255` є міткою ігнорування: воно ніколи не є
класом, не враховується у функції втрат і метриках та не входить до `.classes`.

## Моделі

Три сімейства підтримують і навчання, і передбачення:
[SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) і
[DINOv2](/docs/models/dinov2). SegFormer і LingBot-Vision працюють із базовим
пакетом та постачаються з опублікованими вагами. Для DINOv2 потрібна команда
`pip install "libreyolo[rfdetr]"`, і це сімейство не має контрольної точки,
розміщеної LibreYOLO: воно завантажує початковий бекбон, а його щільна голова
починає з випадкової ініціалізації, тому це відправна точка для навчання,
а не готовий засіб передбачення.

Ще чотири сімейства виконують передбачення, валідацію та експорт, але їхній метод
`train()` породжує `NotImplementedError`: [FCN](/docs/models/fcn),
[DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet) і
[EoMT](/docs/models/eomt).

Набори класів залежать від контрольної точки, а не від сімейства. Опубліковані ваги
походять із датасетів, простори міток яких мають мало спільного, зокрема 150 класів
ADE20K проти 19 класів Cityscapes. Отже, поле `names` контрольної точки вказує,
що саме вона може позначати, а дві контрольні точки можна порівнювати лише тоді,
коли їх навчено на одному датасеті.

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються локально.

<code-tabs name="predict" />

Карта утворюється операцією argmax для кожного пікселя, тому кроку NMS немає,
а `iou` не впливає на результат. Параметри `conf` і `max_det` приймаються для
сумісності API та нічого не роблять у SegFormer, PIDNet та інших щільних засобах
передбачення. Винятком є EoMT, де `conf` фільтрує вибір запитів. Докладніше про
джерела, потокову обробку та роботу з результатами див. у розділі
[передбачення](/docs/predict).

## Формат датасету

Кожному зображенню відповідає щільна одноканальна маска замість файлу міток `.txt`.
Її знаходять заміною `images` на каталог масок у шляху до зображення.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Маски є одноканальними зображеннями без втрат, зазвичай PNG, а PNG у режимі палітри
зчитуються як індекси палітри. Значення кожного пікселя є ідентифікатором класу
в діапазоні `0..nc-1`, значення `255` означає ігнорування, а роздільна здатність
маски має дорівнювати роздільній здатності відповідного зображення.

YAML доповнює спільний контракт двома ключами:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

Ключ `masks_dir` задає назву каталогу, якою замінюється `images`; його типове значення
дорівнює `masks`. Необов'язкове зіставлення `label_mapping` у формі
`{source_id: train_id}` застосовується до значень пікселів маски під час завантаження.
Так датасет із нумерацією від 1 до 150 перетворюється на нумерацію від 0 до 149;
кожне початкове значення без зіставлення стає ігнорованим, а кожен навчальний
ідентифікатор має бути в діапазоні `0..nc-1`.

Якщо пропустити `masks_dir`, завантажувач переходить до резервного способу: маски
растеризуються під час завантаження з полігональних міток, знайдених за звичайним
правилом заміни `images` на `labels`, а після класів об'єктів додається клас
`background`, тому `nc` збільшується на один.

Канонічний завантажувач: `libreyolo.data.SemanticDataset`.

## Навчання

<code-tabs name="train" />

Для цієї задачі `imgsz` має обмеження, якого немає в детекторі. Кожне сімейство
задає дільник, кратним якому має бути розмір вхідних даних. Його визначає сітка
патчів або крок виходу, а навчання та валідація породжують `ValueError` до початку
запуску, якщо `imgsz` не ділиться без остачі. Дільник дорівнює 32 для SegFormer,
16 для LingBot-Vision і EoMT, 14 для DINOv2 та 8 для FCN і PIDNet. Відомості про
датасети, аугментацію, роботу з кількома GPU й логери див. у розділі
[навчання](/docs/train).

## Валідація

Метод `val()` повертає звичайний словник із ключами `metrics/`, обчисленими для
частини датасету, заданої ключем `val` у YAML.

<code-tabs name="val" />

`metrics/mIoU` є середнім перетином над об'єднанням: для кожного класу перекриття
передбачених і правильних пікселів ділиться на їхнє об'єднання, а потім усереднюється
за класами. Це основний показник, який також використовується для вибору найкращої
епохи під час навчання. `metrics/pixel_accuracy` є часткою пікселів із правильним
класом, яку може завищити великий клас фону, тому для порівняння слід використовувати
mIoU. Пікселі з міткою `255` не враховуються в жодній метриці. Словник також містить
`fitness`, копію значення mIoU.

## Експорт

<code-tabs name="export" />

Експортований артефакт знову завантажується через `LibreYOLO()` за суфіксом файлу,
тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає той самий
об'єкт `Results`. Підтримка форматів відрізняється між сімействами; матриця на кожній
сторінці моделі генерується з перевіреного набору, а не вводиться вручну. Формати,
їхні доповнення й обмеження описано в розділі
[експорту й розгортання](/docs/export).
