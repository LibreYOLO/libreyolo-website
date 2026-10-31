---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: семантична сегментація в LibreYOLO'
description: >-
  Використовуйте SegFormer у LibreYOLO для семантичної сегментації ADE20K у
  розмірах b0-b5. Установлення, передбачення, навчання та експорт; попередньо
  навчені ваги призначені лише для некомерційного використання.
lead: >-
  SegFormer є трансформером семантичної сегментації, що поєднує ієрархічний
  кодувальник Mix Transformer (MiT) з легкою all-MLP головою декодування й не
  потребує важких декодерів і фіксованих позиційних кодувань, потрібних
  попереднім трансформерам сегментації. LibreYOLO підтримує його для одного
  завдання, семантичної сегментації, у шести розмірах.
keywords:
  - SegFormer
  - семантична сегментація python
  - сегментація зображень SegFormer
  - Mix Transformer MiT
  - трансформер для сегментації
  - ADE20K
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (донавчання)
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
    - label: З нуля
      language: python
      code: >
        from libreyolo.models.segformer.model import LibreSegformer


        # Без model_path: випадкова ініціалізація, нічого не завантажується. Це
        єдиний шлях

        # до ваг без обмеження некомерційного використання попередньо навчених
        контрольних точок.

        model = LibreSegformer(size="b0", nb_classes=150)

        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
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
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Використання експортованого файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика визначає маршрут за суфіксом файла, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreSegformerb0-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Установлення

SegFormer не потребує додаткових компонентів. Усе, що він імпортує, входить до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються локально.

<code-tabs name="predict" />

`result.semantic_mask` містить щільну карту класів: `.data` є тензором `(H, W)`
з ідентифікаторами класів у початковому розмірі зображення, а `.classes` перелічує
ідентифікатори класів, які фактично присутні. Значення `result.boxes` дорівнює
`None`, оскільки виявлень для окремих екземплярів немає. Аргументи `conf` та `iou`
приймаються задля узгодженості API, але не змінюють вивід: модель повертає один
клас на піксель, а не виявлення окремих екземплярів, які потрібно фільтрувати чи
дедуплікувати. Докладніше про джерела, потокове оброблення та роботу з результатами
дивіться в розділі [передбачення](/docs/predict).

## Варіанти

Шість розмірів, від b0 до b5, на кожному кроці розширюють і поглиблюють
кодувальник Mix Transformer, зберігаючи однакову конструкцію all-MLP голови декодування.

<checkpoint-table />

## Навчання

Типово `train()` донавчає опубліковану контрольну точку. Натомість не передавайте
`model_path` до `LibreSegformer(...)`, щоб створити модель із випадково
ініціалізованими кодувальником і головою та навчати її з нуля. Це єдиний спосіб
отримати ваги без обмеження некомерційного використання попередньо навчених
контрольних точок (дивіться [Ліцензування](#licensing)).

<code-tabs name="train" />

Якщо параметри не змінювати, засіб навчання дотримується рецепта ADE20K зі статті
про SegFormer: AdamW із базовою швидкістю навчання для бекбона, голова декодування
навчається зі швидкістю, більшою в 10 разів, спад ваг застосовується всюди, крім
LayerNorm і позиційної згортки Mix-FFN, а також використовується лінійний розклад
спаду з прогріванням. Збіжність для більших розмірів, від b3 до b5, не
валідовано від початку до кінця.

Докладніше про датасети, аугментацію, кілька GPU та засоби журналювання дивіться
в розділі [навчання](/docs/train).

## Валідація

`val()` повертає словник ключів `metrics/`: mIoU та правильність за пікселями,
виміряні на будь-якому датасеті у форматі, на якому проводилося навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт завантажується назад через `LibreYOLO()` за суфіксом файла,
тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає той
самий об'єкт `Results`. У розділі [Експорт](/docs/export) наведено аргументи,
які приймає кожен формат.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

Кодувальник і голова декодування LibreSegformer перенесені до PyTorch із
реалізації SegFormer бібліотеки Hugging Face Transformers під ліцензією
Apache-2.0, а не з NVlabs/SegFormer. Початковий репозиторій NVIDIA ніколи не
читали й не копіювали, його зазначено тут лише для атрибуції авторам статті.
Обмеження NVIDIA щодо некомерційного використання стосується лише наведених
вище попередньо навчених контрольних точок; архітектура та власний код LibreYOLO
завжди залишаються під ліцензією MIT.

</provenance-box>

## Цитування

<citation-block />
