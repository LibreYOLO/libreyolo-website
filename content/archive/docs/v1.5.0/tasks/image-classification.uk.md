---
title: Класифікація зображень
seo_title: Класифікація зображень у LibreYOLO
description: >-
  Призначте мітку всьому зображенню в LibreYOLO: сімейства для цієї задачі,
  структура датасету ImageFolder і виклики передбачення, навчання, валідації та
  експорту.
lead: >-
  Класифікація зображень призначає один розподіл міток усьому зображенню й
  нічого в ньому не локалізує. Ключ задачі має назву classify.
keywords:
  - класифікація зображень python
  - навчити класифікатор зображень
  - датасет ImageFolder
  - top-1 accuracy
  - класифікація zero-shot
  - MIT бібліотека класифікації
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суфікс -cls у назві файла вибирає задачу, тому аргумент
        # task не потрібний.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Повний розподіл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)

        probs = result.probs


        # .data є повним вектором (C,); top5/top5conf є впорядкованими
        поданнями.

        print(probs.data.shape)

        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: Zero-shot без навчання
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP оцінює зображення за текстовими підказками, тому набір міток

        # задається під час виклику, а не вбудовується в контрольну точку.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # imagenette160 є відомою назвою датасету й завантажується під час
        першого використання.

        # Для власних даних передайте каталог із розбиттям train/.

        model = LibreYOLO("LibreResNet50-cls.pt")

        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() повертає звичайний словник, а не об'єкт.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Використати експортований файл
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика маршрутизує за суфіксом файла, тому експортований артефакт
        # завантажується як контрольна точка й повертає той самий Results.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Визначення

Класифікація зображень створює одну оцінку для кожного класу всього зображення
й не повертає жодних координат. Вона відповідає на запитання, що зображено, але
ніколи не вказує де, і цим відрізняється від
[виявлення об'єктів](/docs/tasks/object-detection).

`classify` є канонічним ключем задачі, а суфікс `-cls` у назві файла
контрольної точки вибирає її. Для сімейств класифікації цей суфікс обов'язковий,
тому `LibreResNet50.pt` не сприймається як класифікатор, ним є лише
`LibreResNet50-cls.pt`.

`predict()` заповнює `result.probs` і залишає `boxes` порожнім. `.data` є
повним вектором оцінок, `.top1` індексом найвищої оцінки, а `.top1conf` її
значенням; `.top5` містить п'ять найвищих індексів у спадному порядку, а
`.top5conf` їхні оцінки. Індекси вказують на `result.names`. Зріз об'єкта
`Results` ніколи не обрізає `probs`, оскільки вектор належить зображенню, а не
одному рядку.

## Моделі

П'ять сімейств підтримують і навчання, і передбачення:
[ResNet](/docs/models/resnet), [ConvNeXt](/docs/models/convnext),
[MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) і
[DINOv2](/docs/models/dinov2). Перші чотири працюють із базовим пакетом і
постачають опубліковані ваги. DINOv2 потребує
`pip install "libreyolo[rfdetr]"` і не має контрольної точки, розміщеної
LibreYOLO: він завантажує upstream-бекбон із випадково ініціалізованою лінійною
головою, тому є початковою точкою для донавчання, а не готовим засобом
передбачення.

Ще п'ять підтримують передбачення, валідацію та експорт, але їхній `train()`
спричиняє `NotImplementedError`: [ViT](/docs/models/vit),
[Swin](/docs/models/swin), [VGG](/docs/models/vgg),
[AlexNet](/docs/models/alexnet) і [DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) і [SigLIP2](/docs/models/siglip2) класифікують без
фіксованого набору міток. Вони оцінюють зображення за текстовими підказками,
тому `set_classes()` визначає класи під час виклику, а етапу навчання для нового
набору міток узагалі немає. Обидва також виконують задачу `embed`.

## Передбачення

Ваги завантажуються з Hugging Face під час першого використання та кешуються
локально.

<code-tabs name="predict" />

`conf`, `iou` і `max_det` тут не мають впливу: немає кандидатів для порогової
фільтрації чи пригнічення, лише один розподіл. Джерела, потокове оброблення й
роботу з результатами описано в розділі [передбачення](/docs/predict).

## Формат датасету

Класифікація використовує дерево каталогів, а не файли міток чи YAML. Параметр
`data` указує на кореневий каталог датасету.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

Для навчання потрібний `train/`, і він визначає відповідність класів індексам
за відсортованими назвами каталогів, тому перший за алфавітом каталог стає
класом 0. Для валідації потрібний `val/`. Розбиття `test/` може бути наявним,
але типові команди навчання й валідації його не використовують. Кожне розбиття,
крім `train`, має містити ті самі назви каталогів класів, що й очікуваний набір
класів, тому невідповідність спричиняє явну помилку, а не оцінюється як
неправильне передбачення. Приймаються розширення зображень `.jpg`, `.jpeg`,
`.png`, `.bmp`, `.webp`, `.tif` і `.tiff`.

`data` приймає три типи значень: шлях до каталогу з розбиттям `train/`,
URL-адресу файла `.zip` або одну з відомих назв датасетів, `imagenette160` і
`smoke10`, які завантажуються й кешуються під час першого використання.

Канонічним завантажувачем є `libreyolo.data.classify_dataset`.

## Навчання

<code-tabs name="train" />

Оголошувати `nc` не потрібно: кількість класів визначається за назвами
каталогів у `train/`, а кінцевий лінійний шар перебудовується відповідно до неї,
тоді як бекбон переноситься без змін. Датасети, аугментацію, кілька GPU й засоби
журналювання описано в розділі [навчання](/docs/train).

## Валідація

`val()` повертає звичайний словник ключів `metrics/`, обчислених для розбиття
`val/` кореневого каталогу датасету.

<code-tabs name="val" />

`metrics/accuracy_top1` є часткою зображень, для яких клас із найвищою оцінкою
є правильним, і головним показником, за яким навчання вибирає найкращу епоху.
`metrics/accuracy_top5` є часткою зображень, для яких правильний клас входить
до п'яти класів із найвищими оцінками; ця метрика менш показова для датасетів
із невеликою кількістю класів. Словник також містить `fitness`, копію значення
top-1.

## Експорт

<code-tabs name="export" />

Експортований артефакт завантажується назад через `LibreYOLO()` за суфіксом
файла, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий об'єкт `Results`. Покриття форматів залежить від сімейства; матриця
на сторінці кожної моделі створюється з валідованого набору, а не вводиться
вручну. Формати, їхні додаткові набори залежностей та обмеження описано в
розділі [експорту й розгортання](/docs/export).
