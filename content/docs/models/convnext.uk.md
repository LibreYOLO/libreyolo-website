---
title: ConvNeXt
families:
  - convnext
seo_title: 'ConvNeXt: навчання, валідація та експорт за ліцензією Apache-2.0'
description: >-
  Використання ConvNeXt у LibreYOLO для класифікації зображень. Встановлення,
  передбачення, донавчання з LoRA, валідація та експорт LibreConvNeXt
  tiny/small/base.
lead: >-
  ConvNeXt є класифікатором зображень, повністю побудованим зі стандартних
  згорток і модернізованим блок за блоком від ResNet до проєктних рішень
  візуального трансформера. LibreYOLO підтримує його для одного завдання:
  класифікації.
keywords:
  - ConvNeXt
  - ConvNeXt tiny
  - класифікація зображень
  - повністю згорткова мережа
  - класифікатор ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreConvNeXtt-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 1682cc69cf2925e6
---

## Встановлення

ConvNeXt не потребує додаткових пакетів. Усі його імпорти входять до базового
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
заміни моделі достатньо змінити один рядок. Класифікатор не має рамок чи масок:
`result.probs` містить передбачення для всього зображення з `top1`, `top5`,
`top1conf` і `top5conf`. Параметри `conf`, `iou` та `max_det` приймаються для
сумісності API, але не впливають на результат, оскільки в одному векторі
ймовірностей немає чого відсікати за порогом або пригнічувати. Джерела, потокове
оброблення та роботу з результатами описано на сторінці
[передбачення](/docs/predict).

## Варіанти

Доступні три розміри, tiny/small/base, які навчаються й оцінюються однаково, тому
вибір є прямим компромісом між кількістю параметрів і правильністю. Завдання
зафіксовано: кожен розмір підтримує лише класифікацію. Назва файлу ваг для кожного
розміру закінчується на `-cls.pt`, і саме за цим суфіксом фабрика виконує
маршрутизацію до сімейства; аргумент `task=` не потрібен.

## Навчання

Донавчання починається з опублікованого бекбона ImageNet, а останній шар
класифікатора автоматично перебудовується відповідно до кількості класів
цільового датасету.

<code-tabs name="train" />

Без додаткових параметрів тренер виконує 100 епох із `lr0=1e-3`, AdamW, батчем
64 і ранньою зупинкою після 50 епох без покращення. `data` приймає кореневий
каталог датасету (`train/` і `val/`, по одному каталогу для кожного класу), відому
коротку назву на кшталт `imagenette160` або URL-адресу `.zip`. Блоки ConvNeXt
містять MLP `nn.Linear`, потрібні LoRA, тому тут підтримується `lora=True`, який
вставляє адаптери в MLP блоків замість донавчання всього бекбона.

Датасети, аугментацію, кілька GPU та системи журналювання описано на сторінці
[навчання](/docs/train).

## Валідація

Метод `val()` повертає словник ключів `metrics/`. Для класифікації це правильність
top-1 і top-5 на валідаційній вибірці.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт повторно завантажується через `LibreYOLO()` за суфіксом
файлу, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий `Results`. На сторінці [Експорт](/docs/export) наведено аргументи, які
приймає кожен формат, і додаткові пакети, потрібні деяким із них.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

У цьому сімействі постачається лише ConvNeXt V1. Малі попередньо навчені
контрольні точки ConvNeXt-V2 мають ліцензію CC-BY-NC 4.0 і свідомо виключені,
оскільки некомерційні ваги не можна розповсюджувати в бібліотеці з ліцензією
MIT і комерційним використанням.

</provenance-box>

## Цитування

<citation-block />

