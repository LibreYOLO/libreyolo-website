---
title: MobileNetV4
families:
  - mobilenetv4
seo_title: 'MobileNetV4: навчання, валідація та експорт під ліцензією Apache-2.0'
description: >-
  Використовуйте MobileNetV4 у LibreYOLO для класифікації зображень.
  Установлюйте, виконуйте передбачення, донавчання, валідацію та експорт
  LibreMobileNetV4 small/medium/large.
lead: >-
  MobileNetV4 є класифікатором зображень для мобільного й edge-обладнання, який
  використовує блок Universal Inverted Bottleneck, щоб об'єднати кілька
  попередніх конструкцій мобільних блоків в одну структуру для пошуку. LibreYOLO
  підтримує її для одного завдання: класифікації.
keywords:
  - MobileNetV4
  - MobileNetV4 conv
  - класифікація зображень
  - мобільний інференс
  - edge-класифікатор
  - класифікатор ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMobileNetV4s-cls.pt source=cat.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160
        epochs=5
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMobileNetV4s-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMobileNetV4s-cls.pt format=onnx

        libreyolo export model=LibreMobileNetV4s-cls.pt format=tensorrt
        half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreMobileNetV4s-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 4a9a1b392ffb136d
---

## Встановлення

MobileNetV4 не потребує додаткових залежностей. Усе, що вона імпортує, входить
до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна моделі
потребує зміни одного рядка. Класифікатор не містить рамок або масок:
`result.probs` містить передбачення для всього зображення з `top1`,
`top5`, `top1conf` і `top5conf`. `conf`, `iou` і `max_det`
приймаються для узгодженості API, але не впливають на результат, оскільки в
одному векторі ймовірностей немає чого порогувати чи придушувати. Типи джерел,
потокове передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступні три розміри small/medium/large, усі лише зі згортками: це сімейство
не охоплює гібридні варіанти з увагою Mobile MQA. Вибір розміру прямо
зіставляє кількість параметрів із правильністю. Завдання фіксоване: кожен
розмір охоплює лише класифікацію. Назва файлу ваг кожного розміру завершується
на `-cls.pt`, і за цим суфіксом фабрика маршрутизує до сімейства; аргумент
`task=` не потрібен.

## Навчання

Донавчання починається з опублікованого бекбона ImageNet і автоматично
перебудовує завершальний шар класифікатора під кількість класів цільового
датасету.

<code-tabs name="train" />

Без додаткових налаштувань тренер виконує 100 епох із `lr0=1e-3`,
оптимізатором AdamW, батчем 64 і ранньою зупинкою після 50 епох без покращення.
`data` приймає кореневий каталог датасету (`train/` і `val/`, по одній
папці на клас), відому коротку назву на кшталт `imagenette160` або URL
`.zip`. `lora=True` тут не підтримується; його передавання спричиняє
помилку, оскільки LoRA в LibreYOLO націлена на трансформерні компоненти із
шарами `nn.Linear`, яких немає в блоках UIB цього сімейства.

Датасети, аугментацію, кілька GPU та логери описано в розділі
[навчання](/docs/train).

## Валідація

`val()` повертає словник ключів `metrics/`. Для класифікації це правильність
top-1 і top-5 на валідаційному поділі.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results`. У розділі
[експорту](/docs/export) наведено аргументи, які приймає кожен формат, і
додаткові параметри деяких форматів.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>
