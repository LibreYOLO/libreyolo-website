---
title: DeiT
families:
  - deit
seo_title: 'Класифікатор зображень DeiT: передбачення, валідація, експорт'
description: >-
  Запускайте класифікатори зображень DeiT у LibreYOLO: заморожене музейне
  сімейство лише для інференсу в розмірах tiny, small і base під ліцензією
  Apache-2.0.
lead: >-
  DeiT (Data-efficient image Transformer) є звичайним класифікатором Vision
  Transformer, навченим лише на ImageNet-1k без додаткових даних попереднього
  навчання. LibreYOLO зберігає розміри tiny, small і base з патчем 16 як
  заморожені експонати лише для інференсу.
keywords:
  - DeiT
  - Vision Transformer
  - ViT
  - класифікація зображень
  - ImageNet
  - ефективне навчання на даних
  - музейне сімейство моделей
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreDeiTb-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## Встановлення

Для DeiT не потрібні додаткові залежності понад базовий пакет.

```bash
pip install libreyolo
```

## Передбачення

Це сімейство призначене лише для інференсу: `train()` спричиняє
`NotImplementedError`, тому на цій сторінці немає розділу «Навчання».
Передбачення, валідація та експорт підтримуються. Під час першого використання
ваги завантажуються з Hugging Face і кешуються локально. Суфікс `-cls` у
назві файлу обов'язковий і вибирає завдання класифікації.

<code-tabs name="predict" />

Повернений об'єкт `Results` містить тензор `probs` замість `boxes`;
`top1` і `top5` індексують 1,000 класів ImageNet-1k, а `top1conf`
містить оцінку softmax для найкращого передбачення. Кожен розмір має фіксовану
вхідну роздільну здатність із власного позиційного ембедингу: попередня обробка
змінює розмір і виконує центральне кадрування до неї. Передавання іншого
`imgsz` спричиняє помилку замість непомітної зміни дискретизації. Типи джерел,
потокове передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Валідація

`val()` повертає словник із правильністю top-1 і top-5, виміряною на датасеті
зі звичайною структурою папок `train/<class>/` і `val/<class>/`.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results`. Граф також можна запускати
безпосередньо в середовищі виконання без установленої LibreYOLO, але тоді
попередню та подальшу обробку потрібно реалізувати самостійно.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
