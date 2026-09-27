---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: усунення шуму, навчання та експорт за ліцензією MIT'
description: >-
  Використання NAFNet у LibreYOLO для усунення шуму та відновлення зображень.
  Встановлення, передбачення, навчання, валідація й експорт контрольної точки
  SIDD за ліцензією MIT.
lead: >-
  NAFNet є згортковою мережею для відновлення зображень, яка вилучає нелінійні
  функції активації з типового блока UNet і замінює їх поелементним множенням.
  LibreYOLO підтримує її для одного завдання, відновлення, з опублікованою
  контрольною точкою усунення шуму зі справжніх зображень, навченою на SIDD.
keywords:
  - NAFNet
  - відновлення зображень
  - усунення шуму зображення
  - усунення розмиття зображення
  - nonlinear activation free network
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Зберегти відновлене зображення
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Походження контрольної точки
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation і dataset записуються у збереженій контрольній точці;
        # вони не змінюють процес навчання.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Кілька GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() повертає звичайний словник, а не об'єкт
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")

        result = model("noisy.jpg")


        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Встановлення

NAFNet не потребує додаткових пакетів. Усі її імпорти входять до базового
встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face та кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` містить одне поле для цього сімейства, `restored`,
щільне RGB-зображення uint8 у компонуванні HWC на початковому полотні; рамок для
перебирання немає. Параметр `save=True` записує відновлене зображення
безпосередньо на диск, а не малює анотацію поверх вхідного зображення. Параметри
`conf`, `iou` та `max_det` приймаються для сумісності сигнатури з усіма іншими
сімействами, але не впливають на результат, оскільки відновлення не створює
виявлень для фільтрування. Джерела, потокове оброблення та роботу з результатами
описано на сторінці [передбачення](/docs/predict).

## Варіанти

Ця архітектура має дві ширини: `s` (ширина 32) та `l` (ширина 64), обидві
побудовано навколо навчального патча 256 px. Передбачення та валідація працюють
із нативною роздільною здатністю зображення незалежно від розміру, додаючи поля
лише до коефіцієнта зменшення дискретизації мережі. Наразі опубліковано лише
ширину `l` як контрольну точку усунення шуму зі справжніх зображень, навчену
на SIDD.

## Навчання

NAFNet донавчається на ваших парах погіршених і чистих зображень: YAML-файл
датасету вказує на каталог `inputs/<split>/` із погіршеними зображеннями та
каталог `targets/<split>/` із чистими цільовими зображеннями, зіставленими за
основою назви файлу. `degradation` і `dataset` є необов'язковими рядками, які
записуються у збереженій контрольній точці для фіксації походження; вони не
беруть участі в навчанні.

<code-tabs name="train" />

Без додаткових параметрів тренер виконує 100 епох з AdamW та `lr0=1e-3`, батчем
16, обрізанням 256 px і ранньою зупинкою після 50 епох без покращення PSNR. Для
цього сімейства немає шляху LoRA: `lora=True` спричиняє помилку замість запуску,
оскільки `NAFNetTrainer` ніколи не вмикає донавчання адаптерів.

Під час навчання мережа працює зі звичайним глобальним усереднювальним пулінгом.
Віконний локальний пулінг NAFNet лише для інференсу (Test-time Local Converter)
від'єднується перед першою епохою та приєднується знову після завершення навчання,
оскільки зворотне поширення через локальний пул фіксованого вікна не відповідало
б використанню контрольної точки під час інференсу.

Датасети, аугментацію, кілька GPU та системи журналювання описано на сторінці
[навчання](/docs/train).

## Валідація

Метод `val()` повертає словник із `metrics/PSNR` та `metrics/SSIM`, обчисленими
в RGB на всьому дійсному полотні: SSIM використовує гауссове вікно 11x11 із
sigma 1.5, а `fitness` для вибору найкращої контрольної точки дорівнює PSNR.
`data` вказує на той самий формат датасету пар зображень, який використовується
для навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт повторно завантажується через `LibreYOLO()` за суфіксом
файлу, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий `Results`, а `restored` містить вихідне зображення. NAFNet експортується
з фіксованою просторовою роздільною здатністю: `imgsz` має бути кратним
коефіцієнту зменшення дискретизації мережі (16 для обох ширин архітектури), і за
`dynamic=True` динамічним є лише вимір батча; висота й ширина фіксуються під час
експорту.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />

