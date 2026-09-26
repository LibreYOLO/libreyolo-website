---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: навчання, донавчання та експорт за ліцензією MIT'
description: >-
  Використання RF-DETR у LibreYOLO для виявлення, сегментації екземплярів,
  оцінювання пози та орієнтованих рамок. Встановлення, передбачення, навчання,
  валідація й експорт, повністю за ліцензією MIT.
lead: >-
  Трансформер виявлення, який передбачає фіксований набір об'єктів замість
  щільної сітки, тому не потребує NMS під час інференсу. LibreYOLO підтримує
  його для чотирьох завдань.
keywords:
  - RF-DETR
  - трансформер виявлення реального часу
  - DETR
  - виявлення об'єктів
  - сегментація екземплярів
  - оцінювання пози
  - орієнтовані обмежувальні рамки
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: 'LibreRFDETRs, виявлення на відео з 512 px.'
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Відео
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # Будь-яке джерело, яке приймає бібліотека: файл, каталог, URL-адреса,
        # індекс вебкамери, потік RTSP або список .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() повертає звичайний словник, а не об'єкт
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: На COCO
      language: bash
      code: |
        # Вбудований YAML-файл COCO містить скрипт завантаження, тому потребує
        # явного дозволу, якщо датасету ще немає локально.
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)


        # Аргументи, які приймає кожен формат:

        #

        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"

        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"

        #             | "tflite" | "coreml" | "coreai".

        #             "engine" є псевдонімом tensorrt, "litert" є псевдонімом
        tflite.

        #   imgsz     ціле число або (висота, ширина). Типовим є нативний

        #             розмір контрольної точки.

        #   batch     ціле число, типово 1.

        #   half      логічне значення, експорт у FP16. Типово False.

        #   int8      логічне значення, експорт у INT8. Типово False. Потрібен
        `data`.

        #   data      шлях до YAML-файлу датасету для калібрування int8.

        #   fraction  число з рухомою комою, частка калібрувальної вибірки.
        Типово 1.0.

        #   dynamic   логічне значення, динамічні осі. Типово True.

        #   simplify  логічне значення, спрощення графа ONNX. Типово True.

        #   opset     ціле число, opset ONNX. Якщо не задано, вибирається за
        сімейством.

        #   device    рядок, пристрій трасування. Типовим є пристрій моделі.

        #   output_path  рядок, типовою є назва, утворена з контрольної точки.

        #   verbose   логічне значення, типово False.

        #   allow_download_scripts  логічне значення, типово False. Дозволяє
        вбудований

        #             Python у YAML-файлі датасету, який потрібно завантажити.

        #

        # Кілька форматів приймають власні додаткові аргументи, наприклад
        цільову

        # платформу RKNN. Їх описано на сторінці кожного формату.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreRFDETRs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
    - label: Без LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Безпосередній запуск графа означає самостійну попередню та подальшу

        # обробку. Перевірте сигнатуру до підключення компонентів.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Встановлення

RF-DETR потребує власного додаткового пакета, який встановлює `transformers`
для бекбона.

```bash
pip install "libreyolo[rfdetr]"
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face та кешуються
локально.

<code-tabs name="predict" />

Повертається той самий об'єкт `Results`, що й для кожного сімейства, тому для
заміни детектора достатньо змінити один рядок. Параметри `conf` і `max_det`
фільтрують вибір запитів; етапу NMS для налаштування немає. Джерела, потокове
оброблення та роботу з результатами описано на сторінці
[передбачення](/docs/predict).

## Варіанти

Доступні чотири розміри й чотири завдання зі спільною архітектурою: сегментація,
оцінювання пози та орієнтовані рамки повторно використовують декодер виявлення з
іншою головою, тому приймають ті самі аргументи. Розміри мають подібну кількість
параметрів і відрізняються переважно роздільною здатністю вхідних даних.

<benchmark-table task="detect" />

<va-embed />

## Навчання

Для всіх чотирьох завдань навчання починається з опублікованої контрольної точки.
RF-DETR перелічує `pretrained` серед аргументів, які ігнорує його нативний тренер,
тому передавання `pretrained=False` не створює тут випадково ініціалізованої моделі.

<code-tabs name="train" />

Два аргументи мають тут більше значення, ніж для детектора CNN. Зберігайте `lr0`
на рівні `1e-4` або нижче, оскільки трансформери виявлення розходяться за швидкостей
навчання, які витримує модель YOLO. Залишайте `imgsz` із нативною роздільною
здатністю контрольної точки, якщо немає причини її змінювати. Розмір вхідних даних
має ділитися без остачі на добуток розміру патча бекбона й кількості вікон;
LibreYOLO перевіряє це до початку запуску й називає найближчі дійсні розміри.

Датасети, аугментацію, кілька GPU та системи журналювання описано на сторінці
[навчання](/docs/train).

## Валідація

Метод `val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50
і mAP 50-95, виміряними для будь-якого датасету у форматі, на якому виконувалося
навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт повторно завантажується через `LibreYOLO()` за суфіксом
файлу, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий `Results`. Запуск графа безпосередньо в середовищі виконання без
установленої LibreYOLO також підтримується, але тоді попередню та подальшу
обробку потрібно написати самостійно.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />

