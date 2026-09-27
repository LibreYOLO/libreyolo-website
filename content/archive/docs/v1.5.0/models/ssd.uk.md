---
title: SSD
families:
  - ssd
seo_title: 'SSD (SSD300): виявлення об''єктів у LibreYOLO'
description: >-
  Запускайте SSD300 у LibreYOLO: однопрохідний детектор VGG16 для передбачення,
  валідації та експорту до ONNX під ліцензією BSD-3-Clause. Навчання не
  підтримується.
lead: >-
  SSD (Single Shot MultiBox Detector) передбачає всі рамки й оцінки класів зі
  щільної сітки типових рамок за один прямий прохід без окремого етапу
  пропозицій областей. LibreYOLO постачає контрольну точку SSD300 із VGG16 як
  детектор лише для інференсу.
keywords:
  - SSD
  - SSD300
  - Single Shot MultiBox Detector
  - детекція об'єктів
  - VGG16
  - детектор на основі якорів
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSSD300.pt")


        # imgsz тут пропущено навмисно: SSD300 трасується на початковому полотні

        # контрольної точки, а будь-яке інше значення спричиняє помилку до
        початку експорту.

        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreSSD300.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## Встановлення

SSD не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. SSD декодує сітку типових рамок з
оцінками для кожного класу, а потім виконує немаксимальне придушення, тому
`conf`, `iou` і `max_det` тут дійсно впливають на результат, на відміну
від детекторів на основі запитів у цій бібліотеці. Типи джерел, потокове
передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

SSD постачає одну контрольну точку: мережу SSD300 із VGG16 на її фіксованому
початковому полотні. Це сімейство не має вибору розміру чи масштабу;
передбачення, валідація та експорт використовують один граф.

Файл ваг має назву `LibreSSD300.pt`, тобто префікс сімейства разом із його
єдиним ключем розміру `"300"`. Йому відповідає клас `LibreSSD`, тому
безпосереднє створення має форму `LibreSSD(size="300")`, а не клас із назвою
файлу.

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому проводилося
навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

SSD експортується лише до ONNX; усі інші формати наразі заблоковано для цього
сімейства. Експорт завжди використовує початкове полотно контрольної точки, а
граф відкриває необроблену запаковану голову SSD замість об'єднаного виходу
немаксимального придушення, тому `nms=True` під час експорту не приймається.
Власні бекенди LibreYOLO виконують декодування й придушення після повторного
завантаження графа.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

Код SSD300 у LibreYOLO не перенесено з власного релізу авторів статті для
Caffe; він походить від реалізації SSD300 у torchvision під ліцензією
BSD-3-Clause, і саме на цей репозиторій веде посилання початкового джерела
вище. Ваги бекбона VGG16 походять також від повністю згорткової скороченої
VGGNet Oxford, випущеної Karen Simonyan та Andrew Zisserman під ліцензією
CC BY 4.0.

</provenance-box>

## Цитування

<citation-block />
