---
title: DETR
families:
  - detr
seo_title: 'DETR: передбачення та експорт під ліцензією Apache-2.0'
description: >-
  Запускайте DETR, початковий трансформер виявлення, у LibreYOLO. Установлюйте,
  виконуйте передбачення, валідацію та експорт чотирьох розмірів на основі
  ResNet, усіх під ліцензією Apache-2.0.
lead: >-
  DETR є початковим трансформером виявлення, який передбачає фіксовану множину
  об'єктів за допомогою трансформерного декодера зі зіставленням за угорським
  алгоритмом замість якорів або щільної сітки. LibreYOLO постачає чотири розміри
  для виявлення лише в режимі інференсу.
keywords:
  - DETR
  - трансформер виявлення
  - детекція об'єктів
  - угорське зіставлення
  - transformer decoder
  - Meta AI
  - Facebook AI Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() повертає звичайний словник, а не об'єкт
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## Встановлення

DETR не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. `conf` і `max_det` фільтрують вибір
запитів; `iou` приймається для узгодженості API, але не впливає на результат,
оскільки декодер передбачає множину без етапу NMS. Типи джерел, потокове
передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

DETR у LibreYOLO призначена лише для інференсу. Початковий проєкт навчає модель
протягом 500 епох зі зіставленням за угорським алгоритмом. Цей рецепт тут не
реалізовано, тому `train()` спричиняє `NotImplementedError`.

## Варіанти

Чотири контрольні точки поєднують дві глибини бекбона, ResNet-50 або
ResNet-101, із необов'язковим розширеним етапом C5. Варіанти DC5 зберігають
останній етап бекбона в повній роздільній здатності замість подальшого
зменшення, тому декодер читає детальнішу карту ознак з того самого розміру
входу. Усі чотири мають 100 навчених запитів об'єктів і шестишаровий
трансформерний енкодер-декодер та працюють з однаковою вхідною роздільною
здатністю.

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому проводилося
навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results`. У розділі
[експорту](/docs/export) наведено аргументи, які приймає кожен формат.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>
