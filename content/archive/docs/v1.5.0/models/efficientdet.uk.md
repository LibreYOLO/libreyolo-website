---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: виявлення об''єктів у LibreYOLO'
description: >-
  Запускайте EfficientDet D0-D4 у LibreYOLO: детектори BiFPN для передбачення,
  валідації та експорту до ONNX, TensorRT і OpenVINO під ліцензією Apache-2.0.
lead: >-
  EfficientDet поєднує бекбон EfficientNet із повторюваною двонапрямною
  пірамідальною мережею ознак (BiFPN) і одночасно масштабує глибину, ширину та
  роздільну здатність у п'яти розмірах. LibreYOLO постачає її як детектор лише
  для інференсу.
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - детекція об'єктів
  - складене масштабування
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreEfficientDetd0.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Встановлення

EfficientDet не потребує додаткових залежностей. Усе, що вона імпортує, входить
до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. EfficientDet декодує кандидатів на
основі якорів, а потім виконує немаксимальне придушення окремо для кожного
класу, тому `conf`, `iou` і `max_det` тут дійсно впливають на результат.
Типи джерел, потокове передбачення та обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступні п'ять розмірів від D0 до D4. Кожен наступний крок поєднує більший
бекбон EfficientNet із глибшою та ширшою BiFPN і глибшою головою передбачення,
тому кількість параметрів та обчислення зростають разом за правилом складеного
масштабування зі статті.

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому проводилося
навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results`.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

Контрольні точки D0-D4 бібліотеки LibreYOLO перетворено за допомогою проєкту
rwightman/efficientdet-pytorch під ліцензією Apache-2.0, який сам відтворює
офіційні ваги google/automl, навчені в TensorFlow, не змінюючи навчених
тензорів. Вихідний код проєкту zylo117/Yet-Another-EfficientDet-Pytorch під
ліцензією LGPL не переглядався й не використовувався.

</provenance-box>
