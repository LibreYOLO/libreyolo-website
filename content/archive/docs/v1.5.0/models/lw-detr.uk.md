---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: передбачення та експорт під ліцензією Apache-2.0'
description: >-
  Запускайте LW-DETR у LibreYOLO для виявлення об'єктів у реальному часі.
  Установлюйте, виконуйте передбачення, валідацію та експорт п'яти розмірів на
  основі ViT, усіх під ліцензією Apache-2.0.
lead: >-
  Звичайний трансформер виявлення на основі ViT, який Baidu представила як
  альтернативу детекторам YOLO у реальному часі. LibreYOLO постачає п'ять
  розмірів для виявлення лише в режимі інференсу.
keywords:
  - LW-DETR
  - трансформер виявлення
  - детекція об'єктів у реальному часі
  - plain ViT
  - DETR
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() повертає звичайний словник, а не об'єкт
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreLWDETRt.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Встановлення

LW-DETR не потребує додаткових залежностей. Усе, що вона імпортує, входить до
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

LW-DETR у LibreYOLO призначена лише для інференсу. Початковий проєкт навчає з
наглядом one-to-many від Group-DETR для кількох груп запитів та функцією втрат
класифікації з урахуванням IoU. Цей рецепт тут не під'єднано, тому `train()`
спричиняє `NotImplementedError`.

## Варіанти

Доступні п'ять розмірів. Усі використовують звичайний енкодер ViT,
багатомасштабний проєктор і деформовний декодер DETR та працюють з однаковою
вхідною роздільною здатністю. Два найменші мають однакову ширину енкодера й
відрізняються глибиною блоків; наступні два мають ширший енкодер і
відрізняються кількістю рівнів проєктора, які подають дані до декодера;
найбільший використовує найширший енкодер.

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

## Цитування

<citation-block />
