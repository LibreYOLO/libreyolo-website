---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: виявлення об''єктів у LibreYOLO'
description: >-
  Запускайте CenterNet (Objects as Points) у LibreYOLO з бекбонами ResDCN-18 і
  DLA-34. Виконуйте передбачення, валідацію та експорт до ONNX під ліцензією
  MIT. Навчання не підтримується.
lead: >-
  CenterNet моделює об'єкт як центральну точку його обмежувальної рамки й
  регресує всі інші властивості з піка теплової карти, тому їй не потрібні якорі
  та етап немаксимального придушення. LibreYOLO постачає її як детектор лише для
  інференсу.
keywords:
  - CenterNet
  - Objects as Points
  - детекція ключових точок
  - детектор без якорів
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCenterNetresdcn18.pt")


        # Для експорту ONNX потрібен opset 16 або новіший: етап підвищення
        роздільної

        # здатності з деформовною згорткою перетворюється на GridSample, доданий
        в opset 16.

        model.export(format="onnx", opset=18)

        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreCenterNetresdcn18.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Встановлення

CenterNet не потребує додаткових залежностей. Усе, що вона імпортує, входить
до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. `conf` і `max_det` фільтрують
ранжовані піки теплової карти; `iou` приймається для узгодженості API, але не
впливає на результат, оскільки декодування top-k піків CenterNet не потребує
етапу придушення рамок за IoU. Типи джерел, потокове передбачення та обробку
результатів описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступні два бекбони. `resdcn18` поєднує основу ResNet-18 із підвищенням
роздільної здатності деформовною згорткою; `dla34` поєднує основу DLA-34 з
ітеративним підвищенням роздільної здатності через глибоку агрегацію. Обидва
подають дані до тих самих трьох щільних голів (теплова карта, ширина/висота,
зсув) і використовують однакове вхідне полотно.

## Валідація

`val()` повертає словник ключів `metrics/` із точністю, повнотою, mAP 50 та
mAP 50-95, виміряними на будь-якому датасеті у форматі, на якому проводилося
навчання.

<code-tabs name="val" />

## Експорт

<export-matrix />

Для експорту ONNX потрібен opset 16 або новіший: етап підвищення роздільної
здатності з деформовною згорткою в обох бекбонах перетворюється на оператор
ONNX `GridSample`, доданий в opset 16. Запит старішого opset спричиняє
помилку до початку трасування.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box>

Граф ResDCN-18 також зазначає авторство
human-pose-estimation.pytorch від Microsoft під ліцензією MIT, а граф DLA-34
зазначає реалізацію DLA від Fisher Yu під ліцензією BSD-3-Clause. LibreYOLO
не включає початкове розширення DCNv2, використане початковим проєктом.
Нативне виконання натомість використовує `deform_conv2d` від torchvision
під ліцензією BSD-3-Clause, а переносну реалізацію лише для експорту окремо
створено для LibreYOLO.

</provenance-box>

## Цитування

<citation-block />
