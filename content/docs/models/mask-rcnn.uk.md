---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: 'Mask R-CNN у LibreYOLO: передбачення, валідація та експорт'
description: >-
  Запускайте Mask R-CNN у LibreYOLO для виявлення об'єктів і сегментації
  екземплярів. Установлюйте, виконуйте передбачення, валідацію та експорт порту
  torchvision під ліцензією BSD-3-Clause.
lead: >-
  Mask R-CNN додає до Faster R-CNN гілку масок для кожної області та передбачає
  маску сегментації разом із кожною виявленою рамкою. LibreYOLO переносить
  реалізацію torchvision для виявлення та сегментації екземплярів.
keywords:
  - Mask R-CNN
  - сегментація екземплярів
  - детекція об'єктів
  - Faster R-CNN
  - torchvision
  - двостадійний детектор
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Лише рамки
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" пропускає голову масок і повертає рамки з тієї самої
        # контрольної точки без масок у результаті.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # Маски
        print(metrics["metrics/mAP50-95(B)"])   # Рамки
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreMaskRCNNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## Встановлення

Mask R-CNN не потребує додаткових залежностей. Усе, що вона імпортує, входить
до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Повернений об'єкт `Results` є однаковим для всіх сімейств, тому заміна
детектора потребує зміни одного рядка. Завантаження контрольної точки без
аргументу `task` повертає маски екземплярів, оскільки сегментація є типовим
завданням цього сімейства; `result.masks` містить їх разом із рамками.
Передавання `task="detect"` завантажує ті самі ваги без голови масок і
повертає лише рамки. `conf` і `iou` задають пороги впевненості та NMS.
На відміну від детектора на основі запитів, Mask R-CNN зберігає початковий
етап NMS. Типи джерел, потокове передбачення та обробку результатів описано в
розділі [передбачення](/docs/predict).

## Варіанти

Доступний один бекбон: ResNet-50 із пірамідою ознак, побудований за допомогою
конструктора Mask R-CNN v2 від torchvision. Опублікована контрольна точка має
ліцензію BSD-3-Clause і обслуговує обидва завдання цього сімейства, тому
вибору розміру немає.

## Валідація

`val()` повертає словник ключів `metrics/`. Для типового завдання
сегментації цієї контрольної точки звичайний ключ `metrics/mAP50-95` містить
оцінку масок, а той самий запуск повідомляє оцінку рамок із суфіксом `(B)`,
тому обидві доступні за один прохід.

<code-tabs name="val" />

## Експорт

<export-matrix />

Mask R-CNN експортується лише до ONNX із розміром батча 1. Експортований граф
зберігає всередині початкові етапи зміни розміру та вставлення масок, тому
LibreYOLO примусово вмикає `dynamic=True` незалежно від переданого значення,
щоб граф залишався коректним для неквадратних джерел. Експортований файл
`.onnx` знову завантажується через `LibreYOLO()` за суфіксом і повертає
той самий об'єкт `Results`.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства. Єдина контрольна точка нижче
наведена для detect, але той самий файл також завантажується для сегментації:
не передавайте аргумент `task`, і типово він поверне маски.

<checkpoint-table />

## Ліцензування

<provenance-box>

Mask R-CNN побудовано як підклас обгортки Faster R-CNN у LibreYOLO: вона
використовує те саме джерело torchvision і ліцензію BSD-3-Clause та додає
предиктор масок і голову RoI масок із того самого перенесеного коміту.

</provenance-box>
