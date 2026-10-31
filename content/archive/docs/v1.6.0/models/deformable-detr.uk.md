---
title: Deformable DETR
families:
  - deformable_detr
seo_title: 'Deformable DETR: передбачення та експорт під ліцензією Apache-2.0'
description: >-
  Запускайте Deformable DETR у LibreYOLO для виявлення об'єктів. Установлюйте,
  виконуйте передбачення, валідацію та експорт п'яти розмірів із розрідженою
  увагою, усіх під ліцензією Apache-2.0.
lead: >-
  Deformable DETR замінює щільну перехресну увагу DETR розрідженим
  багатомасштабним семплюванням навколо кожної опорної точки, що зробило
  навчання трансформерних детекторів практичним. LibreYOLO постачає п'ять
  розмірів для виявлення лише в режимі інференсу.
keywords:
  - Deformable DETR
  - трансформер виявлення
  - розріджена увага
  - багатомасштабна увага
  - детекція об'єктів
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() повертає звичайний словник, а не об'єкт
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreDeformableDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## Встановлення

Deformable DETR не потребує додаткових залежностей. Усе, що вона імпортує,
входить до базового встановлення, а ядро багатомасштабної деформовної уваги
реалізовано лише в PyTorch.

```bash
pip install libreyolo
```

Установлення `libreyolo[hub-kernels]` необов'язкове. Коли пакет `kernels`
доступний, LibreYOLO під час виконання отримує скомпільоване ядро
багатомасштабної деформовної уваги з Hugging Face Hub і використовує його
замість ядра лише на PyTorch; `LIBREYOLO_HUB_KERNELS=0` вимикає його.

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

Deformable DETR у LibreYOLO призначена лише для інференсу. Початковий проєкт
навчає зі зіставленням за угорським алгоритмом і фокальною функцією втрат
класифікації. Цей рецепт тут не реалізовано, тому `train()` спричиняє
`NotImplementedError`.

## Варіанти

П'ять контрольних точок охоплюють випущені конфігурації, усі з однаковою
вхідною роздільною здатністю. `r50ss` обмежує увагу одним масштабом ознак;
`r50ssdc5` додає до нього розширений етап C5 бекбона. `r50` є типовою
багатомасштабною конфігурацією, що семплює чотири рівні карт ознак.
`r50refine` додає ітеративне уточнення обмежувальних рамок у шарах декодера,
а `r50twostage` створює початкові пропозиції областей із виходу енкодера
замість навчених запитів.

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
