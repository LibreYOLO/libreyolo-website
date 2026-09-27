---
title: DINO-DETR
families:
  - dinodetr
seo_title: 'DINO-DETR: передбачення та експорт під ліцензією Apache-2.0'
description: >-
  Запускайте DINO-DETR у LibreYOLO для виявлення об'єктів. Установлюйте,
  виконуйте передбачення, валідацію та експорт трьох розмірів із
  шумоприглушувальними якорями, усіх під ліцензією Apache-2.0.
lead: >-
  DINO-DETR, опублікована IDEA Research як DINO, поєднує контрастивне навчання з
  усуненням шуму та змішаний вибір запитів поверх розрідженої уваги Deformable
  DETR. LibreYOLO постачає три розміри для виявлення лише в режимі інференсу.
keywords:
  - DINO-DETR
  - DINO
  - трансформер виявлення
  - шумоприглушувальні якірні рамки
  - змішаний вибір запитів
  - детекція об'єктів
  - IDEA Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() повертає звичайний словник, а не об'єкт
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreDINODETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## Встановлення

DINO-DETR не потребує додаткових залежностей. Усе, що вона імпортує, входить
до базового встановлення й використовує те саме ядро багатомасштабної
деформовної уваги лише на PyTorch, що й сімейство Deformable DETR у LibreYOLO.

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

DINO-DETR у LibreYOLO призначена лише для інференсу. Початковий проєкт навчає
з контрастивним усуненням шуму та зіставленням за угорським алгоритмом. Цей
рецепт тут не реалізовано, тому `train()` спричиняє
`NotImplementedError`.

## Варіанти

Доступні три контрольні точки, усі з однаковою вхідною роздільною здатністю.
`r50` і `r50s5` мають спільний бекбон ResNet-50 і відрізняються кількістю
масштабів карт ознак, що подаються до декодера: чотири проти п'яти. `swinl`
замінює бекбон на Swin-L і також семплює п'ять масштабів.

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

<provenance-box>

Три офіційні контрольні точки походять із папки релізу авторів на Google
Drive, а не з картки моделі Hugging Face. Початковий репозиторій оголошує
Apache-2.0 на рівні репозиторію, але не додає файл ліцензії чи метадані
ліцензії до самих контрольних точок. Тому підставою для розповсюдження є ця
заява на рівні репозиторію, а не окремий дозвіл для контрольної точки. Кожне
дзеркало LibreYOLO постачає дослівний текст початкової Apache-2.0 разом із
пояснювальним повідомленням.

</provenance-box>

## Цитування

<citation-block />
