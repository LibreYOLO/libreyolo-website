---
title: Paddle
seo_title: Експорт у PaddlePaddle з LibreYOLO
description: >-
  Перетворення детектора LibreYOLO на модель інференсу PaddlePaddle через
  X2Paddle: закріплений набір інструментів, статичні графи FP32 із батчем 1 та
  інференс на CPU.
lead: >-
  Моделі інференсу PaddlePaddle складаються з графа model.pdmodel і
  розташованого поряд файлу ваг model.pdiparams. LibreYOLO експортує статичний
  граф ONNX з opset 15, перетворює його за допомогою X2Paddle та пакує результат
  із metadata.yaml, щоб він завантажувався через ту саму фабрику, що й усі інші
  середовища виконання.
keywords:
  - експорт yolo paddle
  - paddlepaddle інференс
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Параметр
    value: export(format="paddle")
    mono: true
  - label: Результат
    value: 'Каталог із model.pdmodel, model.pdiparams та metadata.yaml'
  - label: Додатково
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Повторне завантаження
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Бекенд
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Форми
    value: 'Статичні, батч 1, opset 15. Усі три умови обов''язкові.'
  - label: Точність
    value: Лише FP32 і лише CPU.
  - label: Набір інструментів
    value: >-
      PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 або старіша версія;
      перевіряються точно
verification: >-
  Перевірено за файлами libreyolo/export/paddle.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py,
  libreyolo/backends/paddle.py, docs/paddle.md та pyproject.toml у гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: >
        # Python 3.10-3.12. Для Windows валідовано шлях через WSL2 з Ubuntu
        22.04.

        pip install "libreyolo[paddle]"
    - label: Перевірити закріплені версії
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записує каталог weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # ціле число; квадратне полотно цього сімейства
            batch=1,          # будь-яке інше значення спричиняє ValueError
            dynamic=False,    # True спричиняє ValueError
            simplify=True,    # False спричиняє ValueError
            opset=15,         # будь-яке інше значення спричиняє ValueError
            output_path=None, # None записує weights/<stem>_paddle
        )
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: Безпосередньо бекенд
      language: python
      code: >
        from libreyolo.backends.paddle import PaddleBackend


        # Те, що LibreYOLO() створює для каталогу Paddle. Той самий об'єкт
        Results,

        # без проміжної маршрутизації фабрики.

        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")

        result = backend.predict("parkour.jpg")

        print(result.boxes.xyxy[:3])
    - label: Безпосередньо Paddle
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # На цьому шляху попередня та подальша обробка покладаються на вас.
  support:
    - label: Перевірити сімейство й завдання перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Встановлення

<code-tabs name="install" />

Додатковий пакет закріплює точний стек, для якого виміряно еквівалентність:
PaddlePaddle 2.6.2, X2Paddle 1.6.0 та ONNX 1.17 або старішу версію. Ці версії
перевіряються під час експорту, а не лише встановлення, і відмінна версія
спричиняє `ImportError` із назвою очікуваної. Новіші випуски Paddle відхиляють
частини статичного коду, який генерує X2Paddle 1.6.0, тому краще завершити процес
заздалегідь, ніж створити артефакт, який ніхто не валідував.

## Експорт

<code-tabs name="export" />

Чотири аргументи зафіксовано, а не лише задано типово. Значення `dynamic` має бути
`False`, `batch` має дорівнювати 1, `simplify` має бути `True` для повністю
статичного графа перетворення, а `opset` має дорівнювати 15, тобто максимальному
значенню, яке приймає X2Paddle 1.6.0. Усі інші значення відхиляються до трасування.

Для проміжного графа виконується одна нормалізація. За визначенням ONNX пропущена
дилатація MaxPool дорівнює одиниці, PyTorch записує явний атрибут з усіх одиниць,
а X2Paddle 1.6.0 відхиляє його. Тому експортер вилучає це надлишкове типове
значення, не змінюючи заданої операції.

Артефакт є каталогом із `model.pdmodel`, `model.pdiparams` та `metadata.yaml`.
Код Python, який X2Paddle генерує під час перетворення, до нього не входить.

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` розпізнає будь-який каталог, що містить і `model.pdmodel`, і
`model.pdiparams`, читає `metadata.yaml` та повертає той самий об'єкт `Results`,
що й контрольна точка. Пристрій, відмінний від `auto` або `cpu`, спричиняє
помилку: цей бекенд працює лише на CPU.

Фабрика створює `PaddleBackend`, який експортується з `libreyolo` та доступний
для імпорту як `libreyolo.backends.paddle.PaddleBackend`. Створюйте його
самостійно, коли потрібен бекенд без маршрутизації фабрики за суфіксом, наприклад
щоб явно передати `task=` для каталогу, файл `metadata.yaml` якого створено не
вами. Його метод `predict()` приймає ті самі джерела й повертає ті самі результати.

Фрагмент із безпосереднім середовищем виконання відтворює конфігурацію бекенду,
а три вимкнені параметри вибрано свідомо. Пайплайн злиття операцій CPU у Paddle
2.6 може аварійно завершитися під час оптимізації великих графів gather і scatter,
створених для деформованої уваги, тому еквівалентність виміряно для переносного
незлитого статичного графа. На цьому шляху попередня обробка, декодування, NMS та
повторне масштабування координат покладаються на вас.

## Обмеження

Немає динамічних форм, FP16, INT8, вбудованого NMS і середовища виконання GPU.

Валідовані поєднання: виявлення YOLO9, YOLO9-E2E та YOLO9-P2; виявлення,
оцінювання пози й сегментація EC; виявлення RT-DETRv4, D-FINE, DEIM і DEIMv2;
а також виявлення й оцінювання пози YOLO-NAS. Для кожного перевірено перетворення,
повторне завантаження у середовищі CPU, еквівалентність необробленого виходу та
збіг загальнодоступних результатів.

Заблоковані поєднання із зафіксованою причиною:

| Поєднання | Причина |
|---|---|
| RF-DETR, усі завдання | Потрібні ONNX opset 17 і GridSample; X2Paddle 1.6.0 приймає opset 15 або нижчий і не має перетворювача GridSample |
| Виявлення RT-DETR та RT-DETRv2 | Навченим графам потрібен GridSample з opset 16 або новішим |
| Сегментація D-FINE | Перетворюється й повторно завантажується, але відносна середньоквадратична помилка логітів маски становить 3.52%, а мінімальний IoU зіставленої маски становить 0.582 |
| Сегментація YOLO9 | У LibreYOLO модель YOLO9 підтримує лише виявлення |
| Сегментація RTMDet-Ins | Декодування масок із динамічним ядром не має контракту експортованого середовища виконання |

Усі поєднання, не позначені як валідовані чи заблоковані, відхиляються з
повідомленням, що їх не валідовано на шляху перетворення ONNX у Paddle.

Повну сітку сімейств і завдань наведено в
[матриці експорту](/docs/reference/export-matrix). Для окремого поєднання:

<code-tabs name="support" />

