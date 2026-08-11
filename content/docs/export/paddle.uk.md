---
title: Paddle
seo_title: Експорт до PaddlePaddle із LibreYOLO
description: >-
  Перетворіть детектор LibreYOLO на модель висновку PaddlePaddle за допомогою
  X2Paddle: закріплений ланцюжок інструментів, статичні пакетні графіки 1 FP32 і
  висновки CPU.
lead: >-
  Моделі висновків PaddlePaddle, це графік model.pdmodel поруч із файлом ваги
  model.pdiparams. LibreYOLO експортує статичний графік opset-15 ONNX,
  перетворює його за допомогою X2Paddle і пакує результат із metadata.yaml, щоб
  він завантажувався через ту саму фабрику, що й будь-яке інше середовище
  виконання.
keywords:
  - yolo весло експорт
  - весловесло висновок
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx опсет 15
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="paddle")
    mono: true
  - label: Writes
    value: 'Каталог з model.pdmodel, model.pdiparams і metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Backend
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Shapes
    value: 'Статичний, партія 1, опсет 15. Усі три виконуються.'
  - label: Precision
    value: 'Лише FP32, Лише CPU.'
  - label: Toolchain
    value: 'PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 або раніше, точно перевірено'
verification: >-
  Читання з libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md і
  pyproject.toml у гілці dev.
snippets:
  install:
    - label: Install
      language: bash
      code: |
        # Python 3.10 до 3.12. WSL2 з Ubuntu 22.04, це перевірений шлях Windows.
        pip install "libreyolo[paddle]"
    - label: Confirm the pinned versions
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
    - label: Arguments
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; це сімейне квадратне полотно
            batch=1,          # будь-яке інше значення викликає ValueError
            dynamic=False,    # True викликає ValueError
            simplify=True,    # False викликає ValueError
            opset=15,         # будь-яке інше значення викликає ValueError
            output_path=None, # Жоден не записує ваги/<stem>_весло
        )
  run:
    - label: Through LibreYOLO
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
    - label: The backend directly
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # Що LibreYOLO() створює для каталогу Paddle. Такі ж результати
        # об'єкт, без фабричної маршрутизації між ними.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Bare Paddle
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

        # Попередня обробка та постобробка є вашими на цьому шляху.
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Встановлення

<code-tabs name="install" />

Додаткові контакти точний стек парності роботи виміряно: PaddlePaddle 2.6.2,
X2Paddle 1.6.0 і ONNX 1.17 або раніше. Ці шпильки перевіряються під час експорту,
не тільки під час встановлення, а інша версія викликає іменування `ImportError`
очікуваний. Новіші випуски Paddle відхиляють частини статичного коду X2Paddle
1.6.0 генерує, тому завчасна помилка краще, ніж створення артефакту, якого ніхто не має
підтверджено.

## Експорт

<code-tabs name="export" />

Чотири аргументи є фіксованими, а не типовими. `dynamic` має бути `False`, `batch`
має бути 1, `simplify` має бути `True` для повністю статичного графіка перетворення, і
`opset` має бути 15, тобто стеля, яку приймає X2Paddle 1.6.0. Передача будь-чого
else викликає перед трасуванням.

Одна нормалізація виконується на проміжному графіку. ONNX визначає пропущений MaxPool
розширення як один, PyTorch записує явний атрибут all-ones, а X2Paddle
1.6.0 відхиляє його, тому експортер видаляє це надлишкове значення за замовчуванням і залишає
зазначена операція без змін.

Артефактом є каталог: `model.pdmodel`, `model.pdiparams` і
`metadata.yaml`. Python, який X2Paddle генерує під час перетворення, не є частиною
цього.

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` розпізнає будь-який каталог, що містить як `model.pdmodel`, так і
`model.pdiparams`, читає `metadata.yaml` і повертає той самий об’єкт `Results`, що й
контрольно-пропускний пункт. Пристрій, відмінний від `auto` або `cpu`, викликає: це серверна частина CPU
тільки.

Завод виготовляє `PaddleBackend`, експортований із `libreyolo` і
імпортується як `libreyolo.backends.paddle.PaddleBackend`. Сконструюйте його самостійно
коли вам потрібен бекенд без маршрутизації фабричного суфікса, наприклад to
передайте `task=` явно для каталогу, `metadata.yaml` якого ви не записали.
Його `predict()` використовує ті самі джерела та повертає ті самі результати.

Фрагмент простого виконання відображає те, що налаштовує бекенд, і три
вимкнені параметри є навмисними. Термоконвеєр Paddle 2.6 CPU може вийти з ладу
одночасно оптимізуючи великі графіки сейсмічних і розкидних випромінюваних для деформівних
Увага, тому портативний необ’єднаний статичний граф є єдиною парністю
проти. Попередня обробка, декодування, NMS і масштабування координат стають вашими на
той шлях.

## Обмеження

Без динамічних форм, без FP16, без INT8, без вбудованого NMS, без середовища виконання GPU.

Перевірені комбінації: виявлення YOLO9, виявлення YOLO9-E2E і виявлення YOLO9-P2, EC
виявлення, поза та сегментація, виявлення RT-DETRv4, D-FINE, DEIM та DEIMv2,
і виявлення та пози YOLO-NAS. Кожен охоплюється перетворенням, середовищем виконання CPU
перезавантаження, паритет необроблених вихідних даних і відповідні публічні результати.

Заблоковано, із записом причини для кожної комбінації:

| Комбінація | чому |
|---|---|
| RF-DETR, всі завдання | Потрібен ONNX opset 17 і GridSample; X2Paddle 1.6.0 приймає опсет 15 або нижчий і не має картографа GridSample |
| Виявлення RT-DETR і RT-DETRv2 | Для навчених графіків потрібен GridSample із параметром 16 або новішим |
| Сегментація D-FINE | Перетворює та перезавантажує, але відносна середньоквадратична помилка logit маски становить 3.52%, а мінімальна відповідна маска IoU становить 0.582 |
| Сегментація YOLO9 | YOLO9 виявляється лише в LibreYOLO |
| Сегментація RTMDet-Ins | Декодування маски динамічного ядра не має експортованого контракту виконання |

Все, що не вказано як підтверджене або заблоковане, відхиляється з приміткою про наявність
не було перевірено шляхом перетворення ONNX у Paddle.

Повну таблицю сімейства та завдань див
[матриця експорту](/docs/reference/export-matrix). Для однієї комбінації:

<code-tabs name="support" />

