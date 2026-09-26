---
title: OpenVINO
seo_title: Експорт до OpenVINO IR з LibreYOLO
description: >-
  Перетворіть модель LibreYOLO на OpenVINO IR: пару model.xml і model.bin,
  стиснення ваг FP16, INT8 через NNCF та інференс на CPU, GPU або NPU.
lead: >-
  OpenVINO IR є форматом середовища виконання Intel, що складається з графа
  model.xml і блоба ваг model.bin. LibreYOLO експортує проміжний файл ONNX,
  перетворює його за допомогою ov.convert_model і записує metadata.yaml у той
  самий каталог.
keywords:
  - експорт yolo openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - nncf квантування int8
  - openvino npu
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: Прапорець
    value: export(format="openvino")
    mono: true
  - label: Створює
    value: 'Каталог із model.xml, model.bin і metadata.yaml'
  - label: Додатково
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Повторне завантаження
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: Форми
    value: 'Відповідають проміжному файлу ONNX: динамічний батч із dynamic=True'
  - label: Точність
    value: 'FP32, стиснення ваг FP16 (half=True), INT8 через NNCF (int8=True із data=)'
verification: >-
  Перевірено за файлами libreyolo/export/openvino.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py,
  libreyolo/backends/openvino.py і pyproject.toml у гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: >
        # IR перетворюється з проміжного файлу ONNX, тому потрібні обидва набори
        залежностей.

        pip install "libreyolo[onnx,openvino]"
    - label: Для INT8 додатково потрібен NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Створює каталог weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True зберігає динамічну вісь батча в IR
            half=False,       # True зберігає ваги FP16
            int8=False,       # True запускає квантування NNCF після навчання
            data=None,        # обов'язково, коли int8=True
            output_path=None, # None створює weights/<stem>_openvino
        )
  int8:
    - label: INT8 із калібрувальними даними
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # обов'язково: для цього формату немає типового значення
            fraction=1.0,
        )
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Вибір пристрою
      language: python
      code: >
        from libreyolo import LibreYOLO


        # "auto" і "cpu" зіставляються з CPU, "gpu" і "cuda" зіставляються з
        GPU,

        # усе інше передається у верхньому регістрі, наприклад "npu" -> NPU.

        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: Безпосередньо через OpenVINO
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # Назви класів, завдання й розмір входу зберігаються в metadata.yaml
        поряд з IR.

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # На цьому шляху попередня та подальша обробка покладаються на вас.
  support:
    - label: Перевірити сімейство й завдання перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## Встановлення

<code-tabs name="install" />

Перетворення проходить через проміжний файл ONNX, тому набір залежностей `onnx`
є обов'язковим, а не додатковим. NNCF установлюється окремо й потрібен лише для
`int8=True`.

## Експорт

<code-tabs name="export" />

Артефакт є каталогом, а не файлом. `weights/LibreYOLO9t_openvino` містить
`model.xml`, `model.bin` і `metadata.yaml`, а з `half=True` перед суфіксом
вставляється `_fp16`. Переміщуйте або копіюйте весь каталог, оскільки ці три
файли утворюють один артефакт.

`half=True` задає `compress_to_fp16` під час збереження. Це стиснення ваг у IR,
а не зміна точності інференсу, яку пристрій вибирає під час виконання.

### INT8

<code-tabs name="int8" />

`int8=True` запускає квантування NNCF після навчання з набором mixed через
завантажувач калібрувальних даних LibreYOLO. Параметр `data` обов'язковий, адже
для цього формату немає резервного набору з восьми зображень. Якщо NNCF не
встановлено, виникає `ImportError` із назвою команди встановлення.

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` розпізнає будь-який каталог із `model.xml` і повертає той самий
об'єкт `Results`, що й контрольна точка, читаючи назви класів, завдання, розмір
входу та схему пози з `metadata.yaml`.

Рядок пристрою зіставляється, а не передається безпосередньо. `auto` та `cpu`
компілюють для CPU, `gpu` та `cuda` компілюють для GPU, а будь-яке інше значення
перетворюється на верхній регістр і передається OpenVINO. Саме так вибирають NPU.

Третій фрагмент призначено для читачів без установленої LibreYOLO. На цьому
шляху попередня обробка, декодування, NMS і масштабування координат покладаються
на вас, а назви класів зберігаються лише в `metadata.yaml`.

## Обмеження

IR без `metadata.yaml` усе одно завантажується, але тоді бекенд використовує
резервні 80 класів і завдання виявлення, що неправильно для всіх інших випадків.
Зберігайте каталог цілісним.

Перед трасуванням блокуються: сегментація YOLO9, сегментація RTMDet-Ins,
виявлення SSD, Faster R-CNN і RetinaNet, а також matting за допомогою BiRefNet
або FeyNobg, де OpenVINO 2026.2 не може знизити стандартну операцію ONNX
`DeformConv-19` зі спільного декодера matte.

Якщо комбінацію не валідовано й не заблоковано, шлях конвертера доступний, але
проєкт не має записаного паритету для середовища виконання OpenVINO. Для кількох
комбінацій валідацію записано з явним контекстом. Наприклад, семантичну
сегментацію DeepLabV3 валідовано з фіксованим входом 520 на 520 у OpenVINO 2026.2
із типовою точністю інференсу на CPU, а оцінювання погляду L2CS із фіксованим
кадруванням обличчя 448 на 448. `libreyolo formats` виводить цей контекст для
кожної комбінації.

Повну таблицю сімейств і завдань наведено в
[матриці експорту](/docs/reference/export-matrix). Для однієї комбінації:

<code-tabs name="support" />
