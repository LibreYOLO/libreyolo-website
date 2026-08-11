---
title: ncnn
seo_title: Експорт до ncnn з LibreYOLO
description: >-
  Експортуйте модель LibreYOLO до ncnn через PNNX: пару param і bin, фіксоване
  полотно експорту, перетворення Focus для YOLOX та підтримувані сімейства.
lead: >-
  ncnn є бібліотекою інференсу Tencent для мобільних цільових платформ із CPU.
  LibreYOLO виконує перетворення через PNNX і записує граф model.ncnn.param
  поряд із файлом ваг model.ncnn.bin та файлом metadata.yaml, що містить
  сімейство, завдання й назви класів.
keywords:
  - експорт yolo ncnn
  - pnnx
  - model.ncnn.param
  - інференс на мобільному cpu
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: Прапорець
    value: export(format="ncnn")
    mono: true
  - label: Створює
    value: 'Каталог із model.ncnn.param, model.ncnn.bin і metadata.yaml'
  - label: Додатково
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Повторне завантаження
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: Форми
    value: Фіксовані. Незалежно від прапорця метадані записують dynamic=False.
  - label: Точність
    value: Лише FP32. half=True та int8=True відхиляються.
verification: >-
  Перевірено за файлами libreyolo/export/ncnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/ncnn.py і pyproject.toml у
  гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: |
        # pnnx виконує перетворення, а ncnn запускає результат.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Створює каталог weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # ціле число або (висота, ширина)
            batch=1,
            simplify=True,    # застосовується лише до резервного шляху ONNX
            opset=None,       # автоматично; лише для резервного шляху ONNX
            output_path=None, # None створює weights/<stem>_ncnn
        )

        # half=True та int8=True відхиляються під час валідації.
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Безпосередньо через ncnn
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnn приймає одне зображення CHW, а не батч.
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # На цьому шляху попередня та подальша обробка покладаються на вас.
  support:
    - label: Перевірити сімейство й завдання перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## Встановлення

<code-tabs name="install" />

Додатковий набір залежностей установлює обидві частини ланцюжка інструментів:
`pnnx` виконує перетворення, а `ncnn` запускає результат. Основний шлях не
використовує ONNX для жодної з цих частин.

## Експорт

<code-tabs name="export" />

Артефакт є каталогом. `weights/LibreYOLO9t_ncnn` містить
`model.ncnn.param`, `model.ncnn.bin` і `metadata.yaml`. Усі три файли утворюють
один артефакт і переміщуються разом.

Спочатку PNNX намагається виконати перетворення безпосередньо з PyTorch. Якщо це
не вдається, статичний граф ONNX експортується до тимчасового каталогу й для
нього викликається інструмент командного рядка `pnnx`. Експорт завершується
помилкою лише тоді, коли не працюють обидва шляхи, і повідомляє про обидві
помилки. Тому `opset` і `simplify` впливають лише на резервний шлях.

Для перетворення YOLOX потрібна одна заміна. Його шар Focus використовує зрізи з
кроком, які PNNX не може знизити, тому під час експорту їх замінюють на
`pixel_unshuffle` і переставляють вхідні канали наступної згортки, щоб
компенсувати інший порядок каналів. Результат чисельно ідентичний, а початкові
ваги відновлюються після експорту.

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` розпізнає будь-який каталог із `model.ncnn.param` і
`model.ncnn.bin`, читає `metadata.yaml` та повертає той самий об'єкт `Results`,
що й контрольна точка.

Другий фрагмент показує шлях безпосередньо через середовище виконання. Дві його
особливості відрізняються від усіх інших форматів на цій сторінці. ncnn працює з
одним зображенням CHW, а не з батчем, тому початкової осі батча немає. Назви
блобів надходять із файлу `.param`. За домовленістю PNNX записує `in0` і `out0`,
а бекенд розбирає файл замість того, щоб покладатися на ці назви. На цьому шляху
попередня обробка, декодування, NMS і масштабування координат покладаються на вас.

## Обмеження

FP32 на фіксованому полотні. `half=True` та `int8=True` відхиляються під час
валідації, а експортовані метадані записують `dynamic=False` незалежно від
значення прапорця, щоб бекенд не припускав наявність осі, якої немає в графі.

На попередній перевірці відхиляється кожне сімейство в стилі DETR: `detr`,
`deformable_detr`, `dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rfdetr` і `ec`. Для всіх них повідомлення однакове:
моделі потрібні операції декодера або семплювання, недоступні в ncnn. Замість
цього воно пропонує ONNX, OpenVINO, TorchScript або TensorRT.

На основі згорток перетворюється широкий набір моделей: YOLO9 і YOLO9-E2E,
YOLOX, PicoDet, виявлення та оцінювання пози YOLO-NAS, старіші детектори YOLO1,
YOLO3, YOLO4 і YOLO7, чотири сімейства класифікації CNN, семантична сегментація
PIDNet, виявлення точок FOMO на фіксованому полотні 96 на 96, ZipDepth, NAFNet і
Real-ESRGAN.

Для заблокованих комбінацій указано конкретну причину збою. Графи
трансформерів зазвичай залишають непідтримувані вузли `pnnx.Expression`, через
що створюється мережа без придатного до запуску вхідного блоба. Саме тому
зупиняються DINOv2, CLIP, SigLIP2 і SegFormer. BiRefNet потребує деформованої
згортки torchvision, яку PNNX не може знизити. Перетворений граф YOLO2 завершує
роботу середовища ncnn у Windows через нативне цілочисельне ділення на нуль під
час отримання виходу.

Повну таблицю сімейств і завдань наведено в
[матриці експорту](/docs/reference/export-matrix). Для однієї комбінації:

<code-tabs name="support" />
