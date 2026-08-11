---
title: TensorRT
seo_title: Експорт у TensorRT із LibreYOLO
description: >-
  Побудова рушія TensorRT із моделі LibreYOLO: проміжний файл ONNX, побудови
  FP16 та INT8, профілі динамічного батча й обмеження переносності рушія.
lead: >-
  TensorRT компілює граф у рушій, налаштований для одного GPU. LibreYOLO
  спочатку експортує проміжний файл ONNX, аналізує його синтаксичним
  аналізатором ONNX у TensorRT, будує рушій і записує метадані моделі поряд у
  допоміжному файлі JSON.
keywords:
  - експорт yolo tensorrt
  - рушій tensorrt
  - trt fp16
  - калібрування tensorrt int8
  - профіль оптимізації tensorrt
  - динамічний батч tensorrt
  - hardware compatibility level
last_verified: 1.5.0
meta:
  - label: Параметр
    value: export(format="tensorrt")
    mono: true
  - label: Результат
    value: Один файл .engine і допоміжний файл метаданих .engine.json
  - label: Додатково
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Повторне завантаження
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: Форми
    value: Типово статичні; dynamic=True додає профіль оптимізації осі батча
  - label: Точність
    value: 'FP32, FP16 (half=True), INT8 (int8=True із data=)'
  - label: Вимоги
    value: >-
      GPU NVIDIA під час побудови та виконання. Рушії не переносяться між
      архітектурами GPU.
verification: >-
  Перевірено за файлами libreyolo/export/tensorrt.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py,
  libreyolo/backends/tensorrt.py та pyproject.toml у гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: >
        # Рушій будується з проміжного файлу ONNX, тому потрібні обидва
        додаткові пакети.

        pip install "libreyolo[onnx,tensorrt]"
    - label: Перевірити набір інструментів перед побудовою
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Записує weights/LibreYOLO9t_fp16.engine і
        weights/LibreYOLO9t_fp16.engine.json

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # обов'язковий, коли int8=True
            dynamic=False,
            workspace=4.0,                  # GiB тимчасової пам'яті під час побудови
            min_batch=1,                    # межі динамічного профілю
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # або "ampere_plus"
            gpu_device=0,                   # пристрій побудови на хості з кількома GPU
            verbose=False,
        )
  dynamic:
    - label: Рушій із динамічним батчем
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Проміжний файл ONNX повинен мати динамічну вісь батча,
        # до якої можна прив'язати профіль.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 із калібрувальними даними
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # обов'язково: для цього формату немає типового значення
            fraction=1.0,
        )
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Безпосередньо TensorRT
      language: python
      code: >
        import json


        import tensorrt as trt


        path = "weights/LibreYOLO9t_fp16.engine"

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # Назви класів, завдання та розмір вхідних даних зберігаються в
        допоміжному файлі, а не в рушії.

        # Тут розподіл буферів, попередня й подальша обробка покладаються на
        вас.

        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Перевірити сімейство й завдання перед побудовою
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## Встановлення

Для побудови й виконання потрібен GPU NVIDIA із робочим стеком CUDA. Для цього
формату немає резервного виконання на CPU.

<code-tabs name="install" />

Додатковий пакет `tensorrt` закріплює `tensorrt-cu12` і `pycuda`, а маркер
виключає обидва в macOS. Не використовуйте цей додатковий пакет на Jetson: він
закріплює збірку CUDA 12 для платформи CUDA 13. Натомість використовуйте TensorRT,
встановлений JetPack, як описано на сторінці
[NVIDIA Jetson](/docs/export/jetson).

## Експорт

<code-tabs name="export" />

Експорт виконується у два етапи. Перший етап записує проміжний файл ONNX за
тимчасовим шляхом, другий аналізує його й будує рушій, після чого проміжний файл
вилучається. `workspace` визначає тимчасову пам'ять побудови в GiB; більше значення
дає побудовнику змогу випробувати більше ядер і не впливає на пам'ять інференсу.

Допоміжний файл метаданих записується поряд із рушієм як `<engine>.json` і фіксує
точність, фактично отриману під час побудови. Якщо GPU не має швидких FP16 або
INT8, побудовник показує попередження та переходить на іншу точність, а допоміжний
файл повідомляє отриману, а не запитану точність.

Для FP16 виявляється бекбон ViT у графі, а його шари з рухомою комою закріплюються
у FP32. Бекбони в стилі DINOv2 переповнюються у FP16 і створюють NaN, тому під час
побудови встановлюється `OBEY_PRECISION_CONSTRAINTS` і повідомляється
`FP16 (FP32 ViT backbone)`. Для бекбонів CNN цей прохід нічого не змінює.

### Динамічний батч

<code-tabs name="dynamic" />

Параметр `dynamic=True` додає один профіль оптимізації від `min_batch` до
`max_batch`, оптимізований для `opt_batch`, і записує ці три значення в допоміжний
файл. Профіль додається лише тоді, коли проміжний файл ONNX справді містить
динамічний вимір батча; інакше побудовник записує в журнал, що використовує
статичну оптимізацію, та продовжує роботу.

### INT8

<code-tabs name="int8" />

INT8 використовує ентропійний калібратор TensorRT із завантажувачем калібрування
LibreYOLO, а `data` є обов'язковим: для цього формату немає резервного набору з
восьми зображень. Для буфера пристрою під час калібрування потрібен `cuda-python`
або `pycuda`. Кеш калібрування пов'язаний із хешем байтів ONNX, тому масштаби
однієї моделі ніколи не використовуються для іншої, яка випадково записується за
тим самим вихідним шляхом.

Поєднання `half=True` та `int8=True` створює попередження й будує INT8, зберігаючи
резервний шлях FP16 для шарів, які TensorRT не може квантувати.

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` виконує диспетчеризацію за суфіксом `.engine`, читає з допоміжного
файлу назви класів, завдання та схему пози й повертає той самий об'єкт `Results`,
що й контрольна точка. Якщо пристрою CUDA немає, одразу виникає помилка.

Другий фрагмент показує шлях із безпосереднім середовищем виконання. Розподіл
буферів хоста й пристрою, попередня обробка, декодування, NMS та повторне
масштабування координат покладаються на вас, а сам рушій не містить назв класів,
тому допоміжний файл потрібно переносити разом із ним.

## Обмеження

Серіалізований рушій прив'язаний до архітектури GPU, стека драйверів і версії
TensorRT, яка його побудувала. Рушій, побудований на робочій станції, не
завантажиться на іншій архітектурі, тому етап побудови виконується на машині
розгортання. Значення `hardware_compatibility="ampere_plus"` частково жертвує
продуктивністю заради переносності між Ampere та новішими архітектурами.
Значення `"same_compute_capability"` зіставляється з `NONE` та створює
попередження: рушій оптимізовано лише для поточного GPU, і експорт повідомляє
про це, а не заявляє про переносність, якої не застосовано.

Профілюється лише вісь батча. Побудова з динамічними просторовими вимірами не
входить до цього контракту, тому FCOS заблоковано: для збереження перетворення
пропорцій 800 на 1333 потрібні динамічні висота й ширина з доповненням.

До трасування блокуються сегментація YOLO9, сегментація RTMDet-Ins, виявлення SSD,
Faster R-CNN і RetinaNet, а також matting BiRefNet або FeyNobg, де TensorRT 10.16
доходить до спільного вузла ONNX `DeformConv` і не може його проаналізувати через
відсутність `ModulatedDeformConv2d` у реєстрі плагінів.

Якщо поєднання не валідовано й не заблоковано, шлях перетворювача доступний, але
проєкт не зафіксував для нього еквівалентність середовища TensorRT. Це твердження
про наявні докази, а не про успішність побудови.

Повну сітку сімейств і завдань наведено в
[матриці експорту](/docs/reference/export-matrix). Для окремого поєднання:

<code-tabs name="support" />

