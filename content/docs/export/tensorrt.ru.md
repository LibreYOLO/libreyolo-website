---
title: TensorRT
seo_title: Экспорт в TensorRT из LibreYOLO
description: >-
  Сборка TensorRT-движка из модели LibreYOLO: промежуточный ONNX, сборки FP16 и
  INT8, профили динамического батча и пределы переносимости движка.
lead: >-
  TensorRT компилирует граф в движок, настроенный под одну конкретную GPU.
  LibreYOLO сначала экспортирует промежуточный ONNX, разбирает его ONNX-парсером
  TensorRT, собирает движок и записывает рядом метаданные модели в виде
  sidecar-файла JSON.
keywords:
  - экспорт yolo в tensorrt
  - tensorrt engine
  - trt fp16
  - калибровка int8 tensorrt
  - профиль оптимизации tensorrt
  - динамический батч tensorrt
  - hardware compatibility level
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="tensorrt")
    mono: true
  - label: Записывает
    value: Один файл .engine плюс sidecar-файл метаданных .engine.json
  - label: Дополнительно
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Загружается обратно
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: Формы
    value: >-
      По умолчанию статические; dynamic=True добавляет профиль оптимизации по
      оси батча
  - label: Точность
    value: 'FP32, FP16 (half=True), INT8 (int8=True вместе с data=)'
  - label: Требуется
    value: >-
      GPU NVIDIA и при сборке, и при запуске. Движки не переносятся между
      архитектурами GPU.
verification: >-
  Прочитано из libreyolo/export/tensorrt.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tensorrt.py и pyproject.toml в
  ветке dev.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        # Движок собирается из промежуточного ONNX, поэтому нужны оба extra.
        pip install "libreyolo[onnx,tensorrt]"
    - label: Проверка тулчейна перед сборкой
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


        # Записывает weights/LibreYOLO9t_fp16.engine и
        weights/LibreYOLO9t_fp16.engine.json

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # обязателен при int8=True
            dynamic=False,
            workspace=4.0,                  # ГиБ временной памяти при сборке
            min_batch=1,                    # границы динамического профиля
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # или "ampere_plus"
            gpu_device=0,                   # устройство сборки на хосте с несколькими GPU
            verbose=False,
        )
  dynamic:
    - label: Движок с динамическим батчем
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Промежуточному ONNX нужна динамическая ось батча, иначе профилю
        # не к чему будет привязаться.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 с калибровочными данными
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # обязателен: значения по умолчанию для этого формата нет
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
    - label: Голый TensorRT
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

        # Имена классов, задача и размер входа лежат в sidecar-файле, а не в
        движке.

        # Выделение буферов, предобработка и постобработка здесь на вашей
        стороне.

        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Проверка одного семейства и задачи перед сборкой
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## Установка

И сборке, и запуску нужна GPU NVIDIA с рабочим стеком CUDA. Запасного пути через
CPU у этого формата нет.

<code-tabs name="install" />

Extra `tensorrt` фиксирует `tensorrt-cu12` и `pycuda`, а маркер отключает оба на
macOS. На Jetson этот extra использовать не нужно: он фиксирует сборку под CUDA 12
на платформе с CUDA 13. Вместо него используйте тот TensorRT, который ставит
JetPack, — как описано на странице [NVIDIA Jetson](/docs/export/jetson).

## Экспорт

<code-tabs name="export" />

Экспорт идёт в два шага. Первый записывает промежуточный ONNX во временный путь,
второй разбирает его и собирает движок, после чего промежуточный файл удаляется.
`workspace` — это временная память при сборке, в ГиБ; большее значение позволяет
сборщику перебрать больше ядер и на память при инференсе не влияет.

Sidecar-файл метаданных записывается рядом с движком как `<engine>.json` и
фиксирует ту точность, которую сборка реально получила. Если у GPU нет быстрого
FP16 или быстрого INT8, сборщик предупреждает и откатывается, и в sidecar-файле
указана та точность, которая вышла, а не та, которую запрашивали.

В режиме FP16 ViT-бэкбон в графе распознаётся, и его вещественные слои
закрепляются за FP32. Бэкбоны в духе DINOv2 переполняются в FP16 и дают NaN,
поэтому сборка выставляет `OBEY_PRECISION_CONSTRAINTS` и сообщает
`FP16 (FP32 ViT backbone)`. На CNN-бэкбонах этот проход ничего не делает.

### Динамический батч

<code-tabs name="dynamic" />

`dynamic=True` добавляет один профиль оптимизации от `min_batch` до `max_batch`,
оптимизированный под `opt_batch`, и записывает эти три значения в sidecar-файл.
Профиль добавляется только тогда, когда промежуточный ONNX действительно несёт
динамическую размерность батча; иначе сборка пишет в лог, что использует
статическую оптимизацию, и продолжает.

### INT8

<code-tabs name="int8" />

INT8 использует энтропийный калибратор TensorRT поверх калибровочного загрузчика
LibreYOLO, и `data` обязателен: запасного варианта на восьми изображениях у этого
формата нет. Калибровке нужен `cuda-python` или `pycuda` для буфера на устройстве.
Ключом кэша калибровки служит хеш байтов ONNX, поэтому масштабы от одной модели
никогда не переиспользуются для другой, которая случайно пишет по тому же
выходному пути.

`half=True` и `int8=True` вместе дают предупреждение и собирают INT8, который
оставляет запасной FP16 для слоёв, которые TensorRT не может квантизовать.

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` выбирает путь по суффиксу `.engine`, берёт из sidecar-файла имена
классов, задачу и схему позы и возвращает тот же объект `Results`, что и чекпойнт.
При отсутствии CUDA-устройства он сразу падает с ошибкой.

Второй сниппет — путь через голую среду выполнения. Выделение буферов на хосте и
на устройстве, предобработка, декодирование, NMS и пересчёт координат становятся
вашей задачей, а сам движок не несёт имён классов, поэтому sidecar-файл должен
путешествовать вместе с ним.

## Ограничения

Сериализованный движок привязан к архитектуре GPU, стеку драйверов и версии
TensorRT, которая его собрала. Движок, собранный на рабочей станции, не
загрузится на другой архитектуре — поэтому шаг сборки выполняется на той машине,
куда идёт развёртывание. `hardware_compatibility="ampere_plus"` меняет часть
производительности на переносимость между Ampere и более новыми архитектурами.
Значение `"same_compute_capability"` отображается в `NONE` и выдаёт
предупреждение: движок оптимизирован только под текущую GPU, и экспорт прямо
говорит об этом, а не заявляет переносимость, которую не применял.

Профилируется только ось батча. Сборка с динамическими пространственными
размерностями в этот контракт не входит — поэтому FCOS заблокирован: ему нужны
динамические высота и ширина с паддингом, чтобы сохранить своё преобразование с
соотношением 800 на 1333.

Заблокировано до трассировки: сегментация YOLO9, сегментация RTMDet-Ins, детекция
SSD, Faster R-CNN и RetinaNet, а также маттинг BiRefNet или FeyNobg — там
TensorRT 10.16 доходит до общего ONNX-узла `DeformConv` и не может его разобрать,
потому что `ModulatedDeformConv2d` отсутствует в реестре плагинов.

Там, где комбинация не валидирована и не заблокирована, путь конвертации доступен,
а паритет со средой выполнения TensorRT для неё проект не фиксировал. Это
утверждение о свидетельствах, а не о том, соберётся ли движок.

Полную сетку семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). Для одной комбинации:

<code-tabs name="support" />
