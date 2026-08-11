---
title: TFLite
seo_title: Экспорт в TFLite (LiteRT) из LibreYOLO
description: >-
  Экспорт модели LibreYOLO в FlatBuffer .tflite через onnx2tf: статические
  формы, только FP32, входы NHWC и семейства, которые конвертируются без
  проблем.
lead: >-
  TFLite — это формат FlatBuffer, который LiteRT исполняет на мобильных и
  встраиваемых устройствах. LibreYOLO экспортирует статический граф ONNX,
  конвертирует его через onnx2tf в режиме flatbuffer-direct и кладёт метаданные
  модели рядом с артефактом в виде сопроводительного JSON-файла.
keywords:
  - экспорт yolo в tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - вход nhwc tflite
  - инференс на edge-устройствах
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="tflite")
    mono: true
  - label: Записывает
    value: Один файл .tflite плюс сопроводительный файл метаданных .tflite.json
  - label: Дополнительно
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Загружается обратно
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: Формы
    value: Только статические. dynamic=True отклоняется.
  - label: Точность
    value: Только FP32. half=True и int8=True отклоняются.
  - label: Требуется
    value: >-
      Python 3.12 или новее, потому что onnx2tf 2.4.x не публикует wheel-пакеты
      под более старые версии
verification: >-
  Прочитано из libreyolo/export/tflite.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tflite.py и pyproject.toml в
  ветке dev.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        # LiteRT — текущее название TensorFlow Lite у Google. Оба extra ставят
        # один и тот же тулчейн и дают на выходе один и тот же .tflite.
        pip install "libreyolo[tflite]"
    - label: Проверка версии Python
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Записывает weights/LibreYOLO9t.tflite и
        weights/LibreYOLO9t.tflite.json

        path = model.export(format="tflite", imgsz=640)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" принимается как псевдоним и ведёт к тому же экспортёру.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int или (высота, ширина)
            batch=1,
            simplify=True,    # onnxsim поверх промежуточного ONNX
            output_path=None, # None пишет в weights/<stem>.tflite
            verbose=False,    # True выводит лог onnx2tf
        )

        # dynamic=True вызывает ValueError: конвертеру нужны статические формы.
        # half=True и int8=True отклоняются ещё до трассировки.
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Чистый LiteRT
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC, а не NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Имена классов, задача и размер входа лежат в сопроводительном файле.

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Предобработка, транспонирование NCHW в NHWC и постобработка — на вас.
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## Установка

<code-tabs name="install" />

Этот extra подтягивает `onnx2tf` для конвертации и `ai-edge-litert` для запуска
результата — обе зависимости за маркером Python 3.12. На более старом
интерпретаторе экспорт вызывает `ImportError`, который называет требование к
версии, а не падает внутри конвертера.

`libreyolo[litert]` ставит ровно то же самое. Строка формата `litert` —
псевдоним для `tflite`, и выходной файл в обоих случаях `.tflite`.

## Экспорт

<code-tabs name="export" />

Семейство и задача проверяются раньше всего остального, поэтому неподдерживаемая
комбинация падает сразу же с конкретной ошибкой конвертера или среды выполнения,
из-за которой её и не пустили, а не с общим сообщением. Сама конвертация — вызов
`onnx2tf` в подпроцессе в режиме `flatbuffer_direct` поверх статического
промежуточного ONNX.

Метаданные лежат в сопроводительном файле. `weights/LibreYOLO9t.tflite.json`
несёт семейство, задачу, имена классов, размер входа и схему позы; в самом
FlatBuffer поля метаданных LibreYOLO нет, поэтому два файла ходят вместе.

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` выбирает ветку по суффиксу `.tflite` и возвращает тот же объект
`Results`, что и чекпойнт. Бэкенд читает сопроводительный файл, транспонирует
тензор NCHW в NHWC, когда интерпретатор просит вход с каналами в конце,
применяет масштаб квантизации и нулевую точку интерпретатора там, где они есть,
и транспонирует выходы обратно в раскладку, которую ждёт постобработка
LibreYOLO.

Второй сниппет — путь через чистую среду выполнения. Предобработка,
транспонирование раскладки, декодирование, NMS и пересчёт координат там целиком
на вас, а деталь с раскладкой упускают чаще всего: onnx2tf выдаёт входы с
каналами в конце, поэтому тензор формы `(1, 3, 640, 640)` не привяжется.

## Ограничения

Только статические формы. `dynamic=True` вызывает `ValueError` ещё до
трассировки, а холст экспорта фиксируется на том значении, в которое разрешился
`imgsz`.

Только FP32. `half=True` и `int8=True` отклоняются на валидации, поэтому
развёртывание с квантизацией через этот экспортёр сегодня недоступно.

Покрытие здесь уже, чем у графовых форматов, и определяется измерениями, а не
семейством. Среди проверенных комбинаций — детекция на YOLO9, YOLOX и YOLO-NAS,
семантическая сегментация PIDNet, четыре семейства CNN-классификации, эмбеддинги
DINOv2 и SigLIP2, классификация SigLIP2, выделение границ TEED и DexiNed,
восстановление Real-ESRGAN и SwinIR. У SwinIR есть дополнительная оговорка:
паритет держится, когда размеры источника точно совпадают с холстом экспорта, а
источники меньшего размера дополняются до холста перед запуском трансформера, и
это может расходиться с нативным инференсом на переменном размере.

Заблокированные записи называют конкретную причину отказа, и её стоит прочитать
до попыток обойти ограничение. Несколько примеров: детекция RF-DETR
конвертируется на своём нативном холсте 384, но LiteRT не может её разместить,
потому что `STRIDED_SLICE` получает вход выше поддерживаемого ранга 5-D; PicoDet
отклоняется, потому что `RESHAPE` отображает 19 200 входных элементов в 9 600
выходных; D-FINE роняет конвертер на обработке форм в `GatherElements`; RTMDet
экспортируется и загружается обратно с сохранением сырого паритета, но публичные
рамки падают до 0.911 IoU с расхождением координат в 29.9 px.

Полную сетку семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). Для одной комбинации, вместе
со строкой причины, стоящей за блокировкой:

<code-tabs name="support" />
