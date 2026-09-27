---
title: ONNX
seo_title: Экспорт в ONNX из LibreYOLO
description: >-
  Экспорт модели LibreYOLO в ONNX: какой opset LibreYOLO выбирает для каждого
  семейства, динамические оси, встроенный NMS, INT8 и как граф загружается
  обратно.
lead: >-
  ONNX — переносимый формат графа. LibreYOLO трассирует модель через
  torch.onnx.export, при необходимости упрощает граф и записывает семейство,
  задачу, имена классов и размер входа в собственные метаданные файла, чтобы
  любой бэкенд LibreYOLO мог восстановить постобработку.
keywords:
  - экспорт yolo в onnx
  - onnxruntime
  - torch.onnx.export
  - onnx opset
  - динамические оси onnx
  - встроенный nms onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="onnx")
    mono: true
  - label: Записывает
    value: 'Один файл .onnx, метаданные встроены в граф'
  - label: Дополнительно
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Загружается обратно
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: Формы
    value: По умолчанию динамический батч в Python; исключения по задачам ниже
  - label: Точность
    value: 'FP32, FP16 (half=True), INT8 (int8=True, детекция YOLO9)'
verification: >-
  Прочитано из libreyolo/export/onnx.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/onnx.py и
  libreyolo/cli/commands/export.py в ветке dev.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int или (высота, ширина)
            batch=1,
            dynamic=True,     # значение по умолчанию в Python; в CLI по умолчанию False
            simplify=True,    # прогнать onnxsim по графу
            opset=None,       # None выбирает 13 или 17 для семейств в стиле DETR
            half=False,       # веса и активации в FP16
            int8=False,       # QDQ INT8, только детекция YOLO9
            data=None,        # калибровочный data.yaml, только для INT8
            device=None,      # устройство трассировки; None берёт устройство модели
            output_path=None, # None записывает weights/<stem>.onnx
        )
  nms:
    - label: Встраивание NMS в граф
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Только детекция YOLO9, батч 1. dynamic принудительно ставится в False.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 с калибровочными данными
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # несколько сотен представительных изображений
            fraction=1.0,
        )
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Голый ONNX Runtime
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # Предобработка и постобработка на этом пути — ваша задача.

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # Граф несёт в себе семейство, задачу, имена классов и размер входа.

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## Установка

<code-tabs name="install" />

Этот extra подтягивает `onnx`, `onnxsim` и `onnxruntime`. Для записи файла
достаточно одного `onnx`; `onnxsim` выполняет проход упрощения, а `onnxruntime`
запускает артефакт и проводит калибровку INT8.

## Экспорт

<code-tabs name="export" />

Без `output_path` файл попадает в `weights/` под именем чекпойнта, с добавлением
`_fp16` или `_int8`, когда запрошена соответствующая точность.

`dynamic` по умолчанию равен `True` в Python и `False` в CLI. Когда он включён,
ось батча становится символьной, а несколько задач раскрываются ещё шире:
семантическая сегментация открывает также высоту и ширину маски, восстановление
Real-ESRGAN открывает пространственные оси, а двухстадийные детекторы держат
динамическими исходные высоту и ширину, потому что изменение размера у них
происходит внутри графа.

`opset` при его отсутствии выбирается для каждого семейства. Семейства в стиле
DETR (`detr`, `deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`,
`lwdetr`, `rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`), а также `deit`, `midas` и
`moge2` получают opset 17 — именно там понижается `aten::scaled_dot_product`.
Всё остальное получает 13. Маттинг в любом случае поднимается до 19, потому что
декодеру BiRefNet нужен оператор `DeformConv`, который ONNX определяет начиная с
opset 19.

`simplify=True` запускает `onnxsim` и сохраняет исходный граф, если проход не
удался, поэтому ошибка упрощения — это предупреждение, а не сбой экспорта. На
macOS arm64 с `onnx` 1.22 или новее и `onnxsim` 0.6.5 или старее проход
пропускается целиком, потому что эта пара может аварийно завершить процесс
Python.

### Встроенный NMS

<code-tabs name="nms" />

`nms=True` работает только для детекции YOLO9 и требует батча 1; запрос вместе с
`dynamic=True` пишет предупреждение и выключает динамические оси. У графа тогда
два выхода: `output` формы `(batch, max_det, 6)` и `raw` — недекодированный
тензор детектора, который использует собственный бэкенд LibreYOLO, чтобы
постобработка осталась идентичной пути через PyTorch.

### DeepStream

`deepstream=True` — опция только для ONNX. Она экспортирует граф в той
раскладке, которую ожидает парсер NVIDIA DeepStream, и записывает рядом два
вспомогательных файла, `config_infer_primary_<stem>.txt` и `<stem>_labels.txt`,
поэтому артефакт встаёт в пайплайн без конфигурации, написанной вручную.

Она взаимоисключающа с `nms=True`, и запрос обоих поднимает `ValueError`:
DeepStream выполняет подавление на собственной стадии кластеризации. Передача её
любому формату, кроме ONNX, тоже приводит к ошибке. Сетку поддерживаемых
семейств и задач и сборку парсера смотрите в разделе
[DeepStream](/docs/export/deepstream).

### INT8

<code-tabs name="int8" />

`int8=True` запускает статическую квантизацию ONNX Runtime и записывает
QDQ-граф с входами и выходами в float32. Квантизуются только узлы `Conv` и
`Gemm`. Декодирование головы детекции оставлено в float32 намеренно: эта
конкатенация смешивает координаты рамок в масштабе пикселей с оценками классов в
диапазоне от 0 до 1, и единый потензорный масштаб активаций, определяемый
величиной рамок, обнулил бы все оценки.

Сейчас этот флаг применим только к детекции YOLO9, а всё остальное поднимает
`NotImplementedError` на предварительной проверке. Без `data` берётся
`coco8.yaml` с предупреждением; восемь изображений — не представительный
калибровочный набор. Модель, уже квантизованная в PyTorch, идёт другим путём, он
описан в разделе [Квантизация](/docs/export/quantization).

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` распознаёт суффикс `.onnx` и возвращает тот же объект `Results`,
что и чекпойнт `.pt`, потому что имена классов, задача, размер входа и схема
позы были записаны в `metadata_props` графа во время экспорта. При
`device="auto"` сессия берёт `CUDAExecutionProvider`, если ONNX Runtime о нём
сообщает, и иначе откатывается на CPU.

Второй сниппет — для читателей, у которых LibreYOLO не установлен.
Предобработка, декодирование, NMS и пересчёт координат становятся там вашей
задачей; блок метаданных по-прежнему на месте, и его можно прочитать.

## Ограничения

Имена выходных тензоров фиксированы для каждой задачи, и именно их приходится
сопоставлять потребителю, который работает без метаданных:

| Задача | Имена выходов |
|---|---|
| Детекция, сеточные и якорные головы | `output` |
| Детекция, в стиле DETR | `pred_logits`, `pred_boxes` |
| Детекция, RF-DETR | `dets`, `labels` |
| Классификация | `output` |
| Семантическая сегментация | `semantic_logits` |
| Глубина | `depth` |
| Нормали поверхности | `normal` |
| Границы | `edges` |
| Восстановление | `restored` |
| Маттинг | `matte` |
| Взгляд | `yaw_logits`, `pitch_logits` |

RF-DETR — ещё и единственное семейство, у которого входной тензор называется
`input`, а не `images`.

У нескольких задач в этой версии фиксированный контракт разрешения в среде
выполнения. Глубина, нормали поверхности и границы отклоняют `batch != 1` и
принудительно ставят `dynamic=False`. Маттинг принудительно ставит родной
квадрат 1024, потому что таблицы относительных позиций в Swin у BiRefNet
привязаны к своему разрешению. Восстановление принудительно ставит фиксированный
холст для всех семейств, кроме Real-ESRGAN, генератор которого полностью
свёрточный.

Прямоугольный `imgsz` работает для семейств YOLO9, HRNet, NAFNet и Real-ESRGAN.
Семейства с фиксированным квадратным контрактом (`clip`, `deformable_detr`,
`detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `moge2`,
`rtdetr`, `rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`) отклоняют его
сразу.

Две комбинации отклоняются до трассировки: сегментация YOLO9, потому что YOLO9 в
LibreYOLO работает только на детекцию, и сегментация RTMDet-Ins, у которой
декодирование масок с динамическими ядрами не имеет контракта для
экспортированной среды выполнения.

Полную сетку семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). Для одной комбинации спросите
библиотеку напрямую:

<code-tabs name="support" />
