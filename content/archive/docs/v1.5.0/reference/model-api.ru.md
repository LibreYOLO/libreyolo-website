---
title: API модели
seo_title: Методы и сигнатуры объекта модели LibreYOLO
description: >-
  Все методы загруженной модели LibreYOLO: predict, embed, track, val, train,
  export, save, quantize, info и управление CUDA-графами, с реальными значениями
  по умолчанию.
lead: >-
  Загруженная модель LibreYOLO — экземпляр BaseModel. На этой странице
  перечислены методы этого экземпляра с сигнатурами и значениями по умолчанию,
  прочитанными из libreyolo/models/base/model.py.
keywords:
  - методы модели libreyolo
  - аргументы predict libreyolo
  - аргументы val libreyolo
  - аргументы export libreyolo
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  Сигнатуры и значения по умолчанию прочитаны из libreyolo/models/base/model.py
  и libreyolo/models/base/inference.py на версии 1.5.0. Классы семейств могут
  сужать или расширять их; train() определяется в каждом семействе, и здесь
  описана только его общая обёртка cfg=.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True возвращает генератор: один Results на кадр или
        изображение.

        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## Создание

Фабрика возвращает экземпляр класса семейства. Если создавать этот класс
напрямую, аргументы те же, только `size` обязателен:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` выбирает CUDA, если она доступна, затем MPS, затем CPU. Целое
число или строка из цифр читается как порядковый номер устройства CUDA, поэтому
`device=0` и `device="0"` означают одно и то же — `cuda:0`. `task` проверяется
по списку `SUPPORTED_TASKS` семейства. `model_path=None` собирает архитектуру и
оставляет её в режиме обучения; переданный `dict` загружается как state dict
напрямую.

## predict и \_\_call\_\_

`predict` — псевдоним для `__call__`.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `source` | `None` | Изображение, список или кортеж изображений в памяти, каталог, видеофайл либо экранный источник вида `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Порог уверенности |
| `iou` | `0.45` | Порог IoU для NMS |
| `imgsz` | `None` | Переопределение входного размера; при `None` берётся родной размер модели |
| `device` | `None` | Переопределение устройства для этого вызова |
| `classes` | `None` | Оставить только эти ID классов |
| `max_det` | `300` | Максимум детекций на изображение |
| `augment` | `False` | Аугментация на этапе инференса |
| `save` | `False` | Записать изображение или видео с нанесёнными предсказаниями |
| `batch` | `1` | Сколько изображений идёт за один прямой проход для источников-каталогов и списков |
| `stream` | `False` | Вернуть генератор вместо готового списка |
| `stream_buffer` | `False` | Хранить все захваченные кадры живого потока, а не только самый свежий |
| `vid_stride` | `1` | Обрабатывать каждый N-й кадр видео или экрана |
| `show` | `False` | Показывать кадры с предсказаниями в окне |
| `output_path` | `None` | Путь вывода при `save=True` |
| `color_format` | `"auto"` | Подсказка о цветовом формате для массивов в памяти |
| `tiling` | `False` | Потайловый инференс для больших изображений |
| `overlap_ratio` | `0.2` | Доля перекрытия тайлов |
| `output_file_format` | `None` | `"jpg"`, `"png"` или `"webp"` |
| `cuda_graph` | `False` | `True` — захват при первом использовании для каждой формы входа, `"auto"` — ждать повторения формы |

Один источник-изображение возвращает один `Results`. Список, кортеж или каталог
возвращают список таких объектов, а `stream=True` возвращает генератор в любом
случае.

Источники живых потоков не ограничены по длине и требуют `stream=True`. `tiling`
и `augment` нельзя сочетать. Для задач `embed`, `point` и `edge` аугментация на
этапе инференса вызывает ошибку.

<code-tabs name="usage" />

При `batch > 1` семейства, у которых `SUPPORTS_BATCHED_PREDICT` истинно,
выполняют один прямой проход на пачку сложенных изображений; при `batch=1`
остаётся один проход на изображение.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Удобная обёртка над `predict`, которая собирает все строки эмбеддингов в один
тензор `(N_total, D)`. Модель должна быть создана с `task="embed"`, иначе метод
вызывает `NotImplementedError`.

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

Отдаёт по одному `Results` на кадр с заполненным `track_id`. `tracker`
принимает `"bytetrack"`, `"botsort"`, `"ocsort"` или `"deepocsort"` и
игнорируется, если задан `tracker_config`, потому что трекер выбирается по типу
конфига. `track_conf` передаётся как `track_high_thresh` для ByteTrack и
BoT-SORT и как `det_thresh` для OC-SORT и Deep OC-SORT. `output_path` по
умолчанию — `runs/track/<video_stem>.mp4`.

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

Возвращает словарь метрик, ключи которого зависят от задачи; для детекции это
`metrics/precision`, `metrics/recall`, `metrics/mAP50` и `metrics/mAP50-95`.
`imgsz` принимает целое число для квадратного размера или кортеж
`(height, width)`, по умолчанию берётся родной входной размер модели. `plots` —
псевдоним для `save_plots`. `allow_download_scripts` разрешает выполнять
встроенный Python-код, который YAML датасета может содержать в поле `download`.

`faster_coco_eval` принимается через `**kwargs` и по умолчанию равен `True`, с
откатом на pycocotools, если пакет не установлен. Использованный бэкенд
сообщается в `model.last_eval_backend`.

Валидация с аугментацией вызывает ошибку для задач `obb` и `pose`.

## train

`train` определяется в каждом семействе, поэтому набор аргументов различается.
Общими остаются два момента, потому что базовый класс оборачивает `train`
любого семейства:

- `cfg=` принимает путь к YAML, ключи которого подмешиваются в вызов. Явно
  переданные именованные аргументы важнее файла.
- `pretrained=False` для семейства из группы покрытия `g0` или `g1`
  переинициализирует модель с нуля перед обучением и не сочетается с
  `resume=True`.

Какие настройки аугментации семейство действительно учитывает — вопрос для
каждого семейства отдельно; см. [матрицу аугментаций](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Возвращает путь к записанному артефакту. `format` разрешается через реестр
экспортёров, где `engine` — псевдоним для `tensorrt`, а `litert` — псевдоним для
`tflite`. Аргументы, общие для всех экспортёров:

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `output_path` | `None` | Путь к выходному файлу; если не задан, генерируется внутри `weights/` |
| `imgsz` | `None` | Кортеж `(height, width)` или одно целое число; по умолчанию родной размер |
| `opset` | `None` | Версия ONNX opset |
| `simplify` | `True` | Выполнять упрощение графа ONNX |
| `dynamic` | `True` | Включить динамические оси |
| `half` | `False` | Точность FP16 |
| `int8` | `False` | Точность INT8 |
| `batch` | `1` | Размер батча, зашитый в артефакт |
| `device` | `None` | Устройство, на котором идёт трассировка |
| `data` | `None` | data.yaml для калибровки INT8 |
| `fraction` | `1.0` | Какую долю калибровочного датасета использовать |
| `allow_download_scripts` | `False` | Разрешить встроенный Python в загрузках из YAML датасета |
| `verbose` | `False` | Подробный лог экспортёра |

Запрещённые комбинации вызывают `NotImplementedError` на предварительной
проверке, до трассировки. Покрытие и его правила описаны на странице
[матрица экспорта](/docs/reference/export-matrix). Если подключены активные
LoRA-адаптеры, они сливаются с плотными весами, причём это слияние происходит
только после всех проверок, способных отклонить запрос.

## save

```python
model.save(path) -> str
```

Пишет чекпойнт LibreYOLO схемы v1.0: state dict плюс метаданные, описанные в
[схеме чекпойнта](/docs/reference/checkpoint-schema). Квантизованная модель
дополнительно несёт свой манифест `quant`, поэтому `LibreYOLO(path)`
восстанавливает квантизованную структуру и масштабы.

## quantize, quant_info и dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

Квантизует модель на месте и возвращает её. `recipe` — это одно из приведений
типов `fp16` и `bf16`, один из рецептов для Conv и Linear `int8` и `fp8` либо
один из рецептов только для Linear `w4a16`, `w4a8`, `nvfp4`, `mxfp4` и `int2`,
которые поддерживаются трансформерными семействами вроде RF-DETR. `int2`
требует QAT. `calib` принимает путь к data.yaml или имя встроенного датасета и
читает изображения только для прямого прохода; метки не читаются никогда. Передайте `calib=None`,
чтобы пропустить калибровку. `algorithm` принимает `"minmax"`, `"percentile"`
или `"auto"`.

`model.quant_info()` возвращает сводку по состоянию квантизации или `None` для
float-модели. `model.dequantize()` восстанавливает float-модули на месте,
сохраняя мастер-веса, обученные с учётом квантизации, — это мост от QAT к
`export(format="onnx", int8=True, data=...)`.

## info и слои

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` возвращает словарь, пригодный для JSON, и при истинном `verbose` пишет в
лог сводку в человекочитаемом виде. `get_available_layer_names` перечисляет
слои, которые может назвать конфиг дистилляции или извлечения признаков.

## CUDA-графы

Доступны в семействах, у которых атрибут класса `SUPPORTS_CUDA_GRAPH` истинен.
Воспроизведение графа бит в бит совпадает с обычным (eager) выполнением.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # контекстный менеджер
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Захваченный граф действителен только для той формы, на которой он был захвачен,
поэтому `batch` и `imgsz` должны совпадать с последующим вызовом `predict`.
`capture_graph` убирает затраты на захват из первого запроса. `mode` принимает
`True` или `"on"` — захват при первом использовании, `"auto"` — ждать
повторения формы, и `False` — ничего не делать. `capture_graph` вызывает
`NotImplementedError`, если семейство не включило поддержку, и
`CudaGraphUnavailable`, если захват не удался.

## Устройство и dtype

У объектов `Results` есть `.to()`, `.cpu()`, `.cuda()` и `.numpy()`; см.
[типы Results](/docs/reference/results-types). Саму модель переносят, передав
`device=` в `predict` или при создании.
