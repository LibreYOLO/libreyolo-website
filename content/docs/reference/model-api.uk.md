---
title: API моделі
seo_title: "Методи й сигнатури об'єкта моделі LibreYOLO"
description: "Кожен метод завантаженої моделі LibreYOLO: predict, embed, track, val, train, export, save, quantize, info та засоби керування графами CUDA з фактичними типовими значеннями."
lead: "Завантажена модель LibreYOLO є екземпляром BaseModel. На цій сторінці перелічено методи цього екземпляра із сигнатурами й типовими значеннями з libreyolo/models/base/model.py."
keywords:
  - "методи моделі libreyolo"
  - "аргументи libreyolo predict"
  - "аргументи libreyolo val"
  - "аргументи libreyolo export"
  - model.track
  - model.quantize
  - capture_graph
last_verified: "1.5.0"
verification: "Сигнатури й типові значення взято з libreyolo/models/base/model.py та libreyolo/models/base/inference.py у v1.5.0. Класи сімейств можуть звужувати чи розширювати їх; train() визначається окремо для кожного сімейства, тому тут документовано лише його спільну обгортку cfg=."
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
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True повертає генератор з одним Results на кадр або зображення.
        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
---

## Створення

Фабрика повертає екземпляр класу сімейства. Безпосереднє створення цього класу
приймає ті самі аргументи, крім обов'язкового `size`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` вибирає доступну CUDA, потім MPS, а потім CPU. Ціле число або
рядок із цифр зчитується як порядковий номер CUDA, тому `device=0` і
`device="0"` обидва означають `cuda:0`. `task` перевіряється за
`SUPPORTED_TASKS` сімейства. Передавання `model_path=None` створює архітектуру
й залишає її в режимі навчання; передавання `dict` безпосередньо завантажує цей
словник стану.

## predict і \_\_call\_\_

`predict` є псевдонімом `__call__`.

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

| Аргумент | Типове значення | Значення |
|---|---|---|
| `source` | `None` | Зображення, список або кортеж зображень у пам'яті, каталог, відеофайл або джерело екрана на кшталт `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Поріг упевненості |
| `iou` | `0.45` | Поріг IoU для NMS |
| `imgsz` | `None` | Перевизначення розміру вхідних даних; `None` використовує рідний розмір моделі |
| `device` | `None` | Перевизначення пристрою для цього виклику |
| `classes` | `None` | Залишити лише ці ідентифікатори класів |
| `max_det` | `300` | Максимальна кількість виявлень на зображення |
| `augment` | `False` | Аугментація під час тестування |
| `save` | `False` | Запис анотованого зображення чи відео |
| `batch` | `1` | Кількість зображень на прямий прохід для джерел у вигляді каталогу чи списку |
| `stream` | `False` | Повернення генератора замість створеного списку |
| `stream_buffer` | `False` | Зберігати кожен захоплений кадр наживо, а не лише найновіший |
| `vid_stride` | `1` | Обробляти кожен N-й кадр відео чи екрана |
| `show` | `False` | Показувати анотовані кадри у вікні |
| `output_path` | `None` | Шлях виводу за `save=True` |
| `color_format` | `"auto"` | Підказка формату кольору для масивів у пам'яті |
| `tiling` | `False` | Тайловий інференс для великих зображень |
| `overlap_ratio` | `0.2` | Коефіцієнт перекриття тайлів |
| `output_file_format` | `None` | `"jpg"`, `"png"` або `"webp"` |
| `cuda_graph` | `False` | `True` захоплює граф під час першого використання кожної форми, `"auto"` чекає повторення форми |

Джерело з одним зображенням повертає один `Results`. Список, кортеж або каталог
повертає список таких об'єктів, а `stream=True` у кожному випадку повертає генератор.

Джерела потоків наживо необмежені й потребують `stream=True`. `tiling` і
`augment` не можна поєднувати. Аугментація під час тестування спричиняє помилку
для завдань `embed`, `point` та `edge`.

<code-tabs name="usage" />

За `batch > 1` сімейства з істинним `SUPPORTS_BATCHED_PREDICT` виконують один
прямий прохід складеного тензора на групу; `batch=1` зберігає один прямий
прохід на зображення.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Зручна обгортка над `predict`, яка складає всі рядки ембедінгів в один тензор
`(N_total, D)`. Модель має бути створена з `task="embed"`, інакше виникає
`NotImplementedError`.

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

Повертає по одному `Results` на кадр з установленим `track_id`. Значення
`tracker` є одним із `"bytetrack"`, `"botsort"`, `"ocsort"` або
`"deepocsort"` та ігнорується за наявності `tracker_config`, оскільки тип
конфігурації вибирає трекер. `track_conf` зіставляється з `track_high_thresh`
для ByteTrack і BoT-SORT та з `det_thresh` для OC-SORT і Deep OC-SORT. Типове
значення `output_path` дорівнює `runs/track/<video_stem>.mp4`.

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

Повертає словник метрик, ключі якого залежать від завдання; виявлення повертає
`metrics/precision`, `metrics/recall`, `metrics/mAP50` і `metrics/mAP50-95`.
`imgsz` приймає квадратне ціле число або кортеж `(height, width)` і типово
дорівнює рідному розміру вхідних даних моделі. `plots` є псевдонімом
`save_plots`. `allow_download_scripts` контролює вбудований Python, який YAML
датасету може містити в полі `download`.

`faster_coco_eval` приймається через `**kwargs` і типово дорівнює `True`, а за
відсутності пакета використовується резервний pycocotools. Використаний бекенд
повідомляється в `model.last_eval_backend`.

Аугментована валідація спричиняє помилку для завдань `obb` і `pose`.

## train

`train` визначається окремо для кожного сімейства, тому його аргументи
відрізняються. Спільними є дві поведінки, оскільки базовий клас обгортає
`train` кожного сімейства:

- `cfg=` приймає шлях YAML, ключі якого об'єднуються з викликом. Явні іменовані аргументи мають перевагу над файлом.
- `pretrained=False` для сімейства в групі покриття `g0` або `g1` повторно ініціалізує модель з нуля перед навчанням і не може поєднуватися з `resume=True`.

Параметри аугментації, які справді враховує сімейство, залежать від нього;
дивіться [матрицю аугментацій](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Повертає шлях до записаного артефакту. `format` визначається через реєстр
експортерів, де `engine` є псевдонімом `tensorrt`, а `litert` є псевдонімом
`tflite`. Аргументи, спільні для кожного експортера:

| Аргумент | Типове значення | Значення |
|---|---|---|
| `output_path` | `None` | Шлях вихідного файла; за відсутності генерується в `weights/` |
| `imgsz` | `None` | Кортеж `(height, width)` або одне ціле число; типово рідний розмір |
| `opset` | `None` | Версія opset ONNX |
| `simplify` | `True` | Спрощення графа ONNX |
| `dynamic` | `True` | Увімкнення динамічних осей |
| `half` | `False` | Точність FP16 |
| `int8` | `False` | Точність INT8 |
| `batch` | `1` | Розмір батча, убудований в артефакт |
| `device` | `None` | Пристрій трасування |
| `data` | `None` | data.yaml для калібрування INT8 |
| `fraction` | `1.0` | Частка калібрувального датасету для використання |
| `allow_download_scripts` | `False` | Дозвіл убудованого Python у завантаженнях YAML датасету |
| `verbose` | `False` | Докладне журналювання експортера |

Заблоковані поєднання спричиняють `NotImplementedError` під час попередньої
перевірки до трасування. Покриття та його правила наведено на сторінці
[матриці експорту](/docs/reference/export-matrix). Якщо наявні активні адаптери
LoRA, вони зливаються в щільні ваги, і це відбувається лише після всіх перевірок
відхилення запиту.

## save

```python
model.save(path) -> str
```

Записує контрольну точку LibreYOLO за схемою v1.0: словник стану разом із
метаданими, описаними в [схемі контрольної точки](/docs/reference/checkpoint-schema).
Квантована модель додатково містить маніфест `quant`, тому `LibreYOLO(path)`
відновлює квантовану структуру й масштаби.

## quantize, quant_info і dequantize

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

Квантує на місці й повертає модель. `recipe` є одним із перетворень `fp16` і
`bf16`, рецептів Conv та Linear `int8` і `fp8` або рецептів лише для Linear
`w4a16`, `w4a8`, `nvfp4`, `mxfp4` та `int2`, які підтримують трансформерні
сімейства на кшталт RF-DETR. Для `int2` потрібне QAT. `calib` приймає шлях
data.yaml або назву вбудованого датасету й зчитує зображення лише в прямому
напрямку; мітки ніколи не зчитуються. Передайте `calib=None`, щоб пропустити
калібрування. `algorithm` має значення `"minmax"`, `"percentile"` або `"auto"`.

`model.quant_info()` повертає підсумок стану квантування або `None` для моделі
з рухомою комою. `model.dequantize()` відновлює модулі з рухомою комою на місці,
зберігаючи основні ваги, навчені з квантуванням. Це міст від QAT до
`export(format="onnx", int8=True, data=...)`.

## info та layers

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` повертає словник, придатний для JSON, і записує зрозумілий користувачу
підсумок, коли `verbose` істинний. `get_available_layer_names` перелічує шари,
які може називати конфігурація дистиляції чи видобування ознак.

## Графи CUDA

Доступні для сімейств, у яких атрибут класу `SUPPORTS_CUDA_GRAPH` істинний.
Повторне виконання побітово ідентичне негайному виконанню.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # context manager
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Захоплений граф дійсний лише для точної форми захоплення, тому `batch` і
`imgsz` мають збігатися з подальшим викликом `predict`. `capture_graph`
переносить витрати захоплення за межі першого запиту. `mode` приймає `True` або
`"on"` для захоплення під час першого використання, `"auto"` для очікування
повторення форми й `False` для відсутності дії. `capture_graph` спричиняє
`NotImplementedError`, коли сімейство не заявило підтримку, і
`CudaGraphUnavailable`, коли захоплення не вдається.

## Пристрій і dtype

Об'єкти `Results` мають `.to()`, `.cpu()`, `.cuda()` і `.numpy()`; дивіться
[Типи Results](/docs/reference/results-types). Сама модель переміщується через
передавання `device=` до `predict` або під час створення.
