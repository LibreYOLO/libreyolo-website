---
title: Источники предсказания
seo_title: Источники предсказания в LibreYOLO
description: >-
  Все источники, которые принимает predict: изображения, папки, URL, видеофайлы,
  веб-камеры, RTSP, YouTube, захват экрана, списки изображений и файлы .streams.
lead: >-
  Аргумент source классифицируется до того, как что-либо будет открыто, поэтому
  один вызов обрабатывает JPEG, папку, MP4, индекс веб-камеры, RTSP-URL, область
  экрана или список камер.
keywords:
  - инференс видео yolo python
  - rtsp
  - детекция объектов с веб-камеры python
  - предсказание по папке с изображениями
  - детекция объектов на экране
  - несколько rtsp потоков
  - файл streams
  - инференс youtube
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  Классификация источников прочитана из libreyolo/utils/source.py
  (classify_source, SourceKind, StreamSource, MultiStreamSource). Принимаемые
  типы изображений и расширения файлов в каталогах — из
  libreyolo/utils/image_loader.py. Расширения видео и пути сохранения — из
  libreyolo/utils/video.py. Синтаксис screen — из libreyolo/utils/screen.py.
  Формы возвращаемых значений и значения аргументов по умолчанию — из
  InferenceRunner.__call__ в libreyolo/models/base/inference.py.
snippets:
  images:
    - label: Одно изображение
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Одиночный источник-изображение возвращает один Results, а не список.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: Изображения в памяти
      language: python
      code: |
        import numpy as np
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        pil_image = Image.open(SAMPLE_IMAGE)
        array = np.asarray(pil_image)
        raw_bytes = open(SAMPLE_IMAGE, "rb").read()

        for source in (pil_image, array, raw_bytes):
            result = model(source)
            print(type(source).__name__, len(result.boxes))
    - label: Папка
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("sample_folder")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(3):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Папка возвращает список: по одному Results на изображение,
        отсортированный по пути.

        results = model(str(folder))

        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Видеофайл (нужен свой ролик)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Замените clip.mp4 на видеофайл на диске.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: 'Каждый третий кадр, с записью на диск'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Веб-камера (нужна подключённая камера)
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Индекс веб-камеры 0. Живые источники не заканчиваются, поэтому цикл
        нужно ограничить.

        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (нужен доступный URL камеры)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Файл .streams (нужны свои камеры)
      language: python
      code: >
        import itertools

        from pathlib import Path


        from libreyolo import LibreYOLO


        Path("cameras.streams").write_text(
            "# one source per line, blank lines and comments are skipped\n"
            "rtsp://192.168.1.64:554/Streaming/Channels/101\n"
            "rtsp://192.168.1.65:554/Streaming/Channels/101\n",
            encoding="utf-8",
        )


        model = LibreYOLO("LibreYOLO9s.pt")

        for result in itertools.islice(model("cameras.streams", stream=True),
        100):
            print(result.frame_idx, len(result.boxes))
    - label: Список камер
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: Один скриншот (нужны mss и графическая сессия)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Без stream=True захватывается один кадр.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: 'Область одного монитора, непрерывно'
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # "screen <monitor> <left> <top> <width> <height>"

        for result in itertools.islice(model("screen 1 100 200 512 256",
        stream=True), 50):
            print(len(result.boxes))
source_hash: c371965951dd0181
---

## Как классифицируется источник

`classify_source` проверяет значение до того, как что-либо будет открыто или
скачано, в таком порядке. Побеждает первое подошедшее правило.

| Источник | Как читается |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Захват экрана |
| Неотрицательный `int` или строка из цифр, для которой нет файла с таким именем | Веб-камера |
| URL вида `rtsp://`, `rtmp://`, `tcp://` или `udp://` | Сетевой поток |
| URL вида `http(s)://`, путь которого оканчивается на `.m3u8` | Сетевой поток |
| URL страницы YouTube | Сетевой поток |
| Список или кортеж, все элементы которого — живые источники или видео | Несколько живых потоков |
| Любой другой список или кортеж | Батч изображений |
| Путь, оканчивающийся на `.streams` | Несколько живых потоков |
| Путь с видеорасширением | Видеофайл |
| Существующий каталог | Папка с изображениями |
| Всё остальное | Одно изображение |

Список, в котором живые источники смешаны с изображениями, вызывает
`TypeError`. Отрицательный индекс веб-камеры вызывает `ValueError`.

Классификатор никогда не обращается к сети, поэтому URL с опечаткой
обнаружится при открытии захвата, а не при вызове `predict`.

## Изображения

<code-tabs name="images" />

Одиночный источник-изображение принимает семь типов.

| Тип | Как читается |
|---|---|
| `str` или `pathlib.Path` | Локальный файл, `http(s)://`, `s3://` или `gs://` |
| `PIL.Image.Image` | Преобразуется в RGB |
| `numpy.ndarray` | 2D в оттенках серого либо 3D HWC или CHW; из 4D-массива берётся первое изображение |
| `torch.Tensor` | CHW или NCHW, читается как RGB; из батчевого тензора берётся первое изображение |
| `bytes` | Закодированные данные изображения |
| `io.BytesIO` | Закодированные данные изображения |

Перед предобработкой всё преобразуется в RGB. NumPy-массивы — единственный
случай, где порядок каналов неоднозначен, поэтому им управляет `color_format`:
`"auto"` (значение по умолчанию) оставляет массив как есть, `"bgr"`
переворачивает каналы — именно это нужно кадру, прочитанному через OpenCV.

Массивы с плавающей точкой масштабируются по собственному диапазону: значения
не выше `1.0` умножаются на 255, более высокие обрезаются до диапазона
`[0, 255]`. У RGBA-массива отбрасывается альфа-канал.

Каждому удалённому пути нужен свой пакет, и по умолчанию не установлен ни один
из них: `requests` для `http(s)://`, `boto3` для `s3://` и `gcsfs` для
`gs://`.

## Папки

Каталог сканируется рекурсивно, результат сортируется, и изображением
становится каждый файл с одним из этих суффиксов: `.jpg`, `.jpeg`, `.png`,
`.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`. Всё остальное в папке пропускается.
Пустая папка возвращает пустой список, а не вызывает ошибку.

Папки и списки — два источника, которые принимают `batch`: в семействах с его
поддержкой он выполняет один общий прямой проход на каждую порцию. См.
[Производительность инференса](/docs/predict/performance).

## Видеофайлы

<code-tabs name="video" />

Путь считается видео, когда его суффикс — один из `.asf`, `.avi`, `.gif`,
`.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`, `.webm`.

`.gif` есть в обоих списках. Путь к `.gif`, переданный прямо в `predict`,
открывается как видео, потому что проверка на видео идёт первой; `.gif`,
лежащий внутри просканированной папки, загружается как неподвижное
изображение.

`vid_stride` обрабатывает каждый N-й кадр, по умолчанию `1`. Без `stream=True`
всё видео декодируется в список, и если после прореживания кадров больше 500,
выводится предупреждение с советом использовать `stream=True`.

У каждого `Results` из видео есть `frame_idx`.

## Веб-камеры, сетевые потоки и YouTube

<code-tabs name="live" />

Живые источники не ограничены по длине, поэтому им нужен `stream=True`. Без
него `predict` вызывает `ValueError`, вместо того чтобы пытаться собрать
бесконечный список.

Кадры читаются в фоновом потоке выполнения — по одному потоку на каждый
захват. По
умолчанию в очереди хранится только самый свежий кадр, поэтому модель, которая
медленнее камеры, пропускает кадры, а не отстаёт. `stream_buffer=True` хранит
каждый захваченный кадр: так они сохраняются, но ценой растущей задержки.

Индекс веб-камеры — это `int` или строка из цифр. В Windows захват сначала
открывается через бэкенд DirectShow, а если это не удаётся, используется бэкенд
по умолчанию.

URL страниц YouTube преобразуются в прямой URL медиапотока без скачивания
видео; для этого нужен `yt-dlp`:

```bash
pip install "libreyolo[stream]"
```

Метки потоков маскируются, прежде чем попасть в логи или в имена файлов. URL с
учётными данными выглядит как `user:***@host`, а строки запроса из меток прямых
потоков отбрасываются, потому что именно там живут подписанные URL и
bearer-токены. Идентификатор видео на YouTube сохраняется — это не учётные
данные.

## Несколько камер сразу

<code-tabs name="streams" />

В файле `.streams` — по одному источнику на строку. Пустые строки и строки,
начинающиеся с `#`, игнорируются. Каждая оставшаяся строка должна сама быть
индексом веб-камеры, сетевым потоком, URL YouTube или путём к видеофайлу; всё
остальное вызывает `ValueError` с указанием номера строки. Пустой файл приводит
к ошибке, а не к запуску без камер.

Список или кортеж живых источников делает то же самое без файла.

У каждого захвата свой поток выполнения, а кадры со всех них мультиплексируются
в один генератор. На каждом проходе опрашивается каждый активный поток и
отдаётся всё, что готово, поэтому медленная камера не задерживает быструю, а
кадры разных камер чередуются. Закончившийся поток выпадает из очереди опроса,
остальные продолжают работать.

## Захват экрана

<code-tabs name="screen" />

Источник экрана — это слово `screen`, за которым идут ноль, одно, четыре или
пять целых чисел. Любое другое количество вызывает `ValueError`.

| Форма | Что захватывает |
|---|---|
| `"screen"` | Все мониторы, объединённые |
| `"screen 1"` | Монитор 1 |
| `"screen 100 200 512 256"` | Прямоугольник на объединённом рабочем столе |
| `"screen 1 100 200 512 256"` | Прямоугольник на мониторе 1 |

Координаты прямоугольника — `left top width height`, относительно левого
верхнего угла выбранного монитора. Источник экрана сообщает свою частоту кадров
как 30, делённое на `vid_stride`, — с этой же частотой пишется сохраняемое
видео. Для захвата нужен пакет `mss`:

```bash
pip install mss
```

Без `stream=True` источник экрана захватывает один кадр и возвращает один
`Results` — для скриншота это то же самое, что предсказание по файлу
изображения. С
`stream=True` он захватывает, пока цикл не будет прерван.

## Что возвращает predict

Форма возвращаемого значения зависит от источника и от `stream`.

| Источник | `stream=False` | `stream=True` |
|---|---|---|
| Одно изображение | Один `Results` | Генератор из одного `Results` |
| Список изображений | Список `Results` | Генератор |
| Папка | Список `Results` | Генератор |
| Видеофайл | Список `Results` | Генератор |
| Экран | Один `Results` | Генератор, неограниченный |
| Веб-камера, сетевой поток, `.streams` | `ValueError` | Генератор, неограниченный |

Для одного изображения возвращается сам объект `Results`. Индексация выбирает
детекцию, а не изображение, поэтому `result[0]` в предсказании по одному
изображению — это первая рамка, а не первая картинка. О том, что хранят эти
объекты, см. [Работа с результатами](/docs/predict/results).

## Куда пишет save

`save=True` записывает аннотированный результат в каталог запуска, а не
возвращает его.

Изображения попадают в автоинкрементные `runs/detect/predict`,
`runs/detect/predict2` и так далее, сохраняя имя исходного файла. Все
изображения одного процесса оказываются в одном каталоге, поэтому две входные
папки с одинаковым именем файла перезаписывают друг друга. У изображений в
памяти нет имени файла, которое можно было бы переиспользовать, поэтому они
нумеруются `image0`, `image1` и так далее.

Видео и живые источники записываются в один `.mp4`, названный по источнику.

`output_path` переопределяет каталог. Путь с суффиксом считается файлом, путь
без суффикса — каталогом. `output_file_format` задаёт формат кодирования
неподвижных изображений и принимает `jpg`, `png` или `webp`.

После сохранения записанный путь также добавляется к результату как
`result.saved_path`.
