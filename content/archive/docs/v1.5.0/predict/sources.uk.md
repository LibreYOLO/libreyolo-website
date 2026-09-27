---
title: Джерела передбачення
seo_title: Джерела передбачення в LibreYOLO
description: >-
  Усі джерела, які приймає predict: зображення, папки, URL, відеофайли,
  вебкамери, RTSP, YouTube, запис екрана, списки зображень і файли .streams.
lead: >-
  Аргумент source класифікується до відкриття будь-яких даних, тому один виклик
  обробляє JPEG, папку, MP4, індекс вебкамери, URL RTSP, область екрана або
  список камер.
keywords:
  - yolo відео інференс python
  - rtsp
  - детекція об'єктів з вебкамери python
  - передбачення для папки зображень
  - виявлення об'єктів на екрані
  - кілька потоків rtsp
  - файл streams
  - youtube інференс
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  Класифікацію джерел перевірено за libreyolo/utils/source.py (classify_source,
  SourceKind, StreamSource, MultiStreamSource). Прийняті типи зображень і
  розширення каталогів взято з libreyolo/utils/image_loader.py. Розширення відео
  та шляхи збереження взято з libreyolo/utils/video.py. Синтаксис екрана взято з
  libreyolo/utils/screen.py. Форми повернених даних і типові значення аргументів
  перевірено за InferenceRunner.__call__ у libreyolo/models/base/inference.py.
snippets:
  images:
    - label: Одне зображення
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Джерело з одним зображенням повертає один Results, а не список.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: Зображення в пам'яті
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


        # Папка повертає список з одним Results на зображення, відсортований за
        шляхом.

        results = model(str(folder))

        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Відеофайл (надайте власний кліп)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Замініть clip.mp4 на відеофайл із диска.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: Кожен третій кадр із записом на диск
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Вебкамера (потрібна під'єднана камера)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Індекс вебкамери 0. Джерела наживо нескінченні, тому обмежте цикл.
        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (потрібен доступний URL камери)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Файл .streams (надайте власні камери)
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
    - label: Один знімок екрана (потрібні mss і сеанс робочого стола)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Без stream=True отримується один кадр.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: Неперервне захоплення області одного монітора
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

## Класифікація джерела

`classify_source` перевіряє значення до відкриття чи завантаження будь-яких
даних у наведеному порядку. Застосовується перше правило, що збігається.

| Джерело | Інтерпретація |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Захоплення екрана |
| Невід'ємне `int` або рядок із цифр, якщо файла з такою назвою немає | Вебкамера |
| URL `rtsp://`, `rtmp://`, `tcp://` або `udp://` | Мережевий потік |
| URL `http(s)://`, шлях якого закінчується на `.m3u8` | Мережевий потік |
| URL сторінки YouTube | Мережевий потік |
| Список або кортеж, усі елементи якого є джерелами наживо чи відео | Кілька потоків наживо |
| Будь-який інший список або кортеж | Батч зображень |
| Шлях, що закінчується на `.streams` | Кілька потоків наживо |
| Шлях із розширенням відео | Відеофайл |
| Наявний каталог | Папка зображень |
| Усе інше | Одне зображення |

Список, у якому джерела наживо змішано із зображеннями, спричиняє `TypeError`.
Від'ємний індекс вебкамери спричиняє `ValueError`.

Класифікатор ніколи не звертається до мережі, тому помилка в URL проявляється
під час відкриття захоплення, а не під час виклику `predict`.

## Зображення

<code-tabs name="images" />

Джерело з одним зображенням приймає сім типів.

| Тип | Інтерпретація |
|---|---|
| `str` або `pathlib.Path` | Локальний файл, `http(s)://`, `s3://` або `gs://` |
| `PIL.Image.Image` | Перетворюється на RGB |
| `numpy.ndarray` | 2D у відтінках сірого або 3D HWC чи CHW; для 4D масиву використовується перше зображення |
| `torch.Tensor` | CHW або NCHW, зчитується як RGB; для пакетного тензора використовується перше зображення |
| `bytes` | Закодовані дані зображення |
| `io.BytesIO` | Закодовані дані зображення |

До попереднього оброблення все перетворюється на RGB. Порядок каналів
неоднозначний лише для масивів NumPy, тому ним керує `color_format`: `"auto"`
(типове значення) залишає масив без змін, а `"bgr"` змінює порядок каналів на
зворотний, що потрібно для кадру, зчитаного через OpenCV.

Масиви з рухомою комою масштабуються за власним діапазоном: значення не вище
`1.0` множаться на 255, а вищі значення обмежуються діапазоном `[0, 255]`.
У масиві RGBA альфа-канал відкидається.

Для віддалених шляхів потрібен окремий пакет, жоден із яких типово не
встановлено: `requests` для `http(s)://`, `boto3` для `s3://` і `gcsfs` для `gs://`.

## Папки

Каталог сканується рекурсивно й сортується, а кожен файл з одним із наведених
суфіксів стає зображенням: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`,
`.tiff`, `.tif`. Усе інше в папці пропускається. Порожня папка повертає порожній
список, а не спричиняє помилку.

Папки та списки є двома джерелами, які приймають `batch`. Для сімейств із
підтримкою ця функція виконує один прямий прохід складеного тензора на групу.
Дивіться [Продуктивність інференсу](/docs/predict/performance).

## Відеофайли

<code-tabs name="video" />

Шлях вважається відео, якщо його суфікс є одним із таких: `.asf`, `.avi`, `.gif`,
`.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`, `.webm`.

`.gif` є в обох списках. Шлях `.gif`, переданий безпосередньо до `predict`,
відкривається як відео, оскільки перевірка відео виконується першою; файл `.gif`
усередині сканованої папки завантажується як нерухоме зображення.

`vid_stride` обробляє кожен N-й кадр і типово дорівнює `1`. Без `stream=True`
усе відео декодується в список, а понад 500 кадрів після застосування кроку
спричиняють попередження з рекомендацією `stream=True`.

Кожен `Results` із відео містить `frame_idx`.

## Вебкамери, мережеві потоки та YouTube

<code-tabs name="live" />

Джерела наживо необмежені, тому потребують `stream=True`. Без нього `predict`
спричиняє `ValueError`, а не намагається зібрати нескінченний список.

Кадри зчитуються у фоновому потоці, по одному на кожне захоплення. Типово черга
зберігає лише найновіший кадр, тому модель, повільніша за камеру, пропускає кадри,
а не відстає. `stream_buffer=True` зберігає кожен захоплений кадр, що не втрачає
їх ціною дедалі більшої затримки.

Індекс вебкамери задається як `int` або рядок із цифр. У Windows захоплення
спочатку відкривається через бекенд DirectShow, а в разі невдачі переходить до
типового бекенда.

URL сторінки YouTube перетворюється на прямий URL медіаданих без завантаження
відео, для чого потрібен `yt-dlp`:

```bash
pip install "libreyolo[stream]"
```

Мітки потоків редагуються перед записом у журнал або використанням як назв
файлів. URL з обліковими даними відображається як `user:***@host`, а рядки
запиту вилучаються з міток прямих потоків, оскільки там розташовуються підписані
URL і токени пред'явника. Ідентифікатор відео YouTube зберігається, бо він не є
обліковими даними.

## Кілька камер одночасно

<code-tabs name="streams" />

Файл `.streams` містить одне джерело на рядок. Порожні рядки й рядки, що
починаються з `#`, ігноруються. Кожен інший рядок має містити індекс вебкамери,
мережевий потік, URL YouTube або шлях до відеофайла; усе інше спричиняє
`ValueError` із номером рядка. Порожній файл спричиняє помилку, а не запускає
оброблення без камер.

Список або кортеж джерел наживо дає той самий результат без файла.

Кожне захоплення отримує власний потік, а кадри з усіх них мультиплексуються в
один генератор. Кожен прохід опитує всі активні потоки й повертає готові дані,
тому повільна камера не затримує швидку, а кадри з різних камер чергуються.
Потік, що завершився, вилучається з ротації, а решта продовжує працювати.

## Захоплення екрана

<code-tabs name="screen" />

Джерело екрана задається словом `screen`, після якого йде нуль, одне, чотири
або п'ять цілих чисел. Будь-яка інша кількість спричиняє `ValueError`.

| Форма | Захоплює |
|---|---|
| `"screen"` | Усі монітори разом |
| `"screen 1"` | Монітор 1 |
| `"screen 100 200 512 256"` | Прямокутник на об'єднаному робочому столі |
| `"screen 1 100 200 512 256"` | Прямокутник на моніторі 1 |

Координати прямокутника задано як `left top width height` відносно верхнього
лівого кута вибраного монітора. Джерело екрана повідомляє частоту кадрів як 30,
поділене на `vid_stride`; з такою частотою записується збережене відео. Для
захоплення потрібен пакет `mss`:

```bash
pip install mss
```

Без `stream=True` джерело екрана захоплює один кадр і повертає один `Results`,
що відповідає передбаченню для знімка екрана як файла зображення. Зі
`stream=True` захоплення триває до переривання циклу.

## Що повертає predict

Форма поверненого значення залежить від джерела та `stream`.

| Джерело | `stream=False` | `stream=True` |
|---|---|---|
| Одне зображення | Один `Results` | Генератор одного `Results` |
| Список зображень | Список `Results` | Генератор |
| Папка | Список `Results` | Генератор |
| Відеофайл | Список `Results` | Генератор |
| Екран | Один `Results` | Необмежений генератор |
| Вебкамера, мережевий потік, `.streams` | `ValueError` | Необмежений генератор |

Одне зображення повертає сам об'єкт `Results`. Індексування вибирає в ньому
виявлення, а не зображення, тому `result[0]` для передбачення одного зображення
є першою рамкою, а не першою картинкою. Вміст цих об'єктів описано в розділі
[Робота з результатами](/docs/predict/results).

## Куди записує save

`save=True` записує анотований вивід у каталог запуску, а не повертає його.

Зображення потрапляють до автоматично нумерованих каталогів
`runs/detect/predict`, `runs/detect/predict2` тощо зі збереженням назви файла
джерела. Усі зображення одного процесу потрапляють до одного каталогу, тому дві
вхідні папки з однаковою назвою файла перезаписують одна одну. Зображення з
пам'яті не мають назви файла для повторного використання й нумеруються як
`image0`, `image1` тощо.

Відео та джерела наживо записуються як один файл `.mp4`, названий за джерелом.

`output_path` перевизначає каталог. Шлях із суфіксом розглядається як файл, а
шлях без нього як каталог. `output_file_format` вибирає кодування нерухомого
зображення та приймає `jpg`, `png` або `webp`.

Після збереження записаний шлях також додається до результату як `result.saved_path`.
