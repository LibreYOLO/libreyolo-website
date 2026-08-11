---
title: Работа с результатами
seo_title: Объект Results в LibreYOLO
description: >-
  Один объект Results на изображение, со слотом под каждый тип данных: рамки,
  маски, ключевые точки, probs, глубина, паноптика, OCR и другое. Отрисовка,
  сохранение и JSON.
lead: >-
  Каждое предсказание возвращает по одному объекту Results на изображение. В нём
  по одному именованному слоту на каждый тип данных, и все они пустые, кроме
  тех, что заполняет модель; те же слоты есть и у экспортированного артефакта.
keywords:
  - объект results yolo python
  - results.boxes xyxy
  - results в json yolo
  - сохранить аннотированное изображение yolo
  - маски сегментации python
  - ключевые точки results yolo
  - карта глубины results
  - results summary yolo
  - onnx те же результаты yolo
last_verified: 1.5.0
verification: >-
  Классы данных, слоты, семантика перемещения, summary(), to_json(), plot(),
  save() и cutout() прочитаны из libreyolo/utils/results.py. Поведение
  аннотирования и записи на диск — из InferenceRunner._save_annotated_image в
  libreyolo/models/base/inference.py и resolve_save_path в
  libreyolo/utils/general.py. Выбор по суффиксу — из LibreYOLO() в
  libreyolo/models/__init__.py.
snippets:
  basic:
    - label: Boxes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape)   # (height, width) исходного изображения
        print(result.path)         # путь к источнику, None для входа из памяти

        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Нормализованные координаты
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy[:1])    # пиксели, x1 y1 x2 y2

        print(result.boxes.xywh[:1])    # пиксели, центр x, центр y, w, h

        print(result.boxes.xyxyn[:1])   # та же рамка, поделённая на ширину и
        высоту

        print(result.boxes.xywhn[:1])
    - label: NumPy и устройства
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # Каждый из них возвращает новый Results; оригинал не меняется.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary и to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # То же содержимое строкой, с теми же именованными аргументами.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Аннотированные изображения
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True рисует данные и пишет их в runs/detect/predict*.
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: Установка extra-зависимости для экспорта
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Тот же Results из экспортированного артефакта
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # возвращает путь записанного файла

        # LibreYOLO() выбирает загрузчик по суффиксу файла.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## Один объект, один слот на каждый тип данных

Предсказание по одному изображению возвращает один `Results`. В нём восемнадцать
слотов под данные, и модель заполняет только те, которые даёт её задача. Все
остальные слоты — `None`, поэтому чтение `result.masks` у детектора даёт `None`,
а не ошибку.

| Слот | Класс | Форма | Что порождает |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` плюс оценки и классы | Детекция и любая задача, которая начинается с локализации |
| `masks` | `Masks` | `(N, H, W)` | Сегментация экземпляров |
| `keypoints` | `Keypoints` | `(N, K, 2)` или `(N, K, 3)` | Оценка позы |
| `probs` | `Probs` | `(C,)` | Классификация |
| `obb` | `OBB` | `(N, 7)` или `(N, 8)` | Повёрнутые рамки |
| `gaze` | `Gaze` | `(N, 2)`, pitch и yaw в радианах | Оценка направления взгляда |
| `points` | `Points` | `(N, 4)` как x, y, класс, уверенность | Локализация точек |
| `semantic_mask` | `SemanticMask` | `(H, W)` с id классов | Семантическая сегментация |
| `panoptic` | `PanopticSegmentation` | `(H, W)` с id сегментов плюс `segments_info` | Паноптическая сегментация |
| `depth_map` | `DepthMap` | `(H, W)` вещественных чисел | Оценка глубины |
| `normal_map` | `NormalMap` | `(H, W, 3)` единичных векторов | Нормали поверхности |
| `edges` | `EdgeMap` | `(H, W)` вещественных чисел в `[0, 1]` | Детекция границ |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | Восстановление и увеличение разрешения |
| `matte` | `Matte` | `(H, W)` вещественных чисел в `[0, 1]` | Альфа-маттинг и удаление фона |
| `ocr` | `OCRRegions` | `(N, 4, 2)` полигонов плюс распознанный текст | Детекция и распознавание текста |
| `embeddings` | `Embeddings` | `(N, D)` строк, нормированных по L2 | Задача `embed` |
| `identities` | `Identities` | N имён и оценок | Задача `embed` с галереей |
| `meshes` | `Meshes` | Параметры тела и опциональные вершины | Восстановление меша тела |

Рядом с ними лежат поля, которые есть у любого результата: `orig_shape` как
`(height, width)`, `path` (путь к источнику или `None` для входа из памяти),
`names` — соответствие id класса имени класса, `frame_idx` для кадров видео и
живого потока, `track_id` при трекинге и `restore_scale` — целочисленный
коэффициент увеличения для результата восстановления.

`result.normals` — псевдоним для `result.normal_map`.

`result.speed` есть у каждого результата, но заполняется только
[ансамблями](/docs/predict/ensembling): там его ключи — `member_0`,
`member_1` и `fusion`, в миллисекундах. Для одиночной модели он остаётся пустым
словарём.

## Boxes

<code-tabs name="basic" />

`Boxes` хранит координаты и оценки в отдельных массивах, а не в одном упакованном
тензоре.

| Атрибут | Содержимое |
|---|---|
| `xyxy` | `(N, 4)` в абсолютных пикселях, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` в абсолютных пикселях, центр x, центр y, ширина, высота |
| `xyxyn`, `xywhn` | То же самое, поделённое на ширину и высоту изображения |
| `conf` | `(N,)` уверенность |
| `cls` | `(N,)` id класса, вещественным числом |
| `id` | `(N,)` id трека или `None` |
| `is_track` | Задан ли `id` |
| `data` | Всё склеенное вместе: рамки, опциональный id, conf, cls |

`cls` — массив вещественных чисел, поэтому используйте его так:
`result.names[int(cls)]`.

`xyxyn` и `xywhn` требуют `orig_shape`, и `Results` подставляет его за вас.

## Плотные данные

Данные, покрывающие всё изображение, ведут себя иначе, чем данные по отдельным
экземплярам, и при взятии срезов это важно.

`SemanticMask` хранит `(H, W)` id классов на исходном холсте, где `255`
зарезервировано как значение «игнорировать» и никогда не считается классом.
`classes` перечисляет присутствующие id и исключает его; `class_mask(id)`
возвращает булев массив `(H, W)`.

`PanopticSegmentation` хранит `(H, W)` id сегментов, где `0` — id пустоты, и
список словарей `segments_info`, в каждом из которых есть как минимум `id` и
`category_id`. `segment_ids` перечисляет присутствующие id, `segment_mask(id)`
выбирает один.

`DepthMap` хранит `(H, W)` относительной обратной глубины: больше — значит ближе,
и значения не в метрах. Он даёт `min`, `max`, `mean` по конечным значениям и
`normalized()`, приводящий диапазон к `[0, 1]`.

`NormalMap` хранит `(H, W, 3)` единичных векторов в системе координат камеры
OpenCV, где `+x` — вправо, `+y` — вниз, а `+z` — вглубь сцены, поэтому
поверхность, обращённая к камере, — это `(0, 0, -1)`. `assert_normalized()`
проверяет, что каждый пиксель конечен и имеет единичную длину.

`EdgeMap` хранит `(H, W)` типа float32 в `[0, 1]`. Непрерывная карта сохраняется
как есть, без порога, поэтому отсечку вы выбираете в `binary(threshold=0.5)`.

`Matte` хранит `(H, W)` типа float32 в `[0, 1]`, где `1` — полностью передний
план. `array` возвращает его обрезанным по диапазону, в float32.

`RestoredImage` хранит `(H, W, 3)` uint8 RGB: `array` даёт сырой ndarray, а
`save(path)` записывает его на диск.

`Probs` хранит один вектор вероятностей для изображения. `top1` и `top5` — это
индексы классов, `top1conf` и `top5conf` — соответствующие оценки.

`Embeddings` хранит `(N, D)` строк, уже нормированных по L2, поэтому косинусная
близость — это скалярное произведение. `similarity(other)` возвращает `(N, M)`
для галереи или `(N,)` для одного вектора, а `verify(i, j, threshold=0.4)`
сравнивает две строки.

`OCRRegions` хранит `(N, 4, 2)` полигонов в порядке чтения; углы идут в порядке
левый верхний, правый верхний, правый нижний, левый нижний. Распознанный текст
лежит в `texts`, оценки распознавания — в `conf`, оценки детекции — в `det_conf`. Это
настоящие повёрнутые полигоны, поэтому они не заполняют `boxes`; если нужны
прямоугольники, `ocr.xyxy` даёт оболочки, выровненные по осям.

## Срезы и перемещение

`result[i]` возвращает новый `Results` с одним экземпляром. Данные по
экземплярам режутся; данные по всему изображению переносятся без изменений,
поэтому срез результата классификации не может урезать его вектор вероятностей
до одного класса, а срез результата с глубиной не может испортить раскладку
`(H, W)`.

`len(result)` считает экземпляры: рамки, точки, эмбеддинги, области OCR или меши.
Любые плотные данные по всему изображению считаются за `1`. Результат, в котором
нет ничего, — это `0`.

`to()`, `cpu()`, `cuda()` и `numpy()` возвращают новый `Results`, в котором
сконвертирован каждый заполненный слот. Оригинал они не меняют.

`update()` — единственный метод, который меняет объект на месте: он заменяет
слоты по имени и возвращает тот же объект.

## JSON

<code-tabs name="json" />

`summary()` возвращает список обычных словарей, а `to_json()` — этот же список,
пропущенный через `json.dumps`. Оба принимают одни и те же три аргумента:
`normalize=False` переводит координаты в `[0, 1]`, `decimals=5` задаёт
округление, а `embeddings=False` управляет тем, попадут ли в вывод векторы
эмбеддингов.

Структура строки зависит от данных. Строки детекции содержат `name`, `class`,
`confidence` и словарь `box`, а также подхватывают `segments`, когда есть маски,
`obb` и `corners` для повёрнутых рамок, углы `gaze` и в радианах, и в градусах,
`track_id` при трекинге и параметры `mesh`, когда есть меши.

Если рамок нет, строки определяются одним типом данных: OCR выдаёт по строке на
область с её `text`, точки — по строке на точку, паноптика — по строке на
сегмент с `pixel_count` и `pixel_fraction`, семантика — по строке на каждый
присутствующий класс, классификация — пять лучших классов. Глубина, нормали,
границы, восстановление и маттинг выдают по одной сводной строке, которая
описывает карту, а не её пиксели.

Два типа данных намеренно сокращены. Вектор эмбеддинга отдаётся только как
`embedding_dim`, потому что строка из 512 вещественных чисел — это около 2 КБ на
одно лицо; чтобы включить сами значения, передайте `embeddings=True`. Вершины
мешей не включаются вообще, потому что это десятки тысяч координат на человека.
Чтобы получить геометрию, читайте `result.meshes.vertices` или вызывайте
`result.meshes.save_obj(path)`.

## Отрисовка и сохранение

<code-tabs name="saving" />

Аннотирует и записывает именно `predict(save=True)`. Способ отрисовки выбирается
по тому, какой слот заполнен: семантический результат записывается как цветная
маска, результат с глубиной — как визуализация глубины, паноптический — со
своими сегментами, результат маттинга — как RGBA PNG с прозрачным фоном, а детектор — как
рамки с масками под ними. Путь записанного файла кладётся в результат как
`result.saved_path`.

`Results.plot()` — метод более узкий, чем кажется по названию. Он определён только для карт
нормалей и карт границ, а для всего остального бросает `NotImplementedError`.
Для прочих задач используйте `save=True`.

`Results.save(path)` так же узок: он записывает результат маттинга как вырезанный
объект в RGBA PNG с прозрачным фоном, а в остальных случаях бросает
`NotImplementedError`. `Results.cutout()` возвращает тот же RGBA-массив, ничего
не записывая. Обоим нужно исходное изображение — оно берётся из `result.path`
или передаётся через `image=`.

У двух типов данных есть свои методы записи: `result.restored.save(path)` для
восстановленного изображения и `result.meshes.save_obj(path, index=0)` для меша.

О том, куда попадают файлы и как ведут себя `output_path` и
`output_file_format`, см. [Источники для предсказания](/docs/predict/sources).

## Экспортированные артефакты возвращают тот же объект

<code-tabs name="exported" />

`LibreYOLO()` выбирает загрузчик по суффиксу файла, поэтому экспортированный
артефакт загружается тем же вызовом, что и `.pt`-чекпойнт, и возвращает тот же
`Results`. По суффиксу распознаются файлы `.onnx`, `.engine`, `.pte` и `.mnn`, а
также каталоги OpenVINO, Paddle и ncnn и URL модели в Triton. Код, который читает
`result.boxes.xyxy`, не меняется, когда модель заменяют на её экспортированную
сборку. Полный набор форматов — в разделе [Экспорт](/docs/export).

Если вместо этого браться за собственный API среды выполнения, предобработку,
постобработку и имена классов придётся брать на себя.
