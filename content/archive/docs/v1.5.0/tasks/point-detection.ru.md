---
title: Детекция точек
seo_title: Детекция точек и подсчёт объектов в LibreYOLO
description: >-
  Поиск объектов как отдельных точек вместо рамок в LibreYOLO. Предсказание
  центроидов, подсчёт объектов, обучение FOMO и чтение точечных метрик.
lead: >-
  Детекция точек возвращает по одной координате x, y на объект вместо
  ограничивающей рамки. LibreYOLO предоставляет её как задачу point, и в
  предсказании на каждый объект приходится одна строка с x, y, классом и
  уверенностью.
keywords:
  - детекция точек python
  - подсчёт объектов на изображении
  - центроиды объектов
  - FOMO point detection
  - посчитать объекты на фото python
  - локализация объектов точками
last_verified: 1.5.0
snippets:
  predict:
    - label: Предсказание точек и их подсчёт
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Веса LibreFOMO автоматически не скачиваются. Сначала возьмите
        # чекпойнт с https://huggingface.co/LibreYOLO и загрузите его по
        # локальному пути.
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        points = result.points
        print(len(points))     # число объектов
        print(points.xy)       # центры (N, 2) в пикселях исходного изображения
        print(points.cls, points.conf)
    - label: Нормализованные координаты и количество по классам
      language: python
      code: |
        from collections import Counter

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE)

        points = result.points.numpy()
        print(points.xyn)                          # те же центры в [0, 1]
        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Обучение FOMO на YOLO-датасете
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Предсказание обученным чекпойнтом
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        results = model.train(data="my-dataset.yaml", epochs=40)

        # train() загружает лучший чекпойнт обратно в тот же объект, поэтому
        # после возврата из вызова модель предсказывает обученными весами.
        print(results["best_checkpoint"])
        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Валидация и чтение ключей метрик
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")

        metrics = model.val(data="my-dataset.yaml")


        print(metrics["metrics/precision"], metrics["metrics/recall"])

        print(metrics["metrics/f1"])

        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness

        print(metrics["metrics/MLE"])               # средняя ошибка локализации

        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # ошибка
        подсчёта
    - label: Изменение порогов расстояния
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # Границы перебора входят в текст ключа, поэтому свой набор порогов

        # переименовывает ключи mAP, которые он создаёт.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Экспорт
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Запуск экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому
        # экспортированный артефакт загружается как обычный чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("./LibreFOMOs-point.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Определение

Задача `point` находит каждый объект одной координатой x, y и классом — без
ширины, высоты и маски. Поскольку предсказание — это плоский список объектов,
число строк равно числу объектов, и именно поэтому это задача подсчёта.

Предсказание заполняет `result.points` — объект `Points`, который оборачивает
массив `(N, 4)` со строками `x, y, class, confidence` в пикселях исходного
изображения. `.xy` возвращает координаты, `.xyn` — те же координаты, делённые
на размер изображения, `.cls` — индексы классов, а `.conf` — оценки
уверенности; `len()` возвращает число точек. `result.boxes` остаётся пустым,
поэтому `iou` и `max_det` применять не к чему.

## Модели

Задачу `point` решают три семейства, и они не взаимозаменяемы.

[FOMO](/docs/models/fomo) — вариант с фиксированным словарём: сеточный
классификатор, который размечает каждую ячейку сетки низкого разрешения как фон
или как центр объекта. Это единственное точечное семейство, которое LibreYOLO
умеет обучать, и единственное, которое экспортируется.

[LocateAnything](/docs/models/locate-anything) принимает текст вместо индекса
класса, поэтому словарь — это любая фраза, которую вы напишете. Модели нужен
extra `vlm`, она создаётся как `LibreLocateAnything`, а не через фабрику
`LibreYOLO()`, и её веса ограничены некоммерческим использованием. Точные
условия и две дополнительные лицензии, которые объединяет чекпойнт, — на её
странице.

[SenseNova-Vision](/docs/models/sensenova-vision) решает `point` тем же
чекпойнтом генерации по промпту, которым решает ещё шесть задач; он загружается
через `LibreVLM("sensenova-vision", task="point")`. Ему нужен extra
`sensenova`, и каждое предсказание — это проход генерации по модели на 7B
параметров, так что задержка на изображение будет заметно выше, чем у
специализированного детектора. Его веса некоммерческие; лицензия — на его
странице.

## Предсказание

Веса LibreFOMO — единственное исключение из автоматической загрузки на этом
сайте. `LibreYOLO("LibreFOMOs-point.pt")` ищет этот файл на диске и вместо
скачивания выбрасывает `ValueError` с его именем. Сначала скачайте чекпойнт из
[организации LibreYOLO](https://huggingface.co/LibreYOLO) на Hugging Face и
загрузите его по локальному пути — или обучите свой.

<code-tabs name="predict" />

Имя файла должно нести суффикс задачи `-point`, иначе загрузчик его не
распознает. `predict(..., nms_radius=1)` задаёт, на сколько ячеек сетки должны
отстоять друг от друга две детекции FOMO, чтобы уцелели обе. Про источники,
стриминг и работу с результатами — в разделе [предсказание](/docs/predict).

## Формат датасета

У `point` нет собственного формата разметки. Точечные семейства читают
стандартную раскладку YOLO для детекции и выводят один центр из каждой строки с
рамкой, так что `cx cy` — это точка, а `w h` только решают, валидна ли строка.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

В каждом файле разметки — по одной строке на объект, с нормализованными
координатами:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

Отсутствующий или пустой файл разметки означает, что объектов нет. Полный
контракт — в разделе [форматы датасетов](/docs/reference/dataset-formats).

## Обучение

FOMO — единственное точечное семейство, для которого реализовано обучение.
`train()` у LocateAnything и у SenseNova-Vision выбрасывает
`NotImplementedError`; дообучайте их в исходных проектах и загружайте результат.

<code-tabs name="train" />

`imgsz` для FOMO выбирается не свободно: по умолчанию берётся родное разрешение
загруженного чекпойнта, а другое значение приводит к `ValueError` с указанием
ожидаемого размера. Про датасеты, логгеры и обучение на нескольких GPU — в
разделе [обучение](/docs/train), а значения по умолчанию для этого семейства —
на [странице FOMO](/docs/models/fomo).

## Валидация

`val()` сопоставляет предсказанные точки с точками эталонной разметки (ground
truth) один к одному венгерским алгоритмом, перебирая пороги расстояния. Порог
— это евклидово расстояние в нормализованных координатах изображения, а перебор
по умолчанию — десять значений от 0.01 до 0.10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` и `metrics/f1` усредняются по классам
(макро) на самом строгом пороге перебора — по умолчанию 0.01.
`metrics/mAP@0.01` — это AP на том же пороге, а `metrics/mAP@[0.01:0.10]` —
среднее по всему перебору. Это же среднее служит и `fitness` — числом, по
которому выбирается лучший чекпойнт. Оба ключа mAP строятся из
используемых порогов, поэтому передача `dist_thresholds=` их переименовывает.

`metrics/MLE` — среднее расстояние между сопоставленными парами на самом
строгом пороге, в тех же нормализованных единицах. `metrics/MAE` и
`metrics/RMSE` — метрики подсчёта, а не локализации: они измеряют разницу между
числом предсказанных точек и числом точек эталонной разметки на каждом
изображении.

Поверх этого FOMO добавляет вторую группу — на уровне сетки. Она перебирает
уверенность и `nms_radius` и публикует комбинацию с лучшим F1 как
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`,
`metrics/grid_mean_distance`, `metrics/grid_TP`, `metrics/grid_FP` и
`metrics/grid_FN`, а давшие её настройки — под `decode/threshold` и
`decode/nms_radius`.

## Экспорт

FOMO экспортируется через общий путь экспорта, а экспортированный артефакт
загружается обратно через `LibreYOLO()` по суффиксу файла, так что файл `.onnx`
или `.engine` ведёт себя как чекпойнт и возвращает тот же `Results`.

<code-tabs name="export" />

Поддержка по форматам — на [странице FOMO](/docs/models/fomo) и в [полной
матрице экспорта](/docs/reference/export-matrix). LocateAnything и
SenseNova-Vision не экспортируются: `export()` выбрасывает исключение в обоих
случаях, потому что у генеративной модели нет трассируемого графа детекции.
