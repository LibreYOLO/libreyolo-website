---
title: Оценка позы
seo_title: Оценка позы в LibreYOLO
description: >-
  Предсказание ключевых точек для каждого экземпляра в LibreYOLO: семейства,
  которые решают эту задачу, формат разметки и вызовы для предсказания,
  обучения, валидации и экспорта.
lead: >-
  Оценка позы находит каждый экземпляр и возвращает для него упорядоченный набор
  именованных ключевых точек, поэтому в выводе есть внутренняя структура
  объекта, а не только его границы. Ключ задачи — pose.
keywords:
  - оценка позы python
  - детекция ключевых точек
  - скелет человека нейросеть
  - COCO keypoints
  - OKS mAP
  - обучить модель позы
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суффикс -pose в имени файла выбирает голову ключевых точек, поэтому
        # аргумент task не нужен.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2) координаты в пикселях
        print(result.boxes.xyxy.shape)     # (N, 4), те же N экземпляров
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Только видимые ключевые точки
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible выводится из третьего столбца ключевых точек и целиком
        # состоит из True, когда чекпойнт предсказывает только (x, y).
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Вариант top-down
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # HRNet работает по схеме top-down: сначала он вырезает каждого
        человека.

        # Если источник людей не задан, он сам подбирает себе детектор
        LibreYOLO9t

        # и пишет этот выбор в лог.

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # В coco8-pose.yaml встроен скрипт скачивания, поэтому нужно явное
        # разрешение, если данных ещё нет локально.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Свой датасет
      language: python
      code: |
        from libreyolo import LibreYOLO

        # В data.yaml должен быть объявлен kpt_shape, а строки разметки должны
        # содержать ровно 5 + K * D полей.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() возвращает обычный dict, а не объект.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика маршрутизирует по суффиксу файла, поэтому экспортированный
        # артефакт загружается как чекпойнт и возвращает тот же объект Results.
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Определение

Оценка позы возвращает структуру, а не только границы. Каждый экземпляр
по-прежнему получает рамку, класс и оценку, а вдобавок — `K` ключевых точек в
фиксированном порядке, так что индекс 5 означает одну и ту же часть тела у
каждого экземпляра и на каждом изображении. Этот порядок задаёт набор меток;
ничто в выводе не называет ключевую точку по имени.

`pose` — канонический ключ задачи, а суффикс `-pose` в имени файла чекпойнта
выбирает эту задачу, поэтому при загрузке опубликованных весов `task=` не нужен.

`predict()` заполняет `result.keypoints` рядом с `result.boxes`. `.data` имеет
форму `(N, K, 2)` или `(N, K, 3)` и выровнен по строкам с рамками, поэтому
экземпляр `i` в одном — это экземпляр `i` в другом. `.xy` выделяет координаты в
пикселях, а `.xyn` нормирует их по размеру исходного изображения. `.conf` — это
третий столбец, когда чекпойнт его предсказывает, и `None`, когда нет, а
`.has_visible` — булева маска, полученная из него, целиком состоящая из True,
когда третьего столбца нет.

Такой вывод дают две архитектуры. Одностадийная модель предсказывает
рамки и ключевые точки за один проход. Top-down-модель сначала запускает
детектор, вырезает каждый экземпляр и регрессирует ключевые точки внутри кропа,
поэтому её точность зависит от стоящего перед ней детектора.

## Модели

Три семейства поддерживают и обучение, и предсказание:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) и
[YOLO-NAS](/docs/models/yolo-nas) — все одностадийные. RF-DETR требует своего
extra, `pip install "libreyolo[rfdetr]"`. RF-DETR и EdgeCrafter поставляются с
опубликованными чекпойнтами для позы, и оба дообучаются на одноклассовых
датасетах только с людьми; голова ключевых точек у EdgeCrafter фиксируется при
создании модели и не принимает датасет, где объявлено другое число точек, тогда
как RF-DETR переинициализирует свою голову под него. YOLO-NAS скачивает веса с
собственного CDN компании Deci.AI под некоммерческой лицензией, и LibreYOLO их
не публикует; его голова позы тоже перестраивается под новое число ключевых
точек, и это единственное из трёх семейств, у которого число классов не
зафиксировано на единице, так что именно оно подходит для многоклассового или
нечеловеческого скелета — например, для позы животных.

[HRNet](/docs/models/hrnet) — вариант top-down. Он предсказывает, валидирует и
экспортирует, а его `train()` бросает `NotImplementedError`. Если источник
людей не задан, он автоматически подбирает себе детектор LibreYOLO9t;
`cropped=True` считает всё изображение одним экземпляром, `person_boxes=`
принимает уже готовые рамки, а `person_detector=` задаёт другой детектор.

[SenseNova-Vision](/docs/models/sensenova-vision) тоже выдаёт ключевые точки.
Это генеративная модель, работающая по промпту, со своей фабрикой, `LibreVLM`, и
своим extra; если словарь не задан, `set_task("pose")` по умолчанию берёт
категорию человека. Её веса некоммерческие, а задержка на изображение намного выше, чем у
специализированной головы позы, потому что каждое предсказание — это
диффузионное декодирование.

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Число и порядок ключевых точек — свойства чекпойнта, а не библиотеки, поэтому
модель, обученная на другом скелете, возвращает другое `K` и другой смысл
каждого индекса. Что лежит в третьем столбце ключевых точек — тоже свойство
чекпойнта: EdgeCrafter пишет туда константу, а не оценку по каждой точке, и у
него вообще нет головы для рамок, поэтому каждая его рамка позы — это область,
охватывающая собственные ключевые точки этого экземпляра. Про источники, стриминг и
обработку результатов см. [предсказание](/docs/predict).

## Формат датасета

Раскладка та же, что и для детекции: один файл разметки `.txt` на изображение,
его находят, заменив `images` на `labels` в пути к изображению и сменив
расширение.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Строка — это строка детекции, к которой дописаны ключевые точки:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Полей ровно `5 + K * D`, где `D` — второе значение `kpt_shape`. Координаты
рамки и ключевых точек — нормированные числа с плавающей точкой относительно
ширины и высоты исходного изображения. Видимость `v` присутствует, только когда
`D` равно 3, и принимает значения `0`, `1` или `2`.

YAML добавляет к общему контракту два ключа:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` обязателен и равен `[K, 2]` или `[K, 3]`. `flip_idx` необязателен и
представляет собой перестановку `0..K-1`, которая для каждой ключевой точки
задаёт индекс, который она получает после горизонтального отражения, — именно так
левое запястье остаётся левым запястьем. Если его не указать, аугментация
горизонтальным отражением для ключевых точек отключается, а не применяется с
неправильным порядком индексов.

## Обучение

<code-tabs name="train" />

Обучение продолжается с опубликованного чекпойнта `-pose`, в котором уже есть
голова ключевых точек; задача читается из загруженного чекпойнта, а не из
флага, переданного при обучении, поэтому чекпойнт детекции не превращается в
запуск оценки позы от одной просьбы. Для EdgeCrafter `kpt_shape` в вашем YAML
должен точно совпадать с головой, поскольку она фиксируется при создании
модели, а RF-DETR и YOLO-NAS вместо этого меняют размер головы под другое
число. Про датасеты, аугментацию, обучение на нескольких GPU и логгеры см.
[обучение](/docs/train).

## Валидация

`val()` возвращает обычный словарь с ключами `metrics/`. Оценка идёт по
COCO-протоколу для ключевых точек на основе Object Keypoint Similarity: эта
мера взвешивает ошибку расстояния каждой ключевой точки масштабом экземпляра и
допуском для конкретной точки, поэтому играет ту же роль, что IoU для рамок.
Нужен `pycocotools`, он входит в базовую установку.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` — главное число, средняя точность, усреднённая по
порогам OKS от 0.50 до 0.95, и именно по ней обучение выбирает лучшую эпоху.
`metrics/keypoints_mAP50` и `metrics/keypoints_mAP75` — версии с одним порогом,
а `metrics/keypoints_mAP_M` и `metrics/keypoints_mAP_L` разбивают среднее по
площади экземпляра, на средние и крупные; корзины для мелких COCO-евалюация
ключевых точек не определяет. Соответствующие показатели средней полноты —
`metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`,
`metrics/keypoints_AR75`, `metrics/keypoints_AR_M` и
`metrics/keypoints_AR_L`. У этой задачи каждый ключ начинается с `keypoints_`,
поэтому ключей `mAP` по рамкам, которые возвращает детектор, здесь нет.

## Экспорт

<code-tabs name="export" />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Набор поддерживаемых форматов различается по семействам;
матрица на странице каждой модели генерируется из проверенного набора, а не
набирается вручную. О форматах, их extras и ограничениях см.
[экспорт и развёртывание](/docs/export).
