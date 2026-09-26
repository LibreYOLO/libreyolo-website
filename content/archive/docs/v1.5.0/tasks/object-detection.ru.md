---
title: Детекция объектов
seo_title: Детекция объектов в LibreYOLO
description: >-
  Детекция объектов рамками по осям в LibreYOLO: семейства, которые решают эту
  задачу, формат разметки и вызовы predict, train, val и export.
lead: >-
  Детекция объектов находит каждый экземпляр объекта на изображении и возвращает
  для него прямоугольник по осям, метку класса и оценку. Ключ задачи — detect.
keywords:
  - детекция объектов python
  - найти объекты на изображении нейросеть
  - ограничивающие рамки bounding box
  - библиотека детекции объектов mit
  - альтернатива ultralytics yolo
  - обучить детектор объектов на своём датасете
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Другое семейство, тот же вызов'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает модель по чекпойнту, и каждый детектор возвращает
        # тот же объект Results, поэтому смена семейства — правка в одну строку.
        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy.shape)
    - label: Видео и потоки
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Любой источник, который принимает библиотека: файл, каталог, URL,
        # индекс веб-камеры, RTSP-поток или список .streams.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yaml при первом запуске скачивает выборку из 128 изображений.
        # Для настоящего запуска укажите в data свой YAML датасета.
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() возвращает обычный dict, а не объект.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Запуск экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает по суффиксу файла, поэтому экспортированный артефакт
        # загружается как чекпойнт и возвращает тот же объект Results.
        model = LibreYOLO("LibreYOLO9t.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## Определение

Детекция объектов отвечает на вопрос, где находится каждый объект и что это
такое. На входе одно изображение, на выходе по одной строке на экземпляр:
четыре числа для прямоугольника, индекс класса и оценка. Ничего о форме на
уровне пикселей, об ориентации или о частях объекта сюда не входит — именно
это отделяет задачу от [сегментации экземпляров](/docs/tasks/instance-segmentation),
[повёрнутых рамок](/docs/tasks/oriented-detection) и
[позы](/docs/tasks/pose-estimation).

`detect` — канонический ключ задачи и значение по умолчанию: чекпойнт, в имени
файла которого нет суффикса задачи, загружается как детектор.

`predict()` заполняет `result.boxes`. `.xyxy` даёт углы в пикселях на холсте
исходного изображения, `.conf` — оценку, а `.cls` — индекс класса в
`result.names`. `.xywh`, `.xyxyn` и `.xywhn` — производные представления тех же
строк, а `.id` содержит id трека, как только подключён трекер. Итерирование
объекта `Boxes` даёт срезы по одной строке, поэтому `box.cls`, `box.conf` и
`box.xyxy` работают для каждой детекции по отдельности.

## Модели

Двенадцать семейств поддерживают и обучение, и предсказание: [YOLOv9](/docs/models/yolov9),
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [D-FINE](/docs/models/d-fine),
[DEIM](/docs/models/deim), [Dome-DETR](/docs/models/dome-detr),
[YOLO-NAS](/docs/models/yolo-nas),
[YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7),
[RTMDet](/docs/models/rtmdet) и [PicoDet](/docs/models/picodet). YOLOv9 и
RF-DETR — два флагманских семейства, новые возможности появляются в них
первыми. RF-DETR требует своего extra, `pip install "libreyolo[rfdetr]"`;
остальные работают на базовом пакете.

Ещё одиннадцать поддерживают предсказание, валидацию и экспорт, но их
`train()` бросает `NotImplementedError`: [LW-DETR](/docs/models/lw-detr),
[DETR](/docs/models/detr), [Deformable DETR](/docs/models/deformable-detr),
[DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) и
[EfficientDet](/docs/models/efficientdet).

Линия Darknet — [YOLOv1](/docs/models/yolov1),
[YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) и
[YOLOv4](/docs/models/yolov4) — сохранена как замороженный экспонат:
предсказание, валидация и экспорт работают, обучение — нет.

Отдельная группа берёт список классов во время выполнения, а не из чекпойнта,
поэтому она детектирует объекты по названиям, которые ни разу не встречались
при обучении:
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) и [OV-DEIM](/docs/models/ov-deim),
а также визуально-языковые семейства
[Florence-2](/docs/models/florence-2), [Kosmos-2](/docs/models/kosmos-2),
[Qwen3-VL](/docs/models/qwen3-vl), [SmolVLM2](/docs/models/smolvlm2),
[InternVL3](/docs/models/internvl3), [LFM2-VL](/docs/models/lfm2-vl),
[LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) и
[LibreMODUS](/docs/models/libremodus). Они загружаются через собственную
фабрику и свои extras; точный вызов приведён на странице каждой модели.

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`conf` задаёт порог уверенности, а `max_det` ограничивает число строк.
`iou` — порог NMS, поэтому он влияет только на те семейства, которые запускают
NMS; RF-DETR и сквозная голова YOLOv9 декодируют фиксированный набор
предсказаний и игнорируют его. Об источниках, стриминге и обработке
результатов см. [предсказание](/docs/predict).

## Формат датасета

Один файл разметки `.txt` на изображение; его путь получается из пути
изображения, если заменить `images` на `labels` и сменить расширение.

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

В каждой строке ровно пять полей: индекс класса, а за ним нормализованная
рамка, заданная центром и размерами:

```text
<class_id> <cx> <cy> <w> <h>
```

Координаты — числа с плавающей точкой в `[0, 1]`, относительно ширины и высоты
исходного изображения. `w` и `h` должны быть положительными. Отсутствующий или
пустой файл разметки означает, что на изображении нет объектов. В строках нет
ни уверенности, ни id трека.

В YAML перечислены сплиты и классы:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` и `val` могут быть каталогами изображений, `.txt`-файлами со списком
изображений или списками того и другого. `nc` необязателен и, если задан,
должен совпадать с `names`. Родной формат COCO JSON тоже работает: добавьте раздел
`annotations`, который сопоставляет имя сплита с JSON-файлом, — тогда путь
сплита задаёт корень изображений. Если `names` присутствует, он определяет id
меток, поэтому названия категорий в JSON должны с ним совпадать.

## Обучение

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` и `lr0` — аргументы, которые крутят первыми. `lr0` —
тот, который не переносится между семействами: скорость обучения, которую
спокойно выдерживает свёрточный детектор, разваливает обучение трансформерного,
поэтому берите значение со страницы модели, а не из примера другого семейства.
Семейство может и вовсе игнорировать аргумент — на его странице перечислено,
какие именно. О датасетах, аугментации, обучении на нескольких GPU и логгерах
см. [обучение](/docs/train).

## Валидация

`val()` возвращает обычный словарь с ключами `metrics/`, посчитанными по
методике COCO на сплите, который указан в поле `val` в YAML датасета.

<code-tabs name="val" />

`metrics/mAP50-95` — средняя точность, усреднённая по порогам IoU от 0.50 до
0.95, и это главное число. `metrics/mAP50` и `metrics/mAP75` — версии с одним
порогом. `metrics/mAP_small`, `metrics/mAP_medium` и `metrics/mAP_large`
разбивают то же усреднение по площади объектов, а `metrics/AR1`,
`metrics/AR10`, `metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium` и
`metrics/AR_large` — соответствующие показатели средней полноты.
`metrics/AR_max_det` и `metrics/max_det` фиксируют ограничение на число
детекций, с которым выполнялся запуск.

К `metrics/precision` и `metrics/recall` в этой задаче стоит отнестись
внимательно. Они оставлены для обратной совместимости и служат псевдонимами, а
не рабочей точкой: в `metrics/precision` лежит то же значение, что и в
`metrics/mAP50-95`, а в `metrics/recall` — то же, что и в `metrics/AR100`.
График, построенный по ним как по паре precision-recall, показывает одно и то
же число дважды. Ещё четыре ключа повторяются с суффиксом `(B)`, от box, чтобы
ключ детекции читался одинаково и на модели, которая предсказывает ещё и
маски: `metrics/mAP50-95(B)`, `metrics/mAP50(B)`,
`metrics/precision(B)` и `metrics/recall(B)`.

## Экспорт

<code-tabs name="export" />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Набор поддерживаемых форматов различается по семействам;
таблица на странице каждой модели генерируется из проверенного набора, а не
набирается вручную. О форматах, их extras и ограничениях см.
[экспорт и развёртывание](/docs/export).
