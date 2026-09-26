---
title: Сегментация экземпляров
seo_title: Сегментация экземпляров в LibreYOLO
description: >-
  Сегментация отдельных объектов в LibreYOLO: семейства моделей для этой задачи,
  формат разметки полигонами и вызовы предсказания, обучения, валидации и
  экспорта.
lead: >-
  Сегментация экземпляров находит каждый экземпляр объекта и возвращает для него
  попиксельную маску — вместе с рамкой, классом и оценкой, которые выдаёт
  детектор. Ключ задачи — segment.
keywords:
  - сегментация экземпляров python
  - маски объектов нейросеть
  - обучение модели сегментации
  - разметка полигонами yolo
  - mit библиотека сегментации
  - mask mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суффикс -seg в имени файла выбирает голову масок, поэтому
        # аргумент task не нужен.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), по одной маске на детекцию
        print(result.boxes.xyxy.shape)   # (N, 4), те же N строк
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Контуры масок
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDFINEn-seg.pt")

        result = model(SAMPLE_IMAGE)


        # .xy — список контуров (P, 2) в пикселях, .xyn — те же, но
        нормализованные.

        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'Другое семейство, тот же вызов'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Обучение продолжается с опубликованных весов сегментации, с головой
        масок.

        # data должен указывать на датасет, в разметке которого есть полигоны.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Из весов детекции
      language: bash
      code: |
        # В весах детекции нет головы масок, поэтому это явный перенос:
        # голова стартует необученной. Разрешает его именно явно
        # указанный task=segment.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # маски
        print(metrics["metrics/mAP50-95(M)"])    # маски, явно
        print(metrics["metrics/mAP50-95(B)"])    # рамки
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как чекпойнт и возвращает тот же объект Results.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Определение

Сегментация экземпляров — это детекция плюс форма. Каждый экземпляр объекта
по-прежнему получает рамку, класс и оценку, а вдобавок — бинарную маску,
которая покрывает принадлежащие ему пиксели. Маски могут перекрываться, а
пиксели, не принадлежащие ни одному объекту, остаются без назначения — именно
это отличает задачу от
[семантической сегментации](/docs/tasks/semantic-segmentation) и
[паноптической сегментации](/docs/tasks/panoptic-segmentation).

`segment` — канонический ключ задачи, а суффикс `-seg` в имени файла чекпойнта
выбирает её, поэтому при загрузке опубликованных весов `task=` не нужен.

`predict()` заполняет `result.masks` рядом с `result.boxes`. `.data` — это
стек масок `(N, H, W)` размером с исходное изображение, построчно выровненный
с рамками, так что маска `i` относится к рамке `i`. `.xy` превращает каждую маску
в её наибольший внешний контур — массив пикселей `(P, 2)`, а `.xyn` даёт тот же
контур в нормализованном виде.

## Модели

Обучать и предсказывать маски умеют четыре семейства:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[D-FINE](/docs/models/d-fine) и [RTMDet](/docs/models/rtmdet). Для RF-DETR
нужна своя дополнительная зависимость, `pip install "libreyolo[rfdetr]"`;
остальные три работают на базовом пакете.

[Mask R-CNN](/docs/models/mask-rcnn) предсказывает, валидирует и экспортирует
маски, но его `train()` выбрасывает `NotImplementedError`.

[EoMT](/docs/models/eomt) предсказывает и валидирует маски и тоже не
поддерживает обучение, а его экспорт ограничен ещё сильнее: `export()`
принимает только семантическую задачу и выбрасывает `NotImplementedError` для
`segment` и `panoptic`, потому что контракт среды выполнения для query-масок,
который нужен этим двум, пока не определён. Используйте EoMT для масок экземпляров в Python, а не через
экспортированный граф.

Отдельная группа сегментирует по промпту, а не по списку классов: клик, рамка
или фраза указывают на объект, и модель возвращает его маску. Так работают
[SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2),
[SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam),
[EdgeTAM](/docs/models/edgetam) и [PicoSAM3](/docs/models/picosam3), а также
[SenseNova-Vision](/docs/models/sensenova-vision), у которого сегментация идёт
по описанию (referring): модель принимает фразу, называющую один объект. Они
загружаются через собственную фабрику и дополнительные зависимости, а точный
вызов приведён на странице каждой модели.

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`conf` и `max_det` формируют вывод так же, как в детекции, а маски фильтруются
вместе с рамками, которым принадлежат. Про источники, стриминг и обработку
результатов — в разделе [предсказание](/docs/predict).

## Формат датасета

Структура та же, что и в детекции: один файл разметки `.txt` на изображение;
путь к нему получается, если заменить `images` на `labels` в пути к
изображению и сменить расширение.

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

Меняется сама строка. Сегмент — это индекс класса, за которым идёт плоский
список координат полигона:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

Точек должно быть не меньше трёх, то есть количество координат после индекса
класса чётное и не меньше шести, а полигон не должен быть вырожденным.
Координаты — числа с плавающей точкой в `[0, 1]` относительно ширины и высоты
исходного изображения. Строка детекции из пяти полей тоже принимается в
датасете сегментации и читается как прямоугольный сегмент — благодаря этому
датасет только с рамками загружается без отдельного прохода конвертации.

YAML — тот же, что и в детекции:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

Нативный COCO JSON тоже работает: добавьте раздел `annotations`, который
сопоставляет имя сплита с JSON-файлом, а путь сплита задаёт корневой каталог
изображений.

## Обучение

<code-tabs name="train" />

По умолчанию обучение продолжается с опубликованного чекпойнта `-seg`.
Стартовать с весов детекции тоже можно, но это осознанный перенос: в таких
весах нет головы масок, поэтому она стартует необученной, а разрешает такую
замену именно `task=segment`. Про датасеты, аугментацию, обучение на нескольких
GPU и логгеры — в разделе [обучение](/docs/train).

## Валидация

`val()` возвращает обычный словарь с ключами `metrics/`. Рамки и маски
оцениваются отдельно, обе — по протоколу COCO, и основными считаются числа по
маскам.

<code-tabs name="val" />

Ключи без суффикса содержат результаты по маскам: `metrics/mAP50-95`,
`metrics/mAP50`, `metrics/mAP75`, затем `metrics/mAP_small`,
`metrics/mAP_medium` и `metrics/mAP_large` по площади объекта, а также
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium`, `metrics/AR_large` для средней полноты. В
`metrics/AR_max_det` и `metrics/max_det` записан лимит числа детекций, с
которым шёл запуск.

Четыре величины публикуются ещё и с явным суффиксом — `(M)` для маски и `(B)`
для рамки, — чтобы сравнение не зависело от того, какое число семейство решило
считать основным: `metrics/mAP50-95(M)` и `metrics/mAP50-95(B)`,
`metrics/mAP50(M)` и `metrics/mAP50(B)`, `metrics/precision(M)` и
`metrics/precision(B)`, `metrics/recall(M)` и `metrics/recall(B)`. Ключей
`metrics/precision` и `metrics/recall` без суффикса в этой задаче нет.

Ключи precision и recall читайте внимательно. Они оставлены для обратной
совместимости и служат псевдонимами, а не рабочей точкой: в
`metrics/precision(M)` лежит то же значение, что и в `metrics/mAP50-95(M)`, а в
`metrics/recall(M)` — то же, что и AR по маскам при 100 детекциях; с `(B)` для
рамок всё устроено так же. График по паре таких ключей покажет одно и то же
число дважды.

## Экспорт

<code-tabs name="export" />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Покрытие форматов для сегментации уже, чем для детекции у
того же семейства. Матрица на странице каждой модели генерируется из
проверенного набора и называет причину, по которой цель недоступна. Про
форматы, их дополнительные зависимости и ограничения — в разделе
[экспорт и развёртывание](/docs/export).
