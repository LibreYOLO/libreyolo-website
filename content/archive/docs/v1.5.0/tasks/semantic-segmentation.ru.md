---
title: Семантическая сегментация
seo_title: Семантическая сегментация в LibreYOLO
description: >-
  Разметка каждого пикселя классом в LibreYOLO: семейства моделей для этой
  задачи, формат плотных масок и вызовы предсказания, обучения, валидации и
  экспорта.
lead: >-
  Семантическая сегментация присваивает класс каждому пикселю изображения и не
  различает экземпляры одного класса. Ключ задачи — semantic.
keywords:
  - семантическая сегментация python
  - сегментация изображений нейросеть
  - разметка пикселей по классам
  - обучение модели сегментации
  - mIoU
  - mit библиотека сегментации
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Суффикс -sem в имени файла выбирает задачу, поэтому аргумент

        # task не нужен.

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) — id классов на холсте исходного
        изображения

        print(mask.classes)      # id присутствующих классов по возрастанию, без
        255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: По одному классу
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # булев массив (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Другое семейство, тот же вызов'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: На ADE20K
      language: bash
      code: |
        # В ade20k.yaml встроен скрипт скачивания архива на ~1 ГБ, поэтому
        # нужно явное разрешение, если данных нет локально.
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() возвращает обычный словарь, а не объект.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как чекпойнт и возвращает тот же объект Results.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Определение

Семантическая сегментация размечает пиксели, а не объекты. Каждый пиксель
получает один id класса, и две соприкасающиеся на изображении машины
становятся одной областью класса «машина» без границы между ними. Считать
экземпляры — это
[сегментация экземпляров](/docs/tasks/instance-segmentation); размечать каждый
пиксель и одновременно разделять экземпляры —
[паноптическая сегментация](/docs/tasks/panoptic-segmentation).

`semantic` — канонический ключ задачи, а суффикс `-sem` в имени файла
чекпойнта выбирает её, поэтому при загрузке опубликованных весов `task=` не
нужен.

`predict()` заполняет `result.semantic_mask`. `.data` — это целочисленная
карта классов `(H, W)` на холсте исходного изображения, `.classes`
перечисляет присутствующие id по порядку, а `.class_mask(id)` возвращает
булеву выборку `(H, W)` для одного класса. Значение `255` — метка
игнорирования: оно никогда не бывает классом, исключается из функции потерь
и метрик, а `.classes` его не показывает.

## Модели

Обучать и предсказывать умеют три семейства:
[SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) и
[DINOv2](/docs/models/dinov2). SegFormer и LingBot-Vision работают на базовом
пакете и поставляются с опубликованными весами. DINOv2 требует
`pip install "libreyolo[rfdetr]"`, и чекпойнта на хостинге
LibreYOLO у него нет: он загружает исходный бэкбон, а его плотная голова
стартует со случайной инициализации, так что это отправная точка для
обучения, а не готовая модель для предсказаний.

Ещё четыре семейства предсказывают, валидируют и экспортируют, но их `train()`
выбрасывает `NotImplementedError`: [FCN](/docs/models/fcn),
[DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet) и
[EoMT](/docs/models/eomt).

Наборы классов различаются от чекпойнта к чекпойнту, а не от семейства к
семейству. Опубликованные веса получены на датасетах, у которых мало общего в
пространствах меток — среди них 150 классов ADE20K против 19 у Cityscapes, —
поэтому именно `names` чекпойнта говорит, что он умеет размечать, а сравнивать
два чекпойнта можно только тогда, когда они обучены на одном и том же наборе.

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Карта строится как argmax по каждому пикселю, поэтому шага NMS нет, а `iou`
ни на что не влияет. `conf` и `max_det` принимаются ради единообразия API и
ничего не делают в SegFormer, PIDNet и остальных моделях плотного
предсказания; исключение — EoMT, где `conf` фильтрует отбор запросов. Про
источники, стриминг и обработку результатов — в разделе
[предсказание](/docs/predict).

## Формат датасета

Каждому изображению соответствует не файл разметки `.txt`, а плотная
одноканальная маска; её путь получается заменой `images` в пути к изображению
на каталог масок.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Маски — одноканальные изображения без потерь, обычно PNG, причём PNG в
палитровом режиме читаются как индексы палитры. Значение каждого пикселя — это
id класса в диапазоне `0..nc-1`, значение `255` означает игнорирование, а
разрешение маски должно совпадать с разрешением парного изображения.

В YAML добавляются два ключа поверх общего контракта:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` — имя каталога, которое подставляется вместо `images`; по
умолчанию `masks`. `label_mapping` — необязательное переназначение
`{source_id: train_id}`, которое применяется к значениям пикселей маски при
загрузке: именно так датасет с нумерацией от 1 до 150 превращается в нумерацию
от 0 до 149; любое исходное значение, оставшееся без соответствия, становится
игнорируемым, а каждый train id должен попадать в `0..nc-1`.

Если `masks_dir` не указан, загрузчик переключается на запасной путь: маски
растеризуются при загрузке из полигональной разметки, которая находится по
обычному правилу замены `images` на `labels`, а после классов объектов
добавляется класс `background`, так что `nc` увеличивается на единицу.

Канонический загрузчик — `libreyolo.data.SemanticDataset`.

## Обучение

<code-tabs name="train" />

На `imgsz` здесь наложено ограничение, которого нет у детектора. Каждое
семейство задаёт делитель, кратным которому должен быть его вход; делитель
определяется сеткой патчей или выходным шагом; и обучение, и валидация
выбрасывают `ValueError` ещё до старта, если `imgsz` не делится
нацело. Делитель равен 32 для SegFormer, 16 для LingBot-Vision и EoMT, 14 для
DINOv2 и 8 для FCN и PIDNet. Про датасеты, аугментацию, обучение на нескольких
GPU и логгеры — в разделе [обучение](/docs/train).

## Валидация

`val()` возвращает обычный словарь с ключами `metrics/`, посчитанными по
сплиту, который назван ключом `val` в YAML датасета.

<code-tabs name="val" />

`metrics/mIoU` — это среднее по классам intersection over union: для каждого
класса берётся пересечение предсказанных и истинных пикселей, делённое на их
объединение, а затем результаты усредняются по классам. Это основное число, и
именно по нему выбирается лучшая эпоха во время обучения.
`metrics/pixel_accuracy` — доля пикселей, получивших верный класс; большой
фоновый класс может её завысить, поэтому сравнивать стоит по mIoU. Пиксели,
помеченные `255`, не учитываются ни в одной из этих величин. В словаре есть
ещё `fitness` — копия значения mIoU.

## Экспорт

<code-tabs name="export" />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Покрытие форматов различается от семейства к семейству;
матрица на странице каждой модели генерируется из проверенного набора, а не
пишется руками. Про форматы, их дополнительные зависимости и ограничения — в
разделе [экспорт и развёртывание](/docs/export).
