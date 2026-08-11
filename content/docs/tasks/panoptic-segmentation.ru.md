---
title: Паноптическая сегментация
seo_title: Паноптическая сегментация в LibreYOLO
description: >-
  Назначайте каждому пикселю ровно один сегмент в LibreYOLO: семейства, которые
  обслуживают задачу, формат датасета COCO-panoptic и вызовы предсказания и
  валидации.
lead: >-
  Паноптическая сегментация назначает каждому пикселю ровно один
  непересекающийся сегмент, объединяя исчисляемые экземпляры объектов с
  аморфными областями фона. Ключ задачи — panoptic.
keywords:
  - паноптическая сегментация python
  - panoptic quality PQ
  - things и stuff сегментация
  - формат COCO panoptic
  - карта идентификаторов сегментов
  - метрика PQ
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суффикс -panoptic в имени файла выбирает задачу, поэтому аргумент
        # task не нужен.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) идентификаторы сегментов
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: По одному сегменту
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # булев массив (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Чекпойнт поменьше
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() возвращает обычный dict, а не объект.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## Определение

Паноптическая сегментация — объединение двух других задач сегментации. Каждый
пиксель получает ровно один сегмент, сегменты никогда не перекрываются, а сам
сегмент — это либо thing, отдельный экземпляр исчисляемого объекта, либо stuff,
аморфная область вроде неба или дороги. Поэтому она строже, чем
[сегментация экземпляров](/docs/tasks/instance-segmentation), которая оставляет
фоновые пиксели без назначения и допускает перекрытие масок, и строже, чем
[семантическая сегментация](/docs/tasks/semantic-segmentation), которая
размечает каждый пиксель, но сливает соприкасающиеся экземпляры одного класса.

`panoptic` — канонический ключ задачи, а суффикс `-panoptic` в имени файла
чекпойнта выбирает её, поэтому при загрузке опубликованных весов `task=` не
нужен.

`predict()` заполняет `result.panoptic`. `.data` — целочисленная карта
идентификаторов сегментов размера `(H, W)` на холсте исходного изображения.
`.segments_info` — список словарей, по одному на сегмент, каждый несёт как
минимум `{"id", "category_id"}`, где `id` совпадает со значением в карте, а
`category_id` индексирует `result.names`. `.segment_ids` перечисляет
присутствующие идентификаторы в отсортированном порядке, а
`.segment_mask(id)` возвращает булеву выборку `(H, W)` для одного сегмента.
Идентификатор сегмента `0` — значение void: неразмеченные пиксели, они
исключены из метрики и не попадают в `.segment_ids`.

Разделение на thing и stuff — свойство категории, а не отдельного сегмента.
Оно хранится в метаданных категорий набора меток, и для удобства предсказание
может копировать его в каждый сегмент как `"isthing"`, но источником истины
остаются метаданные категории.

## Модели

[EoMT](/docs/models/eomt) — семейство, которое обслуживает эту задачу через
`LibreYOLO()`. Оно работает на базовом пакете и поставляет паноптические
чекпойнты трёх размеров — s, b и l, обученные на COCO.

[SenseNova-Vision](/docs/models/sensenova-vision) тоже выдаёт паноптические
карты. Это генеративная модель, работающая по промпту, со своей фабрикой
`LibreVLM` и своим extra; если словарь не задан, она откатывается к категориям
COCO panoptic, на которых её настраивали. Её веса — некоммерческие. Задержка на
изображение намного выше, чем у специализированного сегментатора, потому что
каждое предсказание — это диффузионное декодирование.

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`conf` фильтрует отбор запросов. Про источники, стриминг и обработку
результатов — в разделе [предсказание](/docs/predict).

## Формат датасета

LibreYOLO использует формат COCO-panoptic без изменений, как он описан у
Kirillov et al., CVPR 2019. Отдельной паноптической раскладки для LibreYOLO
нет.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Каждому изображению соответствует один RGB PNG того же разрешения, где цвет
каждого пикселя кодирует идентификатор сегмента, которому этот пиксель
принадлежит:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Идентификатор сегмента `0`, чёрный цвет в RGB, — это void: неразмеченные
пиксели, которые не поощряют и не штрафуют предсказание. Любой другой пиксель
принадлежит ровно одному сегменту.

JSON перечисляет для каждого изображения PNG с идентификаторами сегментов и
сами сегменты внутри него:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` задаёт имя PNG внутри паноптического каталога, а
`segments_info[].id` совпадает со значением в этом PNG. `iscrowd` помечает
групповые области: они никогда не считаются ложноотрицательными, а
предсказание, которое по большей части покрывает такую область, не считается
ложноположительным. `isthing` находится в `categories`, никогда — в отдельном
сегменте.

В YAML указаны оба пути:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` и `panoptic_dir` принимают либо один путь, либо отображение по
сплитам. Исходные идентификаторы категорий COCO обычно идут с пропусками, тогда
как модели предсказывают непрерывный диапазон `0..nc-1`, поэтому идентификаторы
переотображаются через `names` по имени категории. Категория из JSON, которой
нет в `names`, — это ошибка, а не молчаливое отбрасывание, потому что иначе она
навсегда засчитывалась бы как ложноотрицательная.

Канонический загрузчик — `libreyolo.data.PanopticDataset`.

## Обучение

Обучать паноптическую сегментацию в LibreYOLO сегодня не умеет ни одно
семейство: `train()` у EoMT выбрасывает `NotImplementedError`, поэтому
паноптические чекпойнты используются в том виде, в каком они опубликованы.

## Валидация

`val()` возвращает обычный словарь с ключами `metrics/`, посчитанными в
разрешении эталонной разметки (ground truth) по сплиту, который назван ключом
`val` в YAML датасета. Предсказанный и истинный сегменты одной категории
сопоставляются, когда их IoU превышает 0.5, и такое сопоставление единственно.

<code-tabs name="val" />

`metrics/PQ` — это Panoptic Quality, главное число. Внутри одной категории оно
равно произведению двух множителей. Segmentation quality — средний IoU по
сопоставленным сегментам, он говорит, насколько хорошо совпадают сопоставленные
формы. Recognition quality — это `TP / (TP + 0.5 FP + 0.5 FN)`, F1-мера самого
сопоставления, она говорит, сколько сегментов вообще было найдено. Все три
величины затем усредняются по категориям, которые встретились, и возвращаются
как `metrics/PQ`, `metrics/SQ` и `metrics/RQ`, так что итоговый PQ — это
среднее произведений по категориям, а не произведение двух возвращаемых
средних.

`metrics/PQ_things` и `metrics/PQ_stuff` усредняют тот же PQ по категориям
отдельно для категорий thing и категорий stuff, а `metrics/categories` считает
категории, которые встретились и потому попали в усреднение. В
словаре также есть `fitness` — копия значения PQ.

## Экспорт

Паноптические чекпойнты не экспортируются. `export()` выбрасывает
`NotImplementedError` для этой задачи, потому что для выхода с query-масками
пока нет контракта экспорта в среду выполнения. Семантическая задача у EoMT
экспортируется; см.
[семантическую сегментацию](/docs/tasks/semantic-segmentation) и
[экспорт и развёртывание](/docs/export).
