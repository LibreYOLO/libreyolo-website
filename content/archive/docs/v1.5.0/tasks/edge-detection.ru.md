---
title: Выделение границ
seo_title: Выделение границ в LibreYOLO
description: >-
  Предсказывайте плотную карту вероятности границ по одному изображению в
  LibreYOLO. Конвертируйте чекпойнт, применяйте порог к карте, считайте ODS и
  OIS на валидации и экспортируйте.
lead: >-
  Выделение границ предсказывает, насколько вероятно, что каждый пиксель лежит
  на границе объекта. В LibreYOLO это задача edge, которая возвращает плотную
  карту вероятности на исходном холсте изображения, а не набор отрезков.
keywords:
  - выделение границ python
  - детектор границ нейросеть
  - карта вероятности границ
  - ODS OIS F-мера
  - плотное предсказание границ
last_verified: 1.5.0
snippets:
  predict:
    - label: Предсказание карты границ
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Чекпойнта для границ в LibreYOLO нет; сначала конвертируйте свой
        (ниже).

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)          # (H, W) float32 в [0, 1]

        print(edges.binary(0.5).sum())    # число пикселей границ при пороге 0.5
    - label: Выбор своего порога
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # Непрерывная карта сохраняется, чтобы порог оставался вашим решением.
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Сохранение визуализации
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot() рисует карту; он определён для результатов edge и normal.
        result.plot().save("edges.png")
  val:
    - label: Валидация и чтение ключей метрик
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Изменение перебора и допуска сопоставления
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Экспорт
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Запуск экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как любой чекпойнт и возвращает тот же Results.
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## Определение

Задача `edge` предсказывает по одному RGB-изображению одно значение вероятности
на пиксель: `0` — не граница, `1` — граница. Карта остаётся непрерывной, поэтому
выбор порога, который превращает её в бинарное изображение границ, оставлен
вызывающему коду, а подходящий порог зависит от датасета и от того, как
результат используется дальше.

Предсказание заполняет `result.edges` — структуру `EdgeMap` с массивом `(H, W)`
типа float32 со значениями в `[0, 1]` на исходном холсте изображения. `.array`
возвращает эту карту как NumPy-массив, а `.binary(threshold)` — булеву маску.
`result.boxes` остаётся пустым, поэтому `conf`, `iou` и `max_det` ни на что не
влияют. `Results.plot()` эту задачу покрывает и рисует карту напрямую.

## Модели

Задачу `edge` обслуживают три семейства.

[DexiNed](/docs/models/dexined), Dense Extreme Inception Network, объединяет
несколько боковых выходов в одну карту вероятности и работает на родном
разрешении 352 px.

[TEED](/docs/models/teed), Tiny and Efficient Edge Detector, — небольшая сеть с
тем же родным разрешением 352 px, но с шагом понижения разрешения 4 против 16 у
DexiNed, поэтому она принимает больше значений `imgsz`.

[LibreMODUS](/docs/models/libremodus) выдаёт границы в стиле Canny как одну из
целей any-to-any модели. Ему нужен extra `modus` и собственный аутентифицированный
аккаунт Hugging Face, и он не предлагает ни `val()`, ни `export()`, поэтому в
разделах про валидацию и экспорт ниже он не участвует.

## Предсказание

LibreYOLO не публикует чекпойнт для границ. Официально выпущенные веса DexiNed и
TEED обучены на BIPED, а опубликованные условия использования этого датасета
ограничивают его некоммерческими целями, поэтому LibreYOLO не размещает у себя
их копию. Конвертируйте чекпойнт, на который у вас есть лицензия, а затем
загрузите полученный файл по пути:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

Имя файла должно нести суффикс задачи `-edge`, иначе загрузчик его не распознает.
`imgsz` должен делиться на суммарный шаг понижения разрешения сети, и LibreYOLO
выдаёт понятную ошибку с указанием делителя, если это не так. Об источниках,
стриминге и обработке результатов — в разделе [предсказание](/docs/predict).

## Формат датасета

При валидации границ каждому RGB-изображению соответствует одноимённая
одноканальная карта того же разрешения и, необязательно, маска валидности.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Цель — одноканальный PNG или TIF, а не RGB-визуализация. Целочисленные карты
делятся на максимум своего dtype; вещественные должны быть конечными и уже
лежать в `[0, 1]`. Пиксель маски считается валидным, когда он ненулевой, а
пиксели дополнения никогда не влияют на метрику. `edge_invert: true` покрывает
источники, где чёрные границы хранятся на белом. Полный контракт — в разделе
[форматы датасетов](/docs/reference/dataset-formats).

## Обучение

Ни у одного семейства для границ в LibreYOLO нет реализации обучения: `train()`
выбрасывает `NotImplementedError` у всех трёх. На странице каждой модели назван
скрипт конвертации, который превращает чекпойнт, обученный где-то ещё, в
чекпойнт, который может загрузить LibreYOLO.

## Валидация

`val()` выдаёт F-меры в стиле BSDS. Сначала непрерывные предсказания утончаются
подавлением немаксимумов по градиенту в четырёх направлениях, затем
предсказанные и эталонные пиксели границ сопоставляются один к одному в пределах
допуска по расстоянию.

<code-tabs name="val" />

`metrics/ODS` — F-мера при оптимальном пороге для датасета: на каждом пороге
счётчики совпадений суммируются по всему датасету, и выдаётся лучшая из этих
суммарных F-мер. Она же и `fitness` — число, по которому выбирается лучший
чекпойнт. `metrics/OIS` — F-мера при оптимальном пороге для изображения, среднее
по изображениям от лучшей F-меры каждого изображения, так что каждое изображение
выбирает свой порог. `metrics/best_threshold` — тот единственный порог, на
котором получилась ODS; именно его стоит переиспользовать в `edges.binary()` при
инференсе.

Перебор настраивают два аргумента. `edge_thresholds` — набор перебираемых
порогов, по умолчанию от 0.01 до 0.99 с шагом в сотую. `edge_max_dist` — допуск
сопоставления как доля диагонали изображения, по умолчанию `0.0075`; пара,
разнесённая дальше, совпадением не считается.

## Экспорт

Экспортированная модель границ загружается обратно через `LibreYOLO()` по
суффиксу файла, поэтому файл `.onnx` ведёт себя как чекпойнт и возвращает тот же
`Results`.

<code-tabs name="export" />

Экспорт границ работает по контракту среды выполнения с фиксированным
разрешением и batch=1: `dynamic` и `batch`, отличный от 1, отклоняются, а
экспортированный граф выдаёт одну объединённую карту вероятности. Поддержка по
форматам — на страницах [DexiNed](/docs/models/dexined) и
[TEED](/docs/models/teed) и в
[полной матрице экспорта](/docs/reference/export-matrix). В разделе
[экспорт](/docs/export) перечислены аргументы, которые принимает каждый формат.
