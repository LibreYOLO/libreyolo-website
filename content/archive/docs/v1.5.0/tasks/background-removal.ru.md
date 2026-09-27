---
title: Удаление фона
seo_title: Удаление фона в LibreYOLO
description: >-
  Вырезайте объект из фона в LibreYOLO. Предсказание мягкого альфа-матте, запись
  прозрачного PNG и валидация по MAE и S-measure.
lead: >-
  Удаление фона отделяет объект от всего, что находится за ним. В LibreYOLO это
  задача matte: она возвращает мягкое значение альфы на каждый пиксель, а не
  жёсткую маску переднего плана.
keywords:
  - удаление фона python
  - alpha matting нейросеть
  - убрать фон с изображения нейросеть
  - прозрачный png вырезать объект
  - dichotomous image segmentation
last_verified: 1.5.0
snippets:
  predict:
    - label: Предсказание матте
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 в [0, 1]
    - label: Запись прозрачного PNG
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() совмещает исходное изображение с матте как альфа-каналом.
        result.save("subject.png")

        rgba = result.cutout()   # тот же массив (H, W, 4) uint8 в памяти
        print(rgba.shape)
    - label: Наложение на новый фон
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # белый

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: Валидация и чтение ключей метрик
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Вместо YAML-файла датасета подойдёт каталог, в котором лежат images/ и
        # каталог с матте.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # меньше — лучше
        print(metrics["metrics/Smeasure"])   # fitness, больше — лучше
  export:
    - label: Экспорт
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Запуск экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный
        # артефакт загружается как обычный чекпойнт и возвращает тот же Results.
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## Определение

Задача `matte` предсказывает по одному значению альфы на пиксель из одного
RGB-изображения: `1` — полностью передний план, `0` — полностью фон. Значение
непрерывное, а не бинарное, и в этом весь смысл задачи. До жёсткой маски отсюда
один шаг — порог 0.5, а мягкое матте дополнительно несёт частичное покрытие на
волосах, шерсти и смазанных движением краях, которое бинарная маска выбрасывает.

Предсказание заполняет `result.matte` — объект `Matte` с массивом `(H, W)`
float32 в `[0, 1]` на холсте исходного изображения, доступным как NumPy через
`.array`. `result.cutout()` объединяет исходное изображение с этой альфой в
массив RGBA `(H, W, 4)` uint8, а `result.save(path)` записывает то же самое в
PNG с прозрачным фоном. `result.boxes` остаётся пустым, поэтому `conf`, `iou` и
`max_det` ни на что не влияют.

## Модели

За `matte` отвечают два семейства, и прямой проход у них общий.

[BiRefNet](/docs/models/birefnet) — сеть с билатеральной привязкой, вокруг
которой построена задача; здесь она опубликована одним чекпойнтом уровня Swin-L.

[FeyNobg](/docs/models/feynobg) — углублённый вариант от Feyn Inc.: архитектура
BiRefNet, у которой третья стадия Swin выросла с 18 до 24 блоков, после чего
модель переобучили. LibreYOLO переиспользует для него прямой проход,
предобработку и однологитный выход BiRefNet, поэтому предсказание, валидация и
работа с чекпойнтами ведут себя одинаково; веса и принадлежность к семейству у
FeyNobg свои.

Лицензии на веса у них разные. Обе указаны на страницах моделей, а решающей
считается лицензия в репозитории Hugging Face конкретного чекпойнта.

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Оба семейства работают на фиксированном родном холсте 1024x1024 и масштабируют
матте обратно под исходное изображение. Другое разрешение не поддерживается:
таблицы относительных позиций в бэкбоне Swin привязаны к этому размеру, и при
несовпадении ошибки не будет — они просто плохо интерполируются.
`Results.save()` определён только для матте-результатов, и ему нужно исходное
изображение: если не передать его явно, метод перечитает изображение по
`Results.path`. Об источниках, стриминге и обработке результатов — в разделе
[предсказание](/docs/predict).

## Формат датасета

При валидации матте каждому RGB-изображению сопоставляется одноканальное
эталонное альфа-матте (ground truth) с тем же именем без расширения, где 0 —
фон, а 255 — передний план.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Достаточно передать этот корень как `data=`: каталог с матте определяется
автоматически среди `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` и `alpha/`.
Альтернатива — YAML-файл датасета: `path` задаёт корень, а `val_images` и
`val_mattes` — каталоги относительно него:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` и `names` — заглушки схемы; матте-модель возвращает `Results.matte`, а не
детекции. Значения матте читаются как альфа в `[0, 1]` делением на 255, а матте,
форма которого отличается от холста предсказания, билинейно масштабируется под
него. Полный контракт — в разделе
[форматы датасетов](/docs/reference/dataset-formats).

## Обучение

Ни у одного из матте-семейств нет реализации обучения: `train()` в обоих случаях
выбрасывает `NotImplementedError`, а поддержка матте покрывает только
предсказание, валидацию и экспорт. На странице каждой модели указан исходный
проект, в котором есть код обучения, и скрипт конвертации, который переносит
чекпойнт обратно.

## Валидация

`val()` вызывает собственный `predict` модели, поэтому валидация использует ровно
ту предобработку, что и семейство, а обе метрики считаются на холсте исходного
изображения.

<code-tabs name="val" />

`metrics/MAE` — средняя абсолютная ошибка относительно эталонной альфы, в
`[0, 1]`, и меньше — лучше. `metrics/Smeasure` — S-measure из работы Fan et al.
(ICCV 2017), структурное сходство, которое учитывает, насколько верно переданы
форма объекта и отверстия в нём, чего попиксельное усреднение само по себе не
видит; больше — лучше. S-measure заодно служит значением `fitness` — числом, по
которому выбирается лучший чекпойнт. Ни одна из метрик не зависит от разрешения.

## Экспорт

Экспортированная матте-модель загружается обратно через `LibreYOLO()` по
суффиксу файла, поэтому артефакт ведёт себя как чекпойнт и возвращает те же
`Results`.

<code-tabs name="export" />

Проверенный путь для этой задачи — TorchScript. Конвертация в ONNX проходит, но
не дотягивает до той же планки паритета, а остальные форматы недоступны.
Покрытие по форматам — на страницах [BiRefNet](/docs/models/birefnet) и
[FeyNobg](/docs/models/feynobg), а также в
[полной матрице экспорта](/docs/reference/export-matrix).
