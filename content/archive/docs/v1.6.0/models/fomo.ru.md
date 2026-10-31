---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: локализация точек, обучение и экспорт в LibreYOLO'
description: >-
  Запуск FOMO (Faster Objects, More Objects) в LibreYOLO: крошечный детектор для
  локализации точек, чтобы считать множество мелких объектов. Установка,
  предсказание, обучение и экспорт.
lead: >-
  FOMO — это локализатор точек на сетке: каждая ячейка сетки низкого разрешения
  классифицируется как фон или как центр объекта, без всякой регрессии
  ограничивающих рамок. LibreYOLO поддерживает его для задачи point.
keywords:
  - FOMO
  - Faster Objects More Objects
  - локализация точек
  - детекция центроидов
  - детекция мелких объектов
  - edge ai детекция
  - детекция на микроконтроллере
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Веса LibreFOMO не скачиваются автоматически (см. «Чекпойнты» ниже).
        # Укажите здесь путь к чекпойнту, который уже скачан локально.
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz нужно передать явно: в CLI по умолчанию 640, а чекпойнт s

        # принимает только своё родное значение 96.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает загрузчик по суффиксу файла, поэтому экспортированный

        # артефакт загружается как любой чекпойнт и возвращает тот же объект
        Results.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Установка

FOMO не требует ничего сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

В отличие от всех остальных семейств на этом сайте, веса LibreFOMO не
скачиваются автоматически: `LibreYOLO("LibreFOMOs-point.pt")` ищет этот файл на
диске и вызывает `ValueError` с его именем, а не скачивает его с Hugging
Face. Сначала скачайте чекпойнт со [страницы организации LibreYOLO](https://huggingface.co/LibreYOLO)
и загрузите его по локальному пути — или обучите свой (см. «Обучение» ниже).

<code-tabs name="predict" />

В результате вместо `boxes` лежит `points`: каждая строка — это
`x, y, class, confidence`, доступные как `result.points.data` или через
аксессоры `.xy`, `.xyn`, `.cls` и `.conf`. Порог `iou` задавать не нужно,
потому что подавлять нечего — рамок здесь нет; `predict(..., nms_radius=1)`
задаёт, на сколько ячеек сетки должны отстоять друг от друга две детекции,
чтобы уцелели обе, а в имени файла должен быть суффикс задачи FOMO `-point`,
иначе загрузчик его не распознает. Про источники, стриминг и обработку
результатов см. [предсказание](/docs/predict).

## Варианты

Три размера — `s`, `m` и `l` — используют всё более широкие бэкбоны в стиле
MobileNetV2 при соответственно больших фиксированных входных разрешениях, и за
каждым из них стоит одна классификационная голова 1x1. Таблицы бенчмарков у
этого семейства здесь нет; размер файла чекпойнта в таблице ниже — самый
внятный из опубликованных на сегодня признаков разницы между размерами.

## Обучение

<code-tabs name="train" />

`imgsz` выбирается не свободно: по умолчанию берётся родное разрешение
загруженного чекпойнта, а передача другого значения вызывает `ValueError` с
указанием ожидаемого размера. Эти размеры — 96 для `s`, 192 для `m` и 224 для
`l`. В CLI `imgsz` по умолчанию равен 640, поэтому в команде `libreyolo train`
его приходится задавать явно, чтобы он совпал с чекпойнтом.

Если больше ничего не менять, обучение идёт 40 эпох с размером батча 32,
оптимизатором Adam при `lr0=3e-4`, без weight decay и с классом переднего плана,
взвешенным в 100 раз сильнее фона в поячеечной кросс-энтропийной функции
потерь, — потому что в типичной сцене почти каждая ячейка сетки это фон. EMA и
смешанная точность по умолчанию выключены, и ни одна из геометрических или
цветовых аугментаций, применяемых в LibreYOLO в других местах, здесь не
работает: mosaic, mixup, HSV-джиттер, отражение, поворот, сдвиг и наклон — всё
по нулям.

Именно этим путём обучались опубликованные чекпойнты LibreFOMO — с нуля на
COCO.

Про датасеты и логгеры см. [обучение](/docs/train).

## Валидация

`val()` уходит в валидатор уровня сетки, сделанный под это семейство. Помимо
ключей `metrics/precision`, `metrics/recall` и `metrics/mAP@` с сопоставлением
по точкам, общих с другими point-задачами, он перебирает пороги уверенности и
значения `nms_radius` и публикует комбинацию с лучшим F1 под ключами
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall` и
`metrics/grid_mean_distance`, а породившие её порог и радиус — под
`decode/threshold` и `decode/nms_radius`.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запуск графа в чистой среде выполнения, без установленной
LibreYOLO, тоже поддерживается, но тогда препроцессинг и постпроцессинг
придётся писать самостоятельно.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства. Ни один из них не скачивается
автоматически: возьмите нужный файл со связанной страницы Hugging Face и
передайте его локальный путь в `LibreYOLO()`.

<checkpoint-table />

## Лицензирование

<provenance-box>

Ссылаться на исходный репозиторий FOMO не на что: компания Edge Impulse
описывает эту технику в блог-посте и в документации своего продукта, но код
обучения или инференса FOMO не выпускала. Архитектура и обучение здесь —
собственная реализация LibreYOLO по этому опубликованному описанию, а
опубликованные чекпойнты LibreFOMO обучены с нуля на COCO, поэтому и код, и эти
веса — MIT и собственность самой LibreYOLO. Название FOMO и описываемая им
техника остаются за Edge Impulse.

</provenance-box>
