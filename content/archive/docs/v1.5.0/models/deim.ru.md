---
title: DEIM
families:
  - deim
seo_title: DEIM и DEIMv2 в LibreYOLO
description: >-
  Используйте DEIM и DEIMv2 в LibreYOLO для детекции объектов. Установка,
  предсказание, обучение, валидация и экспорт — начиная с размера в полмиллиона
  параметров.
lead: >-
  Детекционный трансформер, обученный плотным взаимно однозначным
  сопоставлением: он сходится за куда меньшее число эпох, чем рецепты DETR, на
  которых он построен. LibreYOLO несёт две его версии, а различают их по
  загружаемому чекпойнту.
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - детекция объектов python
  - DETR
  - трансформер для детекции объектов
  - детекция в реальном времени
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Видео
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Версия входит в имя файла, а фабрика выбирает по чекпойнту, поэтому
        # обе загружаются одинаково.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # Любой источник, который принимает библиотека: файл, папка, URL, индекс
        # веб-камеры, RTSP-поток или список .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml при первом запуске скачивает выборку из 128 изображений.
        # Для реального запуска укажите в `data` YAML своего датасета.
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Если epochs, batch, imgsz и lr0 не заданы, они берутся из
        # опубликованного рецепта для загруженного размера.
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Требуется extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() возвращает обычный dict, а не объект
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: Валидация на COCO
      language: bash
      code: |
        # coco-val-only.yaml скачивает 5000 изображений val2017 и пропускает
        # обучающий набор. Внутри него есть встроенный скрипт загрузки, поэтому
        # нужно явное разрешение, если датасета ещё нет локально.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Требуется extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает по суффиксу файла, поэтому экспортированный артефакт
        # загружается как обычный чекпойнт и возвращает тот же объект Results.
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## Установка

Ни одной из версий не нужен дополнительный extra. Всё, что они импортируют,
входит в базовую установку.

```bash
pip install libreyolo
```

Исключение — дообучение адаптерами через `lora=True`: ему нужен extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
подменить детектор на другой можно одной строкой. `conf` и `max_det` фильтруют
top-k-декодирование по запросам и классам; шага NMS, который нужно было бы
настраивать, здесь нет, а `iou` принимается, но не используется. Про источники,
стриминг и обработку результатов — [предсказание](/docs/predict).

## Варианты

Версия 1 поставляется в пяти размерах, все с одинаковым входным разрешением.
Версия 2 сохраняет эти пять названий и добавляет три меньших — `atto`, `femto`
и `pico`, — первые два из которых нативно работают на меньшем входном
разрешении, чем остальные. Поэтому пять кодов размеров существуют в обеих
версиях и обозначают разные модели; версия записана в имя файла чекпойнта.

<benchmark-table task="detect" />

<va-embed />

Версия 1 сохраняет архитектуру D-FINE и заменяет её целевую функцию
классификации на функцию потерь с учётом сопоставимости из рецепта плотного
взаимно однозначного сопоставления, поэтому у двух семейств совпадают почти все
ключи state dict, и различают их по метаданным в чекпойнте. Версия 2 сохраняет
этот контракт обучения и смешивает бэкбоны: HGNetv2 ниже `s` и vision
transformer DINOv3 с адаптером пространственной подстройки на `s` и выше.
Именно из-за этого бэкбона на этих четырёх чекпойнтах появляется вторая
лицензия, так что прежде чем выкатывать такой чекпойнт, прочитайте
[лицензирование](#licensing).

## Обучение

Обучение начинается с опубликованного чекпойнта. `pretrained` до обучения не
доходит: версия 1 предупреждает, что ключ неизвестен, и игнорирует его, версия
2 его удаляет. Ни та, ни другая не дадут модель со случайной инициализацией.

<code-tabs name="train" />

На версии 1 задавайте `lr0` сами. В сигнатуре Python-метода `train()` по
умолчанию стоит `4e-4` — скорость обучения из опубликованного рецепта для COCO,
— тогда как в конфиге обучения этого семейства значением по умолчанию для
дообучения записано `1e-4`, и именно это меньшее значение подставляет CLI, если
аргумент не передан. В конфиге записано и измерение, которое за этим стоит: при
размерах батча, которые реально используются при дообучении, на небольших
датасетах скорость обучения из рецепта COCO заметно ухудшала перенос.

Версия 2 разрешает эти значения по умолчанию сама. Если `epochs`, `batch`,
`imgsz` и `lr0` не заданы, каждое из них читается из опубликованного рецепта
для загруженного размера, поэтому маленькие размеры обучаются на своём входном
разрешении без дополнительных указаний, а переданное вами значение перекрывает
рецепт. Ограничение накладывается на `imgsz`: он должен быть положительным
числом, кратным 32, иначе версия 2 выбросит ошибку ещё до старта запуска.

Про датасеты, аугментацию, обучение на нескольких GPU и логгеры —
[обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, которые покрывают точность,
полноту, mAP 50 и mAP 50-95, посчитанные на любом датасете в том формате, на
котором вы обучались.

<code-tabs name="val" />

Строки в таблице бенчмарков выше получены на бенчмарк-стенде LibreYOLO; в
примечании под этой таблицей указано, на каком датасете они получены, и даны
ссылки на записи о запусках.

## Экспорт

<export-matrix />

Матрица описывает обе версии на одной странице: там, где они расходятся по
формату, в ячейке показан более слабый из двух вариантов, так что здесь ничего
не приукрашено — какую бы версию вы ни загрузили.

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>
Четыре размера DEIMv2 начиная с S берут бэкбон из DINOv3, поэтому в их
репозиториях с весами действуют одновременно Apache-2.0 и DINOv3 License от
Meta, а исходники бэкбона DINOv3 LibreYOLO поставляет по тому же соглашению.
Остальная часть семейства, включая все размеры DEIMv2 ниже S, — только
Apache-2.0.
</provenance-box>

## Цитирование

<citation-block />

DEIMv2 — отдельная статья, и у неё свой блок цитирования на
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation);
ссылайтесь на него, если использовали чекпойнт версии 2.
