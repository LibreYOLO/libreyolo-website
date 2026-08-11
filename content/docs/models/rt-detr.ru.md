---
title: RT-DETR
families: [rtdetr]
seo_title: "RT-DETR, RT-DETRv2 и RT-DETRv4 в LibreYOLO"
description: "Использование RT-DETR, RT-DETRv2 и RT-DETRv4 в LibreYOLO для детекции объектов, а также повёрнутые рамки в RT-DETRv2. Установка, предсказание, обучение, валидация и экспорт, с весами под Apache-2.0."
lead: "Трансформер для детекции, сделанный под инференс в реальном времени: он декодирует фиксированный набор запросов, а не плотную сетку, поэтому шага NMS в нём нет. LibreYOLO поддерживает три его версии, которые различаются по загружаемому чекпойнту, а версия 2 умеет ещё и повёрнутые рамки."
keywords: [RT-DETR, RT-DETRv2, RT-DETRv4, трансформер для детекции в реальном времени, DETR, детекция объектов python, детекция повёрнутых рамок, OBB, DOTA]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Видео
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Версия зашита в имя файла, а фабрика выбирает загрузчик по
        # чекпойнту, поэтому все три загружаются одинаково.
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # Любой источник, который принимает библиотека: файл, папка, URL,
        # индекс веб-камеры, RTSP-поток или список .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Повёрнутые рамки
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Только версия 2. Суффикс -obb выбирает задачу, а чекпойнт
        # распознаётся как повёрнутый по своим тензорам, поэтому аргумент
        # task не нужен. Эти веса — DOTA v1.0, 15 классов аэросъёмки при 1024 px.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)     # (N, 5): cx, cy, w, h, радианы
        print(obb.xyxyxyxy)  # те же строки в виде четырёх угловых точек
        print(result.boxes.xyxy)  # охватывающие рамки по осям
    - label: Повёрнутые рамки, CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # coco128.yaml при первом запуске скачивает выборку из 128 изображений.
        # Для настоящего запуска укажите в `data` YAML своего датасета.
        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Нужен extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() возвращает обычный словарь, а не объект
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Валидация на COCO
      language: bash
      code: |
        # coco-val-only.yaml скачивает 5000 изображений val2017 и пропускает
        # обучающую часть. В нём есть встроенный скрипт загрузки, поэтому
        # нужно явное разрешение, если датасета ещё нет локально.
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Повёрнутые рамки
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Валидация повёрнутых рамок сопоставляет по повёрнутому IoU, поэтому
        # предсказание в нужном месте, но под неверным углом — это промах.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Нужен extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Повёрнутые рамки
      language: bash
      code: |
        # Для задачи с повёрнутыми рамками проверены ONNX и TorchScript:
        # FP32, батч 1, фиксированный холст 1024 на 1024.
        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024
        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript imgsz=1024
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Установка

RT-DETR не требует опциональных extra. Всё, что он импортирует, входит в базовую
установку, а extra `rtdetr` — стабильное имя, которое ничего к ней не добавляет.

```bash
pip install libreyolo
```

Исключение — дообучение адаптерами с `lora=True`: ему нужен extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена на другой детектор занимает одну строку. `conf` и `max_det` фильтруют
top-k-отбор по запросам и классам; шага NMS, который надо было бы настраивать,
здесь нет, а `iou` принимается, но не используется. Повёрнутый чекпойнт
заполняет `result.obb` напрямую и заодно кладёт в `result.boxes` охватывающие
прямоугольники по осям. Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

## Варианты

Три версии, две задачи на них, и коды размеров не выстраиваются в один ряд.
Версия 1 называет свои размеры по бэкбону — ResNet или HGNetv2. Версия 2
переиспользует только имена ResNet: два размера HGNetv2 уже есть в версии 1, а
результаты версии 2 на них оказались настолько близкими, что LibreYOLO не
публикует для них дублирующих весов. Версия 4 использует обычный буквенный ряд,
который пересекается с именами HGNetv2 из версии 1, поэтому код размера сам по
себе модель не определяет. Версия записана в имя файла чекпойнта.

<benchmark-table task="detect" />

<va-embed />

Версия 2 сохраняет архитектуру и раскладку state dict версии 1 и меняет способ
выборки в deformable attention — поэтому их различают по метаданным в чекпойнте,
а не по формам тензоров. Версия 4 — отдельная линия: она переиспользует
архитектуру и тренер D-FINE, а её веса получены дистилляцией фундаментальной
зрительной модели DINOv3 как учителя в ученика HGNetv2. В LibreYOLO
`LibreRTDETRv4` — подкласс `LibreDFINE` с намертво отключённой головой масок,
поэтому он остаётся только детекцией.

### Повёрнутые рамки в версии 2

Версия 2 — единственная версия, которая несёт вторую задачу. Поддерживаемые
задачи — `detect` и `obb`, и общего графа или общего ряда размеров у них нет.
Детекция использует размеры ResNet при 640 px; детекция повёрнутых рамок — ряд
HGNetv2, n, s, m, l и x, при 1024 px, причём входной размер определяется по
задаче, а не по семейству. Чекпойнт распознаётся как повёрнутый по своим
тензорам — по головам рамок с пятью координатами и параметрам выборки версии 2, —
поэтому веса `-obb` загружаются в повёрнутый граф без аргумента `task`, а
несовпадение между ними даёт жёсткую ошибку, а не тихую переинтерпретацию.

Опубликованные файлы — от `LibreRTDETRv2n-obb.pt` до
`LibreRTDETRv2x-obb.pt`. Это официальные одномасштабные чекпойнты DOTA v1.0,
переведённые в формат LibreYOLO: 15 классов аэросъёмки, от самолёта и корабля до
порта и вертолёта, и имена этих классов зашиты в чекпойнт. В отличие от
детекции, задача с повёрнутыми рамками работает только на инференс:
предсказание, валидация и экспорт работают, а `train()` на модели с повёрнутыми
рамками выбрасывает исключение. Трекинг и аугментация на этапе теста повёрнутые
рамки тоже не поддерживают. [Детекция повёрнутых
рамок](/docs/tasks/oriented-detection) описывает задачу, формат разметки и
метрики.

## Обучение

Обучение начинается с опубликованного чекпойнта. `pretrained` во всех трёх
версиях принимается, а затем отбрасывается, поэтому `pretrained=False` не даёт
случайно инициализированную модель. Всё в этом разделе — про детекцию: задача с
повёрнутыми рамками в версии 2 работает только на инференс, и переноса с весов
детекции на неё нет, потому что бэкбоны у них разные.

<code-tabs name="train" />

Скорость обучения — тот аргумент, который важно выставить правильно, и у каждой
версии здесь своё значение по умолчанию, а не общее для библиотеки.
Python-сигнатура `train()` берёт его из тренировочного конфига этой версии, а
CLI разрешает то же значение, когда `lr0` не передан. Версии 1 и 2 принимают ещё
`lr_backbone` и по умолчанию ставят его в одну двадцатую от `lr0`, следуя
оригинальному рецепту; версия 4 идёт через тренер D-FINE, который вместо этого
масштабирует группу параметров бэкбона через `backbone_lr_mult`.

Оставьте `imgsz` равным родному размеру чекпойнта, если нет причины его менять.
Валидация и предсказание на других размерах работают, с одной оговоркой:
прямоугольный размер, у которого число токенов совпадает с родным, всё равно
переиспользует эмбеддинг, построенный для другого соотношения сторон.

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, куда входят точность, полнота,
mAP 50 и mAP 50-95, посчитанные на любом датасете в том формате, на котором вы
обучали.

<code-tabs name="val" />

Строки в таблице бенчмарков выше получены на бенчмарк-стенде LibreYOLO; в
примечании под этой таблицей записано, какой датасет их дал, и стоят ссылки на
записи о запусках.

Валидация повёрнутых рамок идёт через тот же вызов и сообщает те же ключи плюс
четыре повторённых с суффиксом `(OBB)`. Сопоставление использует повёрнутый IoU,
а не IoU охватывающих прямоугольников, поэтому ошибка в угле — это промах.
`augment=True` для этой задачи отклоняется.

## Экспорт

<export-matrix />

Матрица описывает всю линию как одну страницу: там, где три версии расходятся по
формату, в ячейке стоит самая слабая из трёх, поэтому ничего здесь не приукрашено
для той версии, которую вы загрузите. Строка про повёрнутые рамки относится
только к версии 2. Там проверены ONNX и TorchScript — при FP32, батче 1 и
фиксированном холсте 1024 на 1024; OpenVINO, TensorRT и ExecuTorch конвертируются
и загружаются обратно, но не дотянули до совпадения сырых выходов по всему набору
запросов, так что верхние рамки сходятся с точностью до доли пикселя, а хвост
уплывает.

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

Имя файла несёт версию, затем размер, затем задачу. Веса детекции —
`LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` и `LibreRTDETRv4<size>.pt`, все
при 640 px. Веса для повёрнутых рамок есть только у версии 2 и добавляют суффикс
задачи: от `LibreRTDETRv2n-obb.pt` до `LibreRTDETRv2x-obb.pt`, все при 1024 px и
обучены на DOTA v1.0, а не на COCO.

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />

Блок выше — это то, что авторы публикуют для детекции версий 1 и 2. У весов
версии 2 с повёрнутыми рамками есть третий апстрим — репозиторий RiO-DETR под
Apache-2.0 по адресу
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), откуда
и взяты чекпойнты DOTA; сошлитесь на этот проект, если пользовались таким
чекпойнтом. Версия 4 —
отдельная статья другой группы, и у неё свой блок цитирования по адресу
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation);
сошлитесь на него, если использовали чекпойнт версии 4.
