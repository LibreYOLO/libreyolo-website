---
title: YOLOv9
families:
  - yolo9
seo_title: 'YOLOv9: предсказание, обучение и экспорт под MIT'
description: >-
  Запуск YOLOv9 в LibreYOLO, включая end-to-end голову без NMS и голову с шагом
  4 для мелких объектов. Установка, предсказание, обучение, валидация и экспорт.
lead: >-
  Одностадийный свёрточный детектор: за один проход оценивается плотная сетка
  рамок, а NMS отбрасывает дубликаты. В LibreYOLO есть три его варианта, и у
  одного из них шага NMS нет.
keywords:
  - YOLOv9
  - YOLO9
  - детекция объектов python
  - детекция без nms
  - end-to-end детекция
  - детекция мелких объектов
  - programmable gradient information
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Без NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Тот же вызов, другой чекпойнт. End-to-end голова сама возвращает

        # предсказания с наибольшими оценками, поэтому NMS не запускается, а iou
        игнорируется.

        model = LibreYOLO("LibreYOLO9E2Es.pt")

        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)


        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Мелкие объекты
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # У варианта с шагом 4 нет собственного чекпойнта на COCO, поэтому

        # укажите базовый детекционный: его бэкбон и neck загрузятся без

        # изменений, а башня головы с шагом 4 стартует со случайной
        инициализации.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Проверка на COCO
      language: bash
      code: |
        # Во встроенном COCO yaml есть скрипт скачивания, поэтому нужно явное
        # разрешение, если датасет ещё не лежит локально.
        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: С NMS в графе
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Использование экспортированного файла
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика выбирает путь по суффиксу файла, поэтому экспортированный

        # артефакт загружается как любой чекпойнт и возвращает тот же объект
        Results.

        model = LibreYOLO("LibreYOLO9s.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## Установка

YOLOv9 не требует ничего сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена на другой детектор — правка в одну строку. У базовой модели и у модели с
шагом 4 `conf` задаёт порог уверенности, а `iou` — порог NMS. End-to-end модель
NMS не запускает и `iou` игнорирует, поэтому её вывод определяется `conf` и
`max_det`. Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

## Варианты

Три варианта используют общий бэкбон. Все три решают только задачу детекции и
принимают одни и те же аргументы.

Базовая модель предсказывает на трёх масштабах признаков и убирает
дублирующиеся рамки с помощью NMS.

End-to-end модель сохраняет эту голову и добавляет рядом ветку с сопоставлением
один к одному. Инференс читает только эту ветку и берёт из неё предсказания с
наибольшими оценками, поэтому NMS не запускается. Выбирайте её, когда в среде
выполнения, куда вы разворачиваете модель, нет оператора NMS.

Модель с шагом 4 выводит наружу ещё один уровень бэкбона, протягивает до него
neck и предсказывает на четырёх масштабах вместо трёх. Дополнительный масштаб
нужен для объектов, занимающих мало пикселей; единственный опубликованный для
неё чекпойнт обучен на аэрофотоснимках. Базовые детекционные чекпойнты в неё
переносятся: бэкбон и neck загружаются без изменений, три предобученные башни
головы сдвигаются на одну позицию вверх, а башня с шагом 4 стартует со
случайной инициализации.

<benchmark-table task="detect" />

<va-embed />

## Обучение

<code-tabs name="train" />

`pretrained` определяет, с чего начинается запуск. Передайте `True`, чтобы
загрузить опубликованный чекпойнт для той же модели и того же размера, или имя
либо путь — для всего остального. Тензоры с несовпадающей формой пропускаются,
а не приводят к ошибке, и в лог пишется, сколько их загрузилось, поэтому
чекпойнт, обученный на другом числе классов, всё равно годится как стартовая
точка.

У модели с шагом 4 нет собственного опубликованного чекпойнта на COCO, поэтому
в этом случае `True` указывает на несуществующий файл, и скачивание падает.
Вместо этого укажите базовый детекционный чекпойнт.

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/` для точности, полноты, mAP 50 и
mAP 50-95, измеренных на любом датасете в том формате, на котором вы обучались.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Галочка действует для всех трёх вариантов: там, где они расходятся, в таблице
стоит самый слабый из трёх.

Экспортированный артефакт загружается обратно через `LibreYOLO()` по суффиксу
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запуск графа в чистой среде выполнения, без установленной
библиотеки LibreYOLO, тоже поддерживается, но тогда препроцессинг и
постпроцессинг придётся писать самостоятельно.

Для базовой детекционной модели половину с постпроцессингом можно перенести в
граф. `nms=True` при экспорте в ONNX помещает подавление внутрь модели, и
первый выход становится тензором фиксированной формы `(1, max_det, 6)`, строки
которого — `x1, y1, x2, y2, score, class`, дополненные нулями после числа
детекций. Такой граф рассчитан на батч 1 и не содержит динамических осей.
End-to-end модель и модель с шагом 4 этот флаг не принимают.

Каждый формат устанавливает свой extra и принимает несколько собственных
аргументов.
И то и другое описано на странице этого формата.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

Один чекпойнт здесь не под MIT. Модель с шагом 4, обученная на
VisDrone2019-DET, наследует условия CC BY-NC-SA 3.0 этого датасета: только
некоммерческое использование, share-alike для всего производного от неё — и всё
это за пределами разрешительной лицензии, под которой поставляется остальное
семейство. Она предсказывает классы VisDrone для аэросъёмки, а не классы COCO.
Библиотека выводит всё это перед тем, как скачать файл.

</provenance-box>

## Цитирование

<citation-block />
