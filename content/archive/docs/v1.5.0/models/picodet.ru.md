---
title: PicoDet
families:
  - picodet
seo_title: 'PicoDet в LibreYOLO: предсказание, обучение и экспорт'
description: >-
  Использование PicoDet в LibreYOLO для детекции объектов на мобильных
  устройствах. Установка, предсказание, обучение, валидация и экспорт под
  лицензией Apache-2.0.
lead: >-
  PicoDet — одностадийный детектор, сделанный под мобильные и периферийные CPU:
  бэкбон ESNet, neck CSP-PAN и общая голова с Generalized Focal Loss. LibreYOLO
  поддерживает его для детекции.
keywords:
  - PicoDet
  - PP-PicoDet
  - детекция объектов python
  - лёгкий детектор для CPU
  - детекция объектов на мобильных устройствах
  - обучить picodet на своём датасете
  - экспорт picodet в onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz стоит задать: в CLI по умолчанию стоит 640, тогда как у

        # чекпойнта s родное разрешение — 320.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: Использование экспортированного файла
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика выбирает загрузчик по расширению файла, поэтому
        # экспортированный артефакт загружается как любой чекпойнт и
        # возвращает тот же объект Results.
        model = LibreYOLO("LibrePICODETs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## Установка

PicoDet не требует ничего сверх базового пакета.

```bash
pip install libreyolo
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

Возвращаемый объект `Results` — тот же, что возвращает любое семейство, поэтому
замена на другой детектор занимает одну строку. `conf` задаёт порог уверенности,
а `iou` — порог NMS. Про источники, стриминг и обработку результатов см.
[предсказание](/docs/predict).

## Варианты

Три размера, каждый на своём фиксированном входном разрешении: `s` — самый
маленький, `l` — самый большой. Разрешение растёт вместе с размером, поэтому
крупные чекпойнты не только несут больше параметров, но и обходятся дороже на
каждом изображении.

<benchmark-table task="detect" />

<va-embed />

## Обучение

<code-tabs name="train" />

Компоненты функции потерь и назначение целей повторяют апстрим-рецепт: VFL, DFL,
GIoU и SimOTA, со взвешиванием по качеству классификации и целями VFL по
динамическому IoU. Инференс побитово совпадает с апстримом на одном и том же
чекпойнте.

Что не проверялось, если верить самому docstring у `train()`: сходимость на
полном датасете, поведение при multi-GPU и любая аугментация, кроме
горизонтального отражения. Чекпойнт `s` на своём родном 320 к тому же не всегда
проходит нижний порог точности LibreYOLO на тестовой выборке из 30 изображений и
двух классов, на которой библиотека проверяет небольшие дообучения. Этот размер
лучше подходит для масштаба полного COCO.

`train()` принимает и аргумент `pretrained`, но внутри метода его значение
нигде не читается: обучение всегда продолжается с тех весов, с которыми модель
была создана, поэтому `pretrained=False` не переинициализирует сеть. Если не
задавать `imgsz` в Python, берётся родное разрешение загруженного чекпойнта:
320 для `s`, 416 для `m` и 640 для `l`. CLI всегда передаёт `imgsz`, по
умолчанию 640, поэтому там его нужно выставить под чекпойнт.

Если больше ничего не трогать, обучение идёт 300 эпох на SGD с `lr0=0.01`,
моментом 0.9, weight decay 4e-5 и прогревом в одну эпоху по косинусному
расписанию. Единственная применяемая аугментация — горизонтальное отражение.

Про датасеты, аугментацию, multi-GPU и логгеры см. [обучение](/docs/train).

## Валидация

`val()` возвращает словарь с ключами `metrics/`, которые покрывают точность,
полноту, mAP 50 и mAP 50-95, посчитанные на любом датасете в том формате, на
котором шло обучение.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспортированный артефакт загружается обратно через `LibreYOLO()` по расширению
файла, поэтому файл `.onnx` или `.engine` ведёт себя как чекпойнт и возвращает
тот же `Results`. Запускать граф в голой среде выполнения, без установленного
LibreYOLO, тоже можно, но тогда предобработку и постобработку придётся писать
самостоятельно.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

Порт в LibreYOLO следует за Bo396543018/Picodet_Pytorch — переписанной на
PyTorch реализацией оригинального PP-PicoDet из PaddleDetection, из которой
убран mmcv и в которой все активации сведены один в один, так что чекпойнты
PaddlePaddle, сконвертированные через пайплайн Bo, загружаются без числового
расхождения. Оба источника распространяются на тех же условиях Apache-2.0, что
и у авторов статьи.

</provenance-box>

## Цитирование

<citation-block />
