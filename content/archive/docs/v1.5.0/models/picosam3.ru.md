---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3: сегментация по промпту-рамке на edge-устройствах в LibreYOLO'
description: >-
  Используйте PicoSAM3 в LibreYOLO для сегментации областей по промпту-рамке на
  edge-сенсорах. Установка, предсказание и экспорт чекпойнта pico под лицензией
  Apache-2.0.
lead: >-
  PicoSAM3 — компактная CNN, дистиллированная из SAM 2.1 и SAM 3 и рассчитанная
  на сегментацию области интереса по промпту-рамке на сенсорах вроде Sony
  IMX500. LibreYOLO поддерживает его через отдельную фабрику LibreSAM,
  независимую от фабрики детекторов LibreYOLO(), и только с промптами-рамками.
keywords:
  - PicoSAM3
  - Segment Anything
  - сегментация на edge-устройствах
  - сегментация по рамке
  - область интереса ROI
  - инференс на сенсоре
  - IMX500
  - дистилляция знаний
last_verified: 1.5.0
snippets:
  predict:
    - label: Промпт в виде рамки
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # У PicoSAM3 один размер, "pico", поэтому других алиасов не нужно.
        model = LibreSAM("picosam3")

        # bboxes= — единственный поддерживаемый промпт: [x1, y1, x2, y2] или
        # список рамок, по маске на каждую. Перед запуском CNN каждая рамка
        # расширяется на 10%, приводится к квадрату, обрезается по изображению
        # и масштабируется до 96x96.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
        print(result.masks.xy)      # полигон на каждую маску
        print(result.boxes.xyxy)    # плотная рамка, выведенная из маски
    - label: 'Кодирование один раз, много промптов'
      language: python
      code: |
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE

        model = LibrePicoSAM3()

        # set_image() кэширует исходное изображение; PicoSAM3 делает по одному
        # полному прямому проходу CNN на каждую рамку, так что здесь экономится
        # загрузка и декодирование изображения, а не проход энкодера, как в
        # остальных семействах SAM.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(bboxes=[300, 200, 900, 700])
        b = model.predict(bboxes=[100, 100, 400, 400])
        model.reset_image()
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibrePicoSAM3

        model = LibrePicoSAM3()
        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")

        # opset (по умолчанию 13) и dynamic (по умолчанию True, только ось
        # батча) — единственные аргументы экспорта, которые принимает это
        # семейство.
    - label: Использование экспортированного файла
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3 экспортирует свою сырую ROI-CNN 96x96: roi_image ->

        # mask_logits. Готовой пред/постобработки на стороне LibreYOLO здесь

        # нет, потому что export() не заворачивается обратно в LibreYOLO(),

        # как это происходит с чекпойнтом детектора.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## Установка

PicoSAM3 требует extra-пакет `sam`: скачивание весов в LibreYOLO по-прежнему
идёт через инструменты Hugging Face из `transformers`, хотя сам инференс
работает на нативной CNN без `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Предсказание

`LibreSAM(...)` (или специфичный для семейства `LibrePicoSAM3(...)`) — точка
входа, отдельная от `LibreYOLO(...)`: она возвращает сегментатор, работающий по
промпту, а не детектор, потому что прямой проход здесь без промпта не имеет
смысла. CLI-команды `libreyolo predict` для этого семейства нет; используйте
Python API.

<code-tabs name="predict" />

PicoSAM3 принимает только `bboxes=`; передача `points=`, `labels=`, `masks=`,
`text=`, `multimask=True` или пропуск рамки, чтобы сегментировать всё подряд,
во всех случаях вызывает понятный `ValueError`, потому что ни одного из этих
режимов нет в исходной модели. `conf` фильтрует по предсказанному качеству
маски (IoU), а не по уверенности детекции, и должен быть между `0.0` и `1.0`.
У каждой маски id класса `0` с именем `"object"`. `train()`, `val()` и
`track()` вызывают `NotImplementedError`; для промптов по точкам, тексту, маске
или «сегментировать всё» используйте LibreSAM2 или LibreSAM3. О типах
источников — в разделе [предсказание](/docs/predict).

## Варианты

Один размер, pico, с фиксированным входом ROI 96 px: PicoSAM3 делает по одному
полному прямому проходу CNN на каждую рамку, а не кодирует всё изображение один
раз.

## Экспорт

<export-matrix />

PicoSAM3 — единственное семейство уровня SAM, у которого есть экспорт: оно
выгружает свою сырую ROI-CNN 96x96 в ONNX, `roi_image -> mask_logits`, без
встроенных NMS и постобработки масок. Остальные семейства SAM вызывают
`NotImplementedError` на `export()`, потому что для их разделения на энкодер и
декодер пока нет определённого контракта экспорта под среду выполнения.
Экспортированный граф PicoSAM3 не загружается обратно через `LibreYOLO()`;
запускайте его напрямую в среде выполнения вроде `onnxruntime`, применяя ту же
предобработку с квадратным ROI и отступом 10%, что показана выше.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box>

PicoSAM3 дистиллирован из моделей-учителей SAM 2.1 и SAM 3. LibreYOLO не
встраивает и не распространяет код или веса ни одного из учителей в этом
семействе; поставляются только компактная CNN-ученик и её сконвертированный
чекпойнт.

</provenance-box>

## Цитирование

<citation-block />
