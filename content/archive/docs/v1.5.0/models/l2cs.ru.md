---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net: оценка направления взгляда в LibreYOLO'
description: >-
  Использование L2CS-Net в LibreYOLO для двухэтапной оценки взгляда по углам
  pitch и yaw. Установка, предсказание и экспорт; чекпойнт Gaze360 — только для
  исследований.
lead: >-
  L2CS-Net — двухэтапный оценщик взгляда: детектор лиц находит лица, а ствол
  ResNet с двумя классификационными головами по угловым бинам предсказывает
  pitch и yaw для каждого лица. LibreYOLO поддерживает его только для инференса.
keywords:
  - L2CS-Net
  - оценка направления взгляда
  - gaze estimation python
  - айтрекинг по камере
  - pitch yaw взгляд
  - Gaze360
  - детекция лиц python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # face_detector не задан: используется штатный детектор лиц из
        # OpenCV (Haar в OpenCV 4, YuNet в OpenCV 5), поэтому кроме самого
        # чекпойнта L2CS ничего скачивать не нужно.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Источник лиц
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Передать L2CS рамки от детектора, который вы уже запустили.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Или указать конкретный встроенный детектор лиц.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Использование экспортированного файла
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # В экспортированном графе только ствол ResNet и две головы по
        # угловым бинам: он принимает предобработанный кроп лица 448x448 и
        # возвращает сырые (yaw_logits, pitch_logits), а не готовые углы.
        # Softmax, матожидание по бинам и перевод в градусы остаются в
        # Python; см. libreyolo.models.l2cs.utils.bin_logits_to_angles.
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## Установка

Чтобы создать модель, запустить на ней предсказание или экспортировать её,
когда чекпойнт уже есть, L2CS-Net не нужны опциональные extra.

```bash
pip install libreyolo
```

Единственный чекпойнт, который LibreYOLO может скачать автоматически, —
ResNet-50, обученный на Gaze360, — скачивается через `gdown`, а не с обычного
HTTP-зеркала, потому что лежит на Google Drive автора, а не в организации
LibreYOLO. Для этого пути нужен extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Без него LibreYOLO печатает инструкции по ручному скачиванию, а не молча
падает.

## Предсказание

<code-tabs name="predict" />

L2CS-Net — двухэтапный оценщик: сначала работает детектор лиц, а голова взгляда
считывает pitch и yaw с каждого кропа лица, который тот вернул. Если ничего не
указать, предсказание откатывается к штатному детектору OpenCV, поэтому голый
вызов работает без дополнительных скачиваний, как только на руках есть сам
чекпойнт L2CS. `face_boxes` принимает рамки от детектора, который вы уже
запустили; `face_detector` принимает `"auto"`, `"haar"`, `"yunet"`, модель
детекции LibreYOLO или обычный вызываемый объект. `result.gaze` содержит pitch
и yaw в радианах, построчно выровненные с `result.boxes` — найденными рамками
лиц. Про источники, стриминг и работу с результатом см.
[предсказание](/docs/predict).

## Варианты

Пять глубин бэкбона используют одно входное разрешение и принимают одни и те же
аргументы. На Gaze360, датасете за единственным опубликованным чекпойнтом,
обучали ResNet-50; остальные четыре глубины поддержаны архитектурно, но
опубликованных весов для загрузки у них нет.

## Экспорт

<export-matrix />

<code-tabs name="export" />

## Лицензирование

<provenance-box>

LibreYOLO не размещает и не зеркалит ни один чекпойнт L2CS: в организации
LibreYOLO на Hugging Face для этого семейства нет ничего, в отличие от
большинства других семейств на этом сайте. Единственный чекпойнт, который
библиотека может скачать автоматически, берётся напрямую из дистрибутива на
Google Drive самого автора, за уведомлением о лицензии Gaze360, которое
печатается перед началом передачи, и это не та копия, «переопубликованная на
huggingface.co/LibreYOLO», о которой говорит сводка выше.

</provenance-box>
