---
title: LibreFaceRec
families:
  - facerec
seo_title: 'LibreFaceRec: распознавание и верификация лиц'
description: >-
  Используйте LibreFaceRec в LibreYOLO для детекции лиц, получения эмбеддингов и
  верификации. Установка и предсказание; веса модели эмбеддингов под Apache-2.0.
lead: >-
  LibreFaceRec — задача эмбеддингов лиц в LibreYOLO: детектор лиц находит и
  выравнивает лица, а голова распознавания выдаёт L2-нормированный эмбеддинг
  личности для верификации или поиска.
keywords:
  - LibreFaceRec
  - распознавание лиц python
  - эмбеддинги лиц
  - верификация лиц
  - сравнение лиц onnx
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Имена librefacerec-* ведут в это семейство независимо от
        # суффикса файла и при первом использовании скачиваются из
        # организации LibreYOLO на Hugging Face вместе с детектором лиц
        # по умолчанию.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D), L2-нормированные
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Верификация
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Сравнивает самое заметное лицо на каждом изображении по
        # косинусной близости их L2-нормированных эмбеддингов.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: Поиск по галерее
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        query = model("query.jpg").embeddings          # лица с этого
        изображения

        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)


        # Косинусные близости (query_faces, N_total).

        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## Установка

Голова распознавания LibreFaceRec работает через `onnxruntime`, который не
входит в базовую установку.

```bash
pip install "libreyolo[onnx]"
```

## Предсказание

<code-tabs name="predict" />

Детекция и распознавание — два отдельных ONNX-графа за одним вызовом:
детектор лиц находит каждое лицо и выравнивает его в канонический кроп, а
голова распознавания возвращает по одному L2-нормированному эмбеддингу на
лицо. Если ничего не указывать, `predict()` сам скачивает и подключает
встроенный детектор по умолчанию. `face_detector` принимает вызываемый
объект, модель детекции LibreYOLO или экземпляр `FaceDetector`; `face_boxes`
полностью обходит детекцию и берёт рамки, которые у вас уже есть.
`result.embeddings` содержит по одной строке на каждое обнаруженное лицо, в
том же порядке, что и `result.boxes`; его метод `.similarity()` за один
вызов считает косинусную близость с другим эмбеддингом или с целой
галереей. Чтобы сравнить напрямую два изображения, а не два уже посчитанных
эмбеддинга, вызовите `model.verify(image_a, image_b)`: он прогоняет детекцию
и эмбеддинг для обоих и сравнивает лицо с наибольшей уверенностью. Любую
другую ONNX-модель распознавания в соглашении ArcFace (на входе выровненный
кроп, на выходе эмбеддинги `(N, D)`) можно подставить, передав путь к её
файлу вместо имени `librefacerec-*`. Про источники, стриминг и обработку
результатов см. [предсказание](/docs/predict).

## Экспорт

<export-matrix />

LibreFaceRec и так оборачивает заранее экспортированный ONNX-граф; повторный
экспорт в другой формат не реализован.

## Лицензирование

<provenance-box>

Встроенный детектор лиц по умолчанию — второй артефакт под второй лицензией:
YuNet из OpenCV Zoo, MIT, правообладатель Shiqi Yu. Код архитектуры не
портирован ни из одного из этих проектов; оба графа используются как чёрный
ящик через `onnxruntime`, поэтому собственная обёртка LibreYOLO не содержит
стороннего кода и целиком под MIT.

</provenance-box>

## Цитирование

<citation-block />
