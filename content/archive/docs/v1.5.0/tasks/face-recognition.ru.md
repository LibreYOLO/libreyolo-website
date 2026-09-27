---
title: Распознавание лиц
seo_title: Распознавание лиц в LibreYOLO
description: >-
  Детекция, эмбеддинги и идентификация лиц в LibreYOLO. Регистрация галереи,
  сравнение двух изображений и сопоставление по косинусному сходству — из Python
  или CLI.
lead: >-
  Распознавание лиц — задача embed, применённая к лицам. Детектор находит и
  выравнивает каждое лицо, голова распознавания возвращает L2-нормированный
  вектор на лицо, а личность определяется по косинусному сходству с
  зарегистрированными эталонами, а не по фиксированному списку классов.
keywords:
  - распознавание лиц python
  - эмбеддинги лиц
  - верификация лиц
  - сравнить два лица
  - идентификация людей на фото
  - arcface onnx
  - косинусное сходство лиц
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Имена librefacerec-* независимо от суффикса файла ведут в семейство
        # эмбеддингов лиц, а веса при первом использовании скачиваются из
        # организации LibreYOLO на Hugging Face вместе с детектором лиц
        # по умолчанию.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # (N, 4) рамки лиц
        print(result.embeddings.data.shape)  # (N, D), по строке на лицо
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Сравнение двух изображений
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("librefacerec-l.onnx")


        # Запускает детекцию и построение эмбеддингов на обоих изображениях

        # и сравнивает самое уверенное лицо. Косинусное сходство лежит в [-1,
        1].

        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)

        print(outcome["similarity"], outcome["same_person"])
    - label: Регистрация галереи и идентификация
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # ниже порога name равно None
    - label: Регистрация и идентификация из CLI
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: Свои рамки лиц
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes полностью пропускает детекцию; face_detector принимает
        # вызываемый объект, детектор LibreYOLO или экземпляр FaceDetector.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## Определение

Распознавание лиц возвращает вектор на каждое лицо, а не метку. Предсказание
идёт в два этапа: детектор лиц находит каждое лицо и пять его ключевых точек,
вырезанный фрагмент приводится к каноническому выравниванию 112x112, а голова
распознавания выдаёт L2-нормированный эмбеддинг.

`result.embeddings` — полезная нагрузка `Embeddings` формы `(N, D)`, построчно
соответствующая `result.boxes`, поэтому строка `i` описывает лицо в рамке `i`.
Поскольку строки — единичные векторы, косинусное сходство сводится к скалярному
произведению, а `embeddings.similarity()` считает его с другим `Embeddings` или
с целой матрицей за один вызов.

Назвать лицо — отдельный шаг. `Gallery` хранит именованные эталонные векторы;
передача `gallery=` в `predict()` добавляет `result.identities` — построчно
соответствующие эмбеддингам имя и лучшую косинусную оценку для каждого лица.
Если лицо не дотянуло до порога совпадения, его именем остаётся `None`, и
ближайшее из имён, которые порог не прошли, никогда не подставляется.

Канонический ключ задачи в библиотеке — `embed`. `face-recognition`,
`facial-recognition`, `reid` и `face` нормализуются к нему, поэтому
`task="face-recognition"` и `task="embed"` выбирают одно и то же. Лица — это
работа с областями изображения внутри более широкой задачи;
[эмбеддинги](/docs/tasks/embeddings) охватывают варианты для изображения целиком
и для текста, общий API `Embeddings`, `Identities` и `Gallery`, а также модели,
которые выдают векторы, ничего не детектируя.

## Модели

[LibreFaceRec](/docs/models/librefacerec) — семейство для этой задачи. За одним
вызовом стоят два ONNX-артефакта: `librefacerec-l.onnx` — голова распознавания
на iResNet100, выдающая 512-мерные эмбеддинги, и `librefacerec-det.onnx` —
детектор лиц по умолчанию с пятью ключевыми точками, взятый из каталога моделей
OpenCV. Оба скачиваются из организации LibreYOLO на Hugging Face при первом
использовании. Любой другой ONNX-файл, следующий соглашению ArcFace (на входе
выровненные 112x112, на выходе `(N, D)`), может заменить голову распознавания —
достаточно передать путь к нему вместо имени `librefacerec-*`.

Ключ задачи `embed` шире, чем лица. [CLIP](/docs/models/clip),
[SigLIP2](/docs/models/siglip2) и [DINOv2](/docs/models/dinov2) тоже
поддерживают `task="embed"` и возвращают один вектор на всё изображение, а это
уже поиск похожих изображений, а не установление личности. У них общий API
`Gallery` и `Embeddings`, поэтому описанный ниже сценарий с регистрацией и
сопоставлением переносится и на них, но лица они не детектируют и не
выравнивают.

Голова распознавания работает через `onnxruntime`, которого нет в базовой
установке:

```bash
pip install "libreyolo[onnx]"
```

## Предсказание

<code-tabs name="predict" />

Если ничего не указывать, `predict()` сам скачивает детектор по умолчанию и
подключает его в пару к голове распознавания. `face_detector` заменяет его
вызываемым объектом, детектором LibreYOLO или экземпляром `FaceDetector`; задать
его можно в конструкторе или при каждом вызове. `face_boxes` обходит детекцию,
если рамки у вас уже есть. В CLI `face_detector=` принимает путь к `.onnx`
детектора лиц или имя детектора LibreYOLO.

`model.verify(image_a, image_b)` — короткий путь для двух изображений: строит
эмбеддинг самого уверенного лица в каждом и возвращает
`{"similarity", "same_person", "threshold"}`. `model.embed(sources)` возвращает
все строки лиц по одному или нескольким изображениям, сложенные в один тензор
`(N_total, D)`. Про источники, стриминг и работу с результатами —
[предсказание](/docs/predict).

## Формат датасета

При регистрации читается по одной папке на личность. Имя папки становится именем
личности, а каждое изображение внутри даёт эталоны для этого имени:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` обходит это дерево и записывает галерею `.npz`. Существующий
файл галереи дополняется на месте, а не заменяется, поэтому личности можно
добавлять со временем. Галереи привязаны к весам, которыми они получены, по
размерности эмбеддинга и отпечатку файла; сопоставление с другой моделью
вызывает ошибку, а не сравнивает несовместимые векторные пространства.

По умолчанию каждое исходное изображение даёт одну эталонную строку — самое
уверенное лицо, поэтому с портрета, на который попали посторонние, регистрируется
только сам портретируемый. Чтобы сохранить все возвращённые строки, передайте
`select="all"` в `Gallery.enroll`.

## Обучение

Ни одно семейство в этой задаче не обучается внутри LibreYOLO.
`LibreFaceEmbedder.train()` вызывает ошибку: обучите голову распознавания на
стороне, экспортируйте её в ONNX по соглашению ArcFace и загрузите файл по
пути.

## Валидация

Валидатора датасета для этой задачи нет, и `val()` вызывает ошибку, а не делает
вид, что он есть. Accuracy верификации измеряют на размеченных парах
изображений через `model.verify()`, перебирая `threshold`, чтобы выбрать нужную
рабочую точку. Accuracy идентификации измеряют так: регистрируют галерею и
читают `result.identities.name` и `result.identities.score` на отложенных
изображениях, считая имя `None` отказом.

## Экспорт

Голова распознавания — уже готовый ONNX-граф, так что конвертировать нечего:
`LibreFaceEmbedder.export()` вызывает ошибку. Разворачивайте файл `.onnx`
напрямую или передайте путь к нему LibreYOLO — и пусть семейство само возьмёт на
себя детекцию, выравнивание и нормализацию.
