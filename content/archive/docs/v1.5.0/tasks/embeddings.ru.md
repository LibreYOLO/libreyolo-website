---
title: Эмбеддинги
seo_title: Эмбеддинги изображений и областей в LibreYOLO
description: >-
  Задача embed возвращает L2-нормализованные векторы float32 для всего
  изображения, для каждой найденной области или для текста. Зарегистрируйте
  галерею, сопоставляйте по косинусной близости и ищите из Python или CLI.
lead: >-
  Одна задача покрывает все векторы, которые производит LibreYOLO. embed
  возвращает строки float32 единичной длины, скалярное произведение которых и
  есть оценка близости, независимо от того, описывает ли строка целое
  изображение, одно найденное лицо или строку текста, и все их сопоставляет один
  и тот же Gallery.
keywords:
  - эмбеддинги изображений python
  - l2 нормализация эмбеддингов
  - поиск по косинусной близости
  - libreyolo embed
  - поиск похожих изображений
  - регистрация галереи
  - clip эмбеддинги
  - dinov2 эмбеддинги
  - reid эмбеддинги
last_verified: 1.5.0
verification: >-
  Ключ задачи и псевдонимы взяты из libreyolo/tasks.py. Данные результата — из
  классов Embeddings и Identities в libreyolo/utils/results.py. API галереи — из
  libreyolo/utils/gallery.py. embed и _postprocess_embeddings — из
  libreyolo/models/base/model.py. Поддерживаемые семейства найдены поиском embed
  в SUPPORTED_TASKS по libreyolo/models/**/model.py. Набор команд CLI — из
  libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py и
  libreyolo/cli/commands/predict.py. Замысел разработки — из
  docs/adr/0015-embed-generalization.md.
meta:
  - label: Ключ задачи
    value: embed
    mono: true
  - label: Псевдонимы
    value: 'face-recognition, reid, face'
    mono: true
  - label: Данные результата
    value: 'Embeddings, Identities'
    mono: true
  - label: Тип данных строки
    value: 'float32, единичная длина'
snippets:
  predict:
    - label: Всё изображение
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # По умолчанию у CLIP задача classify, поэтому вектор нужно запросить
        явно.

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        result = model(SAMPLE_IMAGE)


        print(result.embeddings.data.shape)  # (1, 512), по одной строке на
        изображение

        print(result.boxes)                  # None: ничего не локализовано
    - label: По областям
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # Строка i описывает область из рамки i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Несколько изображений сразу
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Все строки всех результатов, объединённые в один тензор.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Текст
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # Для текста есть отдельный метод; источником предсказания он не бывает.
        # Строка, переданная в model(...), — это по-прежнему путь или URL.
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: Сравнение двух наборов строк
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")


        query = model.embed("query.jpg")          # (1, 512)

        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)


        # Строки единичной длины, поэтому косинусная близость — скалярное
        произведение.

        scores = model("query.jpg").embeddings.similarity(pool)

        print(scores.shape)  # (1, 2)
    - label: Сопоставление изображения с текстом
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Регистрация и идентификация
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # ниже порога name равно None
    - label: Поиск top-k
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(name, score), ...] для первой строки
    - label: Регистрация готового вектора
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # нормализуется при добавлении
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Регистрация дерева папок
      language: bash
      code: >
        # source/<identity>/*.jpg. Существующая галерея дополняется на месте.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: Идентификация во время предсказания
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Сравнение двух изображений
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify — та же команда под вторым именем.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## Определение

`embed` превращает изображение, область изображения или строку текста в строку
float32 фиксированной ширины, длина которой равна единице. Каждая строка —
единичный вектор, поэтому сравнение двух строк сводится к скалярному
произведению, а сравнение двух наборов строк — к одному матричному умножению.
Больше ничто в этой задаче не зависит от модели: поиск похожих, поиск дубликатов,
повторная идентификация и распознавание лиц — это одна и та же арифметика над
разными строками.

Вектор и есть результат. Списка классов нет, поэтому имя присваивается позже —
сравнением с эталонами, которые вы предоставляете, а не тем, что сеть обучили
предсказывать.

### Три формы

| Форма | `Results.embeddings` | `Results.boxes` | Что выдаёт |
|---|---|---|---|
| Всё изображение | `(1, D)` | `None` | Передача изображения семейству, работающему с изображением целиком |
| Область | `(N, D)` | `(N, 4)`, построчно согласованные | Семейства, которые сначала локализуют, например распознавание лиц |
| Текст | вообще не `Results` | | `model.embed_text(texts)`, возвращает `(M, D)` |

Результат для целого изображения остаётся двумерным даже для одного изображения.
`(D,)` — недопустимая форма возврата, поэтому потребителю никогда не приходится
отдельно обрабатывать случай одной строки. Текст возвращает обычный тензор, а не
`Results`, потому что строка — не источник изображений: если передать её
в `model(...)`, это по-прежнему путь или URL, и библиотека никогда не предполагает,
что строка — это текст.

Канонический ключ задачи — `embed`. `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` и
`reid` нормализуются к нему, поэтому `task="reid"` и `task="embed"` выбирают
ровно одно и то же.

## Модели

Задачу обслуживают четыре семейства, и они чётко делятся по тому, локализуют ли
сначала хоть что-нибудь.

| Семейство | Форма | Размерность | Что ещё поддерживает |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Область, по одной строке на каждое найденное лицо | 512 | Ничего; `embed` — его единственная задача |
| [CLIP](/docs/models/clip) | Всё изображение, с парной текстовой башней | 512 для `b32` и `b16`, 768 для `l14` | `classify`, который остаётся задачей по умолчанию |
| [SigLIP 2](/docs/models/siglip2) | Всё изображение, с парной текстовой башней | 768 для `b16`, 1152 для `so400m` | `classify`, который остаётся задачей по умолчанию |
| [DINOv2](/docs/models/dinov2) | Всё изображение, только изображение | 384 | `semantic`, `classify` |

CLIP и SigLIP 2 сохраняют `classify` как задачу по умолчанию, поэтому `task="embed"`
нужно запрашивать явно. Их существующий чекпойнт `-cls` — это общий двухбашенный
артефакт; отдельный чекпойнт `-embed` для тех же самых весов не публикуется.

`embed_text` есть только у CLIP и SigLIP 2 — двух семейств с текстовой башней. У
DINOv2 её нет. Эмбеддинг в DINOv2 обходит семантическую и классификационную
головы и читает финальный нормализованный CLS-токен при разрешении 224 пикселя; варианты
`n`, `s`, `m` и `l` используют один и тот же энкодер DINOv2-S, поэтому все четыре
возвращают `D = 384`.

Добавленные в этом релизе бэкбоны только для классификации — [ViT](/docs/models/vit),
[Swin](/docs/models/swin) и [DeiT](/docs/models/deit) — объявляют только `classify`
и эту задачу не обслуживают.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` — короткий путь для батчей: он запускает `predict`
и объединяет все строки всех результатов в один CPU-тензор float32 формы
`(N_total, D)`, выбрасывая исключение, если размерности строк не совпадают.
Семейство, у которого `embed` нет в списке поддерживаемых задач, выбрасывает
`NotImplementedError`.

## Данные результата

`result.embeddings` — это объект `Embeddings`. Его `data` всегда `(N, D)` float32,
уже L2-нормализованный на этапе инференса, а недвумерный вход приводит
к исключению, а не к молчаливому изменению формы.

| Член | Значение |
|---|---|
| `.data` | Матрица `(N, D)` |
| `.dim` | `D` |
| `.normalized` | Те же строки, на всякий случай перенормированные |
| `.similarity(other)` | `(N, M)` при сравнении с другим набором или `(N,)` при сравнении с одним вектором `(D,)` |
| `.verify(i, j, threshold=0.4)` | Один ли субъект в строках `i` и `j` |

`result.identities` — это объект `Identities`, он появляется только тогда, когда
была передана галерея. Это обычный контейнер, а не тензор, поэтому перенос
`Results` между устройствами его не трогает.

| Член | Значение |
|---|---|
| `.name` | Список имён, `None` там, где ничто не преодолело порог |
| `.score` | `(N,)` float32, лучшая косинусная оценка; сохраняется, даже если имя — `None` |
| `.data` | Список кортежей `(name, score)` |

<code-tabs name="similarity" />

Векторы по умолчанию не попадают в `summary()` и `to_json()`, поскольку строка из
512 чисел float — это около двух килобайт на субъект. Вместо них каждая строка
сообщает `embedding_dim`, а также `identity` и `identity_score`, если
использовалась галерея. Передайте `summary(embeddings=True)`, чтобы включить сами
числа.

## Галереи

`Gallery` — это именованный набор эталонных строк. Каждый эталон хранится
отдельно, а не усредняется, поэтому имя оценивается по одному лучшему совпавшему
эталону, и добавление плохой фотографии не может утащить центроид личности в
сторону.

<code-tabs name="gallery" />

`Gallery(model)` привязывается к весам, которые будут производить его векторы.
`enroll(name, sources, select="best")` запускает предсказание на каждом источнике
и оставляет из каждого результата строку с наибольшей уверенностью; `select="all"`
оставляет вместо этого все строки — это то, что нужно, когда на эталонном
изображении действительно есть несколько субъектов. `enroll_embedding(name, vector)`
пропускает инференс и принимает вектор напрямую, нормализуя его и отклоняя строку
из одних нулей.

`FaceGallery` — постоянный псевдоним того же класса, и архивы, записанные прежними
релизами с поддержкой одних только лиц, по-прежнему загружаются.

### Сопоставление и пороги

Сопоставление — это плотное матричное умножение на все сохранённые эталоны,
сведённое к одной оценке на имя взятием максимума. Приближённого индекса нет,
поэтому числа остаются точными, но размер галереи на практике ограничен.

Две точки входа различаются тем, что делают ниже порога. `match()` возвращает
`[(name, score), ...]` для каждой строки, отбрасывая всё, что ниже порога, поэтому
строка без совпадений — это пустой список. `identify()` возвращает объект
`Identities`, который всегда сохраняет лучшую оценку и ставит имя `None`, когда
она ниже порога. Ни одна из них никогда не подставляет ближайшее имя, не
дотянувшее до порога.

Порог по умолчанию везде `0.4`. Это косинусное значение, а не вероятность, и
правильная рабочая точка — свойство ваших данных и вашей терпимости к ложным
совпадениям, поэтому подбирайте её на размеченных парах, а не принимайте значение
по умолчанию. `libreyolo enroll` и аргумент предсказания `gallery=` используют то
же число.

### Сохранение

`save(path)` пишет сжатый `.npz` с векторами, именами и блоком метаданных, где
записаны версия формата, размерность эмбеддинга и отпечаток весов, породивших эти
строки. `Gallery.load(path, model=...)` проверяет и то и другое, прежде чем
что-либо сравнивать, поэтому попытка направить галерею на другую модель приводит к
исключению, а не к тихому сравнению векторов из двух несвязанных пространств.
Сохранить пустую галерею нельзя.

## Командная строка

| Команда | Назначение |
|---|---|
| `libreyolo enroll` | Обойти дерево «папка на личность» и записать или дополнить галерею `.npz` |
| `libreyolo compare` | Построить эмбеддинги главного субъекта на двух изображениях и вывести косинусную близость |
| `libreyolo verify` | Та же команда под вторым именем |
| `libreyolo predict gallery=...` | Прикрепить личности к обычному запуску предсказания |

<code-tabs name="cli" />

Каждая команда LibreYOLO принимает и `key=value`, и `--key value`, поэтому
`gallery=refs.npz` и `--gallery refs.npz` — один и тот же аргумент.

`enroll` принимает `model`, `source` и `gallery`, а также необязательные
`face-detector`, `device`, `--json` и `--quiet`. Он читает по одной папке на
личность, где имя папки — это личность, а каждое изображение внутри даёт эталоны:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Изображение, которое ничего не дало, не прерывает запуск: оно пропускается
со строкой в stderr, а в сводке сообщается, сколько эталонов сохранено для каждого
имени. Существующий файл галереи дополняется на месте, поэтому личности можно
добавлять со временем.

`compare` и `verify` — одна функция, зарегистрированная дважды. Они принимают
`model`, `source`, `source2` и необязательный `threshold` и печатают косинусную
близость, вердикт «тот же или другой» и порог, который его дал. `--json` печатает
те же три поля объектом.

В `predict` аргумент `gallery` указывает на сохранённый `.npz`, а
`gallery_threshold` переопределяет значение по умолчанию `0.4`. Передача галереи
модели, задача которой не `embed`, — это ошибка, а не тихое бездействие, а
отсутствующий файл галереи подсказывает команду `libreyolo enroll`, которая его
создаст.

## Лица

Распознавание лиц — это форма «область» этой задачи и единственная поставляемая
реализация такой формы. Оно добавляет стадию детекции и выравнивания перед головой
эмбеддинга, а ещё метод `verify()`, аргумент для собственных рамок, опубликованные
значения accuracy и рекомендации по калибровке порога. Всё это описано на странице
[распознавания лиц](/docs/tasks/face-recognition) — именно её стоит читать, когда
речь о лицах. Всё на этой странице применимо к ней без изменений.

## Обучение, валидация и экспорт

Ничто в этой задаче не обучается внутри LibreYOLO. Голова эмбеддинга лиц — это
ONNX-артефакт, у которого `train()`, `val()` и `export()` выбрасывают исключение;
обучите голову вне библиотеки и загрузите файл по пути. CLIP, SigLIP 2 и DINOv2
обучаются и экспортируются через свои задачи классификации и сегментации, а не
через `embed`.

Валидатора для поиска похожих нет. Измеряйте accuracy верификации на размеченных
парах, подбирая `threshold`, а accuracy идентификации — регистрируя галерею и
читая `identities.name` и `identities.score` на отложенных изображениях, считая
имя `None` отказом.
