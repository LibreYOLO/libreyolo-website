---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreMODUS в LibreYOLO: анализ изображений any-to-any'
description: >-
  Использование LibreMODUS в LibreYOLO для глубины, нормалей, границ и детекции,
  а также их композиции через any2any(). Только инференс; веса загружаются из
  EPFL-VILAB.
lead: >-
  LibreMODUS — интеграция чекпойнта MODUS 14B-A7B, работающая только на
  инференс. Это модель any-to-any, которая превращает один производный от
  изображения вход в другой: RGB на входе — глубина на выходе; глубина на входе
  — нормали на выходе; любой из них плюс фраза — рамки на выходе. LibreYOLO
  поддерживает четыре задачи через стандартный API предсказания и более широкий
  набор через any2any().
keywords:
  - LibreMODUS
  - MODUS
  - any-to-any
  - оценка глубины по изображению
  - нормали к поверхности
  - детекция границ python
  - детекция по текстовому запросу
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # Без своего словаря detect декодирует COCO-токены меток из
        # чекпойнта в непрерывные идентификаторы классов COCO-80.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Grounding по фразе
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(task="detect")
        # set_classes() переключает детекцию на grounding по фразе: каждая фраза
        # обрабатывается независимо и возвращается через тот же контракт Boxes.
        model.set_classes(["red bus", "cyclist"])
        result = model.predict("street.jpg", conf=0.2)
        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: >
        from libreyolo import LibreMODUS


        model = LibreMODUS()


        # От одного до трёх производных от изображения входов (rgb, depth,
        normal,

        # canny/edge) плюс необязательный вспомогательный текст, сведённые к
        одной цели.

        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )

        normals = result.normal_map.data


        # Для grounding через any2any() нужен текстовый вход с самой фразой.

        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )

        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## Установка

LibreMODUS требует собственного extra, который подтягивает `accelerate` ради диспетчеризации большой модели, нужной этому чекпойнту.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO не распространяет и не зеркалирует веса MODUS. По умолчанию загрузка модели `LibreMODUS` скачивает нужные файлы напрямую из `EPFL-VILAB/MODUS` на закреплённой ревизии Hugging Face, и для скачивания с нуля всегда нужен собственный аутентифицированный аккаунт Hugging Face, даже если гейт на стороне оригинального хостинга временно открыт. Ознакомьтесь с условиями оригинального проекта, примите их, а затем аутентифицируйтесь:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Чтобы вообще не ходить в сеть, укажите снапшот, который у вас уже есть:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

В этом каталоге должны лежать `model.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json` и `merges.txt`. Что именно разрешают условия чекпойнта — в разделе «Лицензирование» ниже.

## Предсказание

<code-tabs name="predict" />

Стандартный API задач покрывает четыре задачи, каждая из которых отображается в одну цель MODUS: `depth` — в относительную глубину (`result.depth_map`), `normal` — в нормали к поверхности (`result.normal_map`), `edge` — в границы в стиле Canny (`result.edges`), а `detect` — в рамки COCO-80 (`result.boxes`), если только `set_classes()` не переключит её на grounding по фразе. `set_task()` переключает их на одной и той же уже загруженной модели. Опубликованный рецепт использует десять шагов flow-сэмплирования с guidance по тексту 4.0 и guidance по изображению 2.0; переопределить их можно параметрами `inference_steps=`, `inference_cfg=` и `inference_image_cfg=` при создании модели.

`any2any()` открывает более широкую публичную поверхность анализа: от одного до трёх производных от изображения входов (`rgb`, `depth`, `normal`, `canny`/`edge`) плюс необязательный вспомогательный текст, сведённые к любой из целей — глубине, нормалям, границам, границам на основе SAM, детекции COCO или grounding по фразе. Все производные от изображения входы должны описывать один и тот же выровненный холст; при несовпадении ширины и высоты LibreMODUS отклоняет их, а не масштабирует по отдельности. `chain=(...)` генерирует промежуточные цели и подаёт их обратно в тот же контекст, в пределах обучающего бюджета чекпойнта в три условия. `verify=N` (N >= 2) генерирует N кандидатов и оставляет того, кто набирает больше всего по ограниченной проверке самосогласованности; она доступна как `result.verification_score`.

`dtype="bf16"` (по умолчанию) соответствует точности опубликованного чекпойнта; `dtype="fp8"` хранит подходящие линейные веса ствола декодера в E4M3 с масштабом на каждый выходной канал, один раз конвертирует их в локальный кэш в `~/.cache/libreyolo/modus/fp8` и деквантует до dtype входа на каждое матричное умножение, то есть выигрывает в памяти, а не жертвует точностью на уровне активаций.

`train()`, `val()` и `export()` выбрасывают исключение: LibreMODUS работает только на инференс, валидации на датасете нет, а пути экспорта в ONNX, TensorRT или TFLite не существует. Пакетный `predict()` и аугментация во время инференса тоже не поддерживаются; за один вызов обрабатывается одно изображение.

## Лицензирование

<provenance-box>

LibreYOLO нигде не размещает и не зеркалирует чекпойнт MODUS, в том числе в собственной Hugging Face-организации: при загрузке он всегда тянет закреплённую ревизию напрямую из EPFL-VILAB/MODUS либо читает снапшот, уже лежащий на диске по пути `checkpoint_path`.

</provenance-box>

## Цитирование

<citation-block />
