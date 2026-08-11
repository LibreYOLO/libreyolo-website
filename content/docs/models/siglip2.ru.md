---
title: SigLIP2
families:
  - siglip2
seo_title: 'SigLIP2 в LibreYOLO: zero-shot классификация и эмбеддинги'
description: >-
  Используйте SigLIP2 в LibreYOLO для zero-shot классификации изображений и
  эмбеддингов изображений и текста, с независимой сигмоидной оценкой по каждому
  классу. Обучение не нужно.
lead: >-
  SigLIP2 — двухбашенная модель, которая оценивает изображение по текстовым
  запросам независимой сигмоидой на каждый класс, а не общим softmax по
  фиксированному набору меток. LibreYOLO поддерживает её для zero-shot
  классификации и эмбеддингов изображений и текста, без этапа обучения.
keywords:
  - SigLIP2
  - SigLIP 2
  - zero-shot классификация изображений
  - эмбеддинги изображений
  - эмбеддинги текста
  - открытый словарь
  - мультиязычная модель
  - sigmoid loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Без вызова set_classes() предсказание из CLI использует 1000 имён

        # классов ImageNet, с которыми модель загружается по умолчанию.

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Оценка сигмоидой с несколькими метками
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)
        r = model(SAMPLE_IMAGE)

        # Независимые вероятности по каждому классу: высоко сразу могут
        # оценить несколько классов или ни одного. Softmax (по умолчанию)
        # вместо этого нормирует их в распределение с одной меткой,
        # повторяя поведение LibreCLIP.
        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Эмбеддинги изображений и текста
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Оба L2-нормированы, поэтому обычное скалярное произведение —
        косинусная близость.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        # data — это корень ImageFolder со сплитом train/; имена его папок
        # становятся zero-shot запросами классов для этого запуска.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Текущие метки из set_classes() и разрешение входа запекаются
        # в граф. После смены любого из двух экспортируйте заново.
        # На момент экспорта multi_label должен быть False (по умолчанию).
    - label: CLI
      language: bash
      code: |
        # Здесь нет вызова set_classes(), поэтому запекаются 1000 классов
        # ImageNet, с которыми модель загружается по умолчанию.
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Экспорт эмбеддингов
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" трассирует только башню изображений; классы не нужны.
        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: f992655747fd8819
---

## Установка

SigLIP2 требует собственного extra, который подтягивает пакет SentencePiece, используемый его мультиязычным токенизатором.

```bash
pip install "libreyolo[siglip2]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`set_classes()` — тот самый примитив, который делает эту модель классификатором с открытым словарём: он подставляет каждую метку во все шаблоны запросов, кодирует и усредняет результаты, а получившуюся матрицу `[K, D]` кэширует как голову классификатора, поэтому она не пересчитывается для каждого изображения. Вызовите его снова, чтобы сменить классы в любой момент. Без такого вызова LibreSigLIP2 загружается с уже заданными 1000 именами классов ImageNet-1k.

SigLIP оценивает каждый класс независимо: `logit = scale * (image . text) + bias`. По умолчанию этот набор логитов всё же проходит через softmax, давая распределение с одной меткой, которое повторяет поведение `top1`/`top5` у LibreCLIP. Передача `multi_label=True` в `set_classes()` (или при создании модели) переключает на независимые сигмоидные вероятности, поэтому на одном и том же изображении высоко могут оценить сразу несколько классов или ни одного. Токенизатор — мультиязычная модель SentencePiece (словарь Gemma), поэтому имена классов на языках, отличных от английского, работают точно так же.

С `task="embed"` предсказание возвращает по одному L2-нормированному вектору изображения на каждый вход вместо вероятностей классов, а `embed_text()` возвращает нормированные строки текста в том же векторном пространстве, поэтому обычное скалярное произведение между ними — косинусная близость. `iou` не влияет ни на одну из этих задач; шага NMS здесь нет. Про источники, стриминг и обработку результатов см. [предсказание](/docs/predict).

## Валидация

`val()` читает имена папок классов в сплите `train/` формата ImageFolder, вызывает с ними `set_classes()`, а затем измеряет zero-shot точность top-1 и top-5 при оценке через softmax. Точность зависит от того, как имена классов читаются в роли запросов, а не от какого-либо обновления весов, поскольку обучать здесь нечего. Валидация покрывает только `task="classify"`; у `task="embed"` валидатора датасета нет.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспорт запекает текущее состояние модели в фиксированный граф. Для `task="classify"` те метки, которые последний раз задал `set_classes()`, и разрешение на момент экспорта запекаются в финальный линейный слой с обученными scale и bias, поэтому экспортированный граф — обычный классификатор изображений `[B, K]` без текстовой башни и без токенизатора; после смены классов или размера экспортируйте заново. Экспорт в режиме `multi_label=True` не реализован; сначала верните значение `False`. Экспорт с `task="embed"` трассирует только башню изображений. Обоим нужен ONNX opset 14 или выше, который экспортёр ставит по умолчанию.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства. Оба сконвертированы из чекпойнтов Google `siglip2-base-patch16-256` и `siglip2-so400m-patch14-384` под Apache-2.0, а не из какого-либо запуска обучения на COCO.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
