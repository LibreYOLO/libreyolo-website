---
title: CLIP
families:
  - clip
seo_title: 'CLIP в LibreYOLO: zero-shot классификация и эмбеддинги'
description: >-
  Используйте CLIP в LibreYOLO для zero-shot классификации изображений и
  получения эмбеддингов изображений и текста. Обучение не нужно: set_classes()
  задаёт набор меток во время выполнения.
lead: >-
  CLIP — двухбашенная модель, которая оценивает изображение по текстовым
  запросам, а не по фиксированному набору меток. LibreYOLO поддерживает её для
  zero-shot классификации и эмбеддингов изображений и текста, без этапа
  обучения.
keywords:
  - CLIP
  - OpenCLIP
  - zero-shot классификация изображений
  - эмбеддинги изображений clip
  - эмбеддинги текста
  - открытый словарь
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Без вызова set_classes() предсказание из CLI использует 1000 имён

        # классов ImageNet, с которыми модель загружается по умолчанию.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Эмбеддинги изображений и текста
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

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

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data — это корень ImageFolder со сплитом train/; имена его папок
        # становятся zero-shot запросами классов для этого запуска.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Текущие метки из set_classes() и разрешение входа запекаются
        # в граф. После смены любого из двух экспортируйте заново.
    - label: CLI
      language: bash
      code: |
        # Здесь нет вызова set_classes(), поэтому запекаются 1000 классов
        # ImageNet, с которыми модель загружается по умолчанию.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Экспорт эмбеддингов
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" трассирует только башню изображений; классы не нужны.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Установка

CLIP требует собственного extra, который подтягивает пакеты, нужные его встроенному BPE-токенизатору, чтобы воспроизводить точные id токенов.

```bash
pip install "libreyolo[clip]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

`set_classes()` — тот самый примитив, который делает эту модель классификатором с открытым словарём: он подставляет каждую метку во все шаблоны запросов, кодирует и усредняет результаты, а получившуюся матрицу `[K, D]` кэширует как голову классификатора, поэтому она не пересчитывается для каждого изображения. Вызовите его снова, чтобы сменить классы в любой момент. Без такого вызова LibreCLIP загружается с уже заданными 1000 именами классов ImageNet-1k.

С `task="embed"` предсказание возвращает по одному L2-нормированному вектору изображения на каждый вход вместо вероятностей классов, а `embed_text()` возвращает нормированные строки текста в том же векторном пространстве, поэтому обычное скалярное произведение между ними — косинусная близость. `iou` не влияет ни на одну из этих задач; шага NMS здесь нет. Про источники, стриминг и обработку результатов см. [предсказание](/docs/predict).

## Валидация

`val()` читает имена папок классов в сплите `train/` формата ImageFolder, вызывает с ними `set_classes()`, а затем измеряет zero-shot точность top-1 и top-5. Точность зависит от того, как имена классов читаются в роли запросов, а не от какого-либо обновления весов, поскольку обучать здесь нечего. Валидация покрывает только `task="classify"`; у `task="embed"` валидатора датасета нет.

<code-tabs name="val" />

## Экспорт

<export-matrix />

Экспорт запекает текущее состояние модели в фиксированный граф. Для `task="classify"` те метки, которые последний раз задал `set_classes()`, и разрешение на момент экспорта запекаются в финальный линейный слой, поэтому экспортированный ONNX- или TensorRT-граф — обычный классификатор изображений `[B, K]` без текстовой башни и без токенизатора; после смены классов или размера экспортируйте заново. Экспорт с `task="embed"` трассирует только башню изображений. Обоим нужен ONNX opset 14 или выше, который экспортёр ставит по умолчанию.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства. Оба сконвертированы из чекпойнтов OpenCLIP, обученных на LAION-2B (`ViT-B-32` и `ViT-B-16`), а не из какого-либо запуска обучения на COCO.

<checkpoint-table />

У обучающих данных LAION-2B есть задокументированная история присутствия материалов CSAM (Stanford Internet Observatory, декабрь 2023). Позже LAION выпустил Re-LAION — очищенное переиздание; если вы размещаете эти веса у себя и раздаёте дальше, отдавайте предпочтение чекпойнтам на основе Re-LAION, где они доступны.

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
