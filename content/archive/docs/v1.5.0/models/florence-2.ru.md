---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 в LibreYOLO: детекция с открытым словарём'
description: >-
  Florence-2 в LibreYOLO: установка, задание открытого словаря и предсказание
  рамок моделью компьютерного зрения от Microsoft под лицензией MIT.
lead: >-
  Florence-2 — фундаментальная модель Microsoft для компьютерного зрения: задачу
  ей задают токеном, а не прогоняют вход через фиксированную голову детекции.
  LibreYOLO оборачивает её как детектор объектов с открытым словарём: список
  классов передаётся во время предсказания.
keywords:
  - Florence-2
  - детекция с открытым словарём
  - florence-2 python
  - vision-language модель
  - grounding
  - Microsoft VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Видео
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # Любой источник, который принимает библиотека: файл, папка, URL, индекс
        # веб-камеры, RTSP-поток или список .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Установка

Florence-2 относится к уровню «VLM как детектор» в LibreYOLO — это отдельная часть продукта со своей фабрикой, а не одно из семейств на основе чекпойнтов. Ему нужен extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально. LibreYOLO скачивает перезалитый чекпойнт от florence-community, а не оригинальный репозиторий `microsoft/Florence-2-*`; почему так — см. раздел «Лицензирование».

<code-tabs name="predict" />

Это семейство загружается через фабрику `LibreVLM()`, а не `LibreYOLO()`: у VLM-семейств нет загрузчика чекпойнтов, поэтому маршрутизация по суффиксу файла, описанная на страницах других моделей, здесь не работает. `set_classes()` задаёт словарь, который Florence-2 просят найти на изображении; значение запоминается, поэтому оно действует во всех последующих вызовах `predict()`/`track()`, пока вы не зададите его снова. Возвращаемый `Results` содержит `boxes` той же формы, что и у любого другого семейства, но у каждой детекции одна и та же уверенность-заглушка, поэтому фильтрация по `conf` работает по принципу «всё или ничего», а не как ранжирование, и `iou` ни на что не влияет: обёртка Florence-2 строит список детекций напрямую из разобранного вывода по токену задачи, без шага удаления дубликатов. `chat()` здесь выбрасывает `NotImplementedError`, потому что Florence-2 управляется токеном задачи `<OPEN_VOCABULARY_DETECTION>`, а не шаблоном чата. CLI LibreYOLO этот уровень не покрывает: формы `libreyolo predict model=...` для него нет. Про источники, стриминг и обработку результатов см. [предсказание](/docs/predict).

## Варианты

Два размера: Florence-2-base и Florence-2-large, оба на 768 px, загружаются как `LibreVLM("florence-2-base")` или `LibreVLM("florence-2-large")`. LibreYOLO не публиковал бенчмарк, сравнивающий их точность.

LibreYOLO не обучает, не валидирует и не экспортирует Florence-2: `train()`, `val()` и `export()` выбрасывают `NotImplementedError` для каждого семейства этого уровня (см. уровень поддержки выше). Если нужен свой словарь, запечённый в модель, дообучите Florence-2 в исходном проекте и загрузите получившиеся веса; вывод `predict()` проверяйте глазами, а не прогоном валидации в стиле COCO, поскольку у каждой детекции одна и та же уверенность-заглушка.

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
