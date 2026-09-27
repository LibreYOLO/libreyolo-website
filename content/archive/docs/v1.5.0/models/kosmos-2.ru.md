---
title: Kosmos-2
families:
  - kosmos2
seo_title: 'Kosmos-2 в LibreYOLO: детекция объектов с привязкой к тексту'
description: >-
  Kosmos-2 в LibreYOLO: установка, задание открытого словаря и предсказание
  рамок с привязкой к тексту моделью Microsoft под лицензией MIT.
lead: >-
  Kosmos-2 — модель Microsoft для привязки текста к изображению: она составляет
  описание картинки, а затем находит каждую именную группу из этого описания и
  обводит её рамкой. LibreYOLO оборачивает её как детектор объектов с открытым
  словарём: список классов передаётся во время предсказания.
keywords:
  - Kosmos-2
  - kosmos-2 python
  - grounding изображений
  - детекция с открытым словарём
  - vision-language модель
  - Microsoft VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Видео
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])

        # Любой источник, который принимает библиотека: файл, папка, URL, индекс
        # веб-камеры, RTSP-поток или список .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: 60e0796f34be6d59
---

## Установка

Kosmos-2 относится к уровню «VLM как детектор» в LibreYOLO — это отдельная часть продукта со своей фабрикой, а не одно из семейств на основе чекпойнтов. Ему нужен extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально. LibreYOLO загружает напрямую собственный репозиторий Microsoft `microsoft/kosmos-2-patch14-224`; в отличие от Florence-2, перезалив от сообщества здесь не нужен.

<code-tabs name="predict" />

Это семейство загружается через фабрику `LibreVLM()`, а не `LibreYOLO()`: у VLM-семейств нет загрузчика чекпойнтов, поэтому маршрутизация по суффиксу файла, описанная на страницах других моделей, здесь не работает. `set_classes()` задаёт словарь, который Kosmos-2 просят найти на изображении; значение запоминается, поэтому оно действует во всех последующих вызовах `predict()`/`track()`, пока вы не зададите его снова. Kosmos-2 привязывает к изображению именные группы, а не сопоставляет метку точь-в-точь, поэтому обёртка LibreYOLO допускает частичное совпадение: класс с именем `"boat"` совпадёт и со сгенерированной фразой вроде «the boats». У каждой детекции одна и та же уверенность-заглушка, поэтому фильтрация по `conf` работает по принципу «всё или ничего», а не как ранжирование, и `iou` здесь ни на что не влияет: обёртка строит список детекций напрямую из привязанных сущностей, без шага удаления дубликатов. `chat()` выбрасывает `NotImplementedError`, потому что Kosmos-2 управляется промптом `<grounding>`, а не шаблоном чата. CLI LibreYOLO этот уровень не покрывает: формы `libreyolo predict model=...` для него нет. Про источники, стриминг и обработку результатов см. [предсказание](/docs/predict).

## Варианты

Один размер: `kosmos-2-patch14-224`, на 224 px, загружается как `LibreVLM("kosmos-2")`. Это модель 2023 года, и в самой обёртке LibreYOLO отмечено, что привязка у неё грубее, чем у более новых детекторов этого уровня.

LibreYOLO не обучает, не валидирует и не экспортирует Kosmos-2: `train()`, `val()` и `export()` выбрасывают `NotImplementedError` для каждого семейства этого уровня (см. уровень поддержки выше). Если нужен свой словарь, запечённый в модель, дообучите Kosmos-2 в исходном проекте и загрузите получившиеся веса; вывод `predict()` проверяйте глазами, а не прогоном валидации в стиле COCO, поскольку у каждой детекции одна и та же уверенность-заглушка.

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
