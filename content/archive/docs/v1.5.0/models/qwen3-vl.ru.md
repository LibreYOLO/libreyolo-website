---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: 'Qwen3-VL в LibreYOLO: детекция с открытым словарём'
description: >-
  Qwen3-VL в LibreYOLO: установка, задание открытого словаря, предсказание рамок
  или чат с vision-language моделью Alibaba под лицензией Apache-2.0.
lead: >-
  Qwen3-VL — vision-language модель Alibaba с нативной 2D-привязкой к
  изображению (grounding). LibreYOLO оборачивает её как детектор объектов с
  открытым словарём и заодно открывает её свободный чат напрямую: передайте
  список классов, чтобы детектировать, или просто задайте вопрос.
keywords:
  - Qwen3-VL
  - qwen3-vl python
  - детекция с открытым словарём
  - детекция объектов без обучения
  - vision-language модель
  - grounding
  - Alibaba
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Чат
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # Запасной выход под удобной обёрткой для детекции: любой вопрос,

        # а не только запрос про рамки.

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: ee225b6221d624d9
---

## Установка

Qwen3-VL относится к уровню «VLM как детектор» в LibreYOLO — это отдельная часть продукта со своей фабрикой, а не одно из семейств на основе чекпойнтов. Ему нужен extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально. `LibreVLM()` без аргумента по умолчанию берёт Qwen3-VL-4B.

<code-tabs name="predict" />

Это семейство загружается через фабрику `LibreVLM()`, а не `LibreYOLO()`: у VLM-семейств нет загрузчика чекпойнтов, поэтому маршрутизация по суффиксу файла, описанная на страницах других моделей, здесь не работает. `set_classes()` задаёт словарь, который Qwen3-VL просят найти на изображении; значение запоминается, поэтому оно действует во всех последующих вызовах `predict()`/`track()`, пока вы не зададите его снова. У каждой детекции одна и та же уверенность-заглушка, поэтому фильтрация по `conf` работает по принципу «всё или ничего», а не как ранжирование; а вот `iou` для этого семейства на результат влияет: очередная рамка того же класса отбрасывается, как только её перекрытие с уже оставленной превышает порог, — иначе генератор, начав повторяться, может выдать почти одинаковые рамки для одного объекта. В отличие от Florence-2 и Kosmos-2, Qwen3-VL отвечает и на произвольные вопросы через `chat()` — тот же запасной выход, что описан у фабрики `LibreVLM`. CLI LibreYOLO этот уровень не покрывает: формы `libreyolo predict model=...` для него нет. Про источники, стриминг и обработку результатов см. [предсказание](/docs/predict).

## Варианты

Три размера: Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct и Qwen3-VL-8B-Instruct, загружаются как `LibreVLM("qwen3-vl-2b")`, `LibreVLM("qwen3-vl-4b")` и `LibreVLM("qwen3-vl-8b")`. Все три заявляют номинальный вход 1024 px, но фактический холст, который уходит в сеть, определяется собственным smart-resize процессора Qwen, поэтому эта цифра — не фиксированное рабочее разрешение, как у остальных семейств на этом сайте. LibreYOLO не публиковал бенчмарк, сравнивающий точность трёх размеров.

LibreYOLO не обучает, не валидирует и не экспортирует Qwen3-VL: `train()`, `val()` и `export()` выбрасывают `NotImplementedError` для каждого семейства этого уровня (см. уровень поддержки выше). Если нужен свой словарь, запечённый в модель, дообучите Qwen3-VL в исходном проекте и загрузите получившиеся веса; вывод `predict()` проверяйте глазами, а не прогоном валидации в стиле COCO, поскольку у каждой детекции одна и та же уверенность-заглушка.

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
