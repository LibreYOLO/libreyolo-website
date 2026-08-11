---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything: детекция с открытым словарём и указание точек'
description: >-
  LocateAnything в LibreYOLO: детекция с открытым словарём и указание точек.
  Предсказание по любой текстовой метке; обучение, валидация и экспорт не
  поддерживаются.
lead: >-
  LocateAnything — vision-language модель grounding от NVIDIA, которая
  декодирует ограничивающие рамки и точки параллельно, а не по одному
  координатному токену за раз. LibreYOLO оборачивает её как детектор с открытым
  словарём, умеющий ещё и указывать точки: набором классов становится любой
  список текстовых меток, без фиксированной головы и без дообучения.
keywords:
  - LocateAnything
  - NVIDIA
  - детекция с открытым словарём
  - locateanything python
  - vision-language модель
  - детекция точек
  - VLM grounding
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        model = LibreLocateAnything(size="3b")


        # Открытый словарь: годятся любые слова, фиксированной головы классов
        нет.

        # Запоминается и действует во всех последующих predict()/track(), пока
        не задать снова.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Указание точек
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        # task="point" возвращает по одной точке на найденный объект вместо
        рамки.

        # Переключить задачу у уже загруженной модели: model.set_task("point").

        model = LibreLocateAnything(size="3b", task="point")

        model.set_classes(["the person closest to the camera"])

        result = model(SAMPLE_IMAGE, save=True)


        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Чат напрямую
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        model = LibreLocateAnything(size="3b")


        # Запасной выход под удобной обёрткой детекции: произвольные вопросы,

        # подсчёт объектов или любой промпт, который обёртка с рамками не
        покрывает.

        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")

        print(text)
source_hash: 378ea758e507a096
---

## Установка

LocateAnything нужен extra `vlm`: он подтягивает `transformers`, а также пакеты
`decord`, `lmdb` и `peft`, которые удалённый код с Hugging Face импортирует при
загрузке модели.

```bash
pip install "libreyolo[vlm]"
```

## Предсказание

`LibreLocateAnything` — это Python-класс, а не `.pt`-чекпойнт: через фабрику
`LibreYOLO()` он не загружается, и CLI `libreyolo` его не распознаёт. До этого
семейства можно добраться и через фабрику `LibreVLM(...)`
(`from libreyolo import LibreVLM`) — по алиасу, например
`LibreVLM("locate-anything")`; она конструирует как раз тот класс, который
используется ниже. При загрузке скачивается и выполняется собственный удалённый
код модели от NVIDIA с Hugging Face, поэтому LibreYOLO закрепляет скачивание за
одной фиксированной ревизией коммита, а не за изменяемой веткой `main`, и один
раз выводит в лог уведомление о лицензии перед первым скачиванием.

<code-tabs name="predict" />

`result.boxes` (задача `detect`) и `result.points` (задача `point`) содержат
разобранный вывод, как и у любого другого семейства. Уверенность здесь
заглушка: LocateAnything не выдаёт оценку для каждой рамки, поэтому у всех
детекций одна и та же константа, а `conf=` лишь отбрасывает строки ниже этой
константы, но не ранжирует их. Если не вызывать `set_classes()`, словарём по
умолчанию становятся имена классов COCO-80. Про источники, стриминг и обработку
результатов см. [предсказание](/docs/predict).

## Варианты

Опубликован один размер, 3b. Две задачи используют одни и те же веса: `detect`
(по умолчанию) возвращает рамки, а `task="point"` — вместо этого одну точку на
каждый найденный объект, в `result.points`; переключаться между ними у уже
загруженной модели можно через `model.set_task("point")`. Бенчмарк-обвязка
LibreYOLO это семейство не измеряла, поэтому опубликованных цифр точности для
сравнения нет.

LibreYOLO открывает это семейство только для предсказания. `train()`, `val()` и
`export()` выбрасывают `NotImplementedError`: дообучать нужно в исходном
проекте, а сюда загружать результат; валидация на датасете пропущена, потому что
с уверенностью-заглушкой mAP по COCO вводил бы в заблуждение; а экспорт выходит
за рамки для генеративной модели, у которой нет state dict для трассировки.

## Лицензирование

<provenance-box>

NVIDIA License разрешает использование, воспроизведение и модификацию, но для
всех, кроме NVIDIA и её аффилированных лиц, ограничивает модель и любые
производные некоммерческим использованием, исследованиями или оценкой: порога по
выручке или платного исключения нет. LocateAnything-3B к тому же собран из двух
других лицензированных компонентов: языкового бэкбона Qwen2.5-3B-Instruct под
Qwen Research License и визуального энкодера MoonViT-SO-400M под MIT. LibreYOLO
ничего из этого не размещает у себя, не зеркалирует и не распространяет
повторно: при первом запуске `LibreLocateAnything` скачивает веса и нужный
удалённый код напрямую из `nvidia/LocateAnything-3B` на Hugging Face, с
закреплением за одним фиксированным коммитом.

</provenance-box>

## Цитирование

<citation-block />
