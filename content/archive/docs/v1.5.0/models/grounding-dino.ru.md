---
title: Grounding DINO
families:
  - grounding_dino
seo_title: 'Grounding DINO в LibreYOLO: детекция с открытым множеством классов'
description: >-
  Используйте Grounding DINO в LibreYOLO, чтобы находить любой объект, описанный
  текстом. Установите extra openvocab и предсказывайте со словарём из свободного
  текста.
lead: >-
  Grounding DINO — детектор с открытым множеством классов от IDEA Research,
  который сопоставляет изображение со свободным текстовым запросом вместо
  фиксированного списка классов. LibreYOLO подключает его как семейство только
  для предсказания на своём уровне детекторов с открытым словарём.
keywords:
  - Grounding DINO
  - детекция объектов по текстовому запросу
  - детекция с открытым словарём
  - zero-shot детекция объектов
  - grounding dino python
  - open-set детекция
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Текстовый порог
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf фильтрует по оценке рамки, а text_threshold — по оценке
        # токенов декодированной фразы. Если их не задавать, оба равны 0.25.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Установка

Grounding DINO загружается через уровень детекторов с открытым словарём в
LibreYOLO, которому нужен extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Этот extra подтягивает `transformers` и `timm` — библиотеки Hugging Face, к
которым обращается этот уровень.

## Предсказание

Grounding DINO — не тот чекпойнт, который LibreYOLO загружает через
`LibreYOLO()`. Он загружается через соседнюю фабрику `LibreOpenVocab`: она
скачивает снапшот с Hugging Face при первом запуске и кэширует его в
`weights/`.

<code-tabs name="predict" />

`set_classes()` задаёт постоянный текстовый словарь: вызовите его ещё раз,
чтобы заменить список, или не вызывайте вовсе, чтобы остались стандартные
метки COCO-80. Grounding DINO сам декодирует свободные фразы из своего
текстового выхода и сопоставляет их обратно с этим словарём, точное
нормализованное совпадение выигрывает, совпадение по целым токенам
принимается, а неоднозначная или несопоставленная фраза отбрасывается, а не
угадывается, поэтому `school bus` никогда не превратится в `bus` или в одно
только `school`. Словарь, который длиннее лимита токенов текстового энкодера,
разбивается на несколько запросов, прогоняется отдельными прямыми проходами и
снова сливается в один набор детекций, ограниченный `max_det`.

`iou` принимается ради совместимости API, но выдаёт предупреждение и ничего не
делает, потому что подавления немаксимумов здесь нет. `imgsz` и `augment=True`
отклоняются сразу: изменением размера занимается процессор `transformers`, а
аугментация на этапе предсказания выходит за рамки этого уровня. `predict()`
на одном изображении возвращает один `Results`, а не список; чтобы получить
несколько, передайте директорию, список изображений или `stream=True` для
видеоисточника. CLI-пути у этого семейства нет, `libreyolo predict` загружает
только чекпойнты `.pt` через `LibreYOLO()`, поэтому семейства
`LibreOpenVocab` запускаются из Python. Про типы источников и стриминг см.
[предсказание](/docs/predict).

## Варианты

Два чекпойнта, `t` и `b`. `t` — размер по умолчанию для этого уровня, когда
ничего не задано. Оба повторяют официальный релиз IDEA Research через
`GroundingDinoForObjectDetection` из `transformers` и один раз скачиваются в
размещённый LibreYOLO снапшот Hugging Face, где сохранены исходные файлы. Ни
точности, ни задержек для этого семейства пока не опубликовано.

Обучение, валидация на датасете и экспорт выходят за рамки этого уровня:
`train()`, `val()` и `export()` безусловно вызывают `NotImplementedError`.
Это обёртка только для предсказания вокруг опубликованного чекпойнта.

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
