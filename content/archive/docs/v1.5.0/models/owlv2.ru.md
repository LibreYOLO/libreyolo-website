---
title: OWLv2
families:
  - owlv2
seo_title: 'OWLv2 в LibreYOLO: zero-shot детекция объектов'
description: >-
  Используйте OWLv2 в LibreYOLO, чтобы находить любой объект, описанный текстом.
  Установите extra openvocab и предсказывайте со словарём из свободного текста.
lead: >-
  OWLv2 — детектор с открытым словарём от Google Research, который оценивает
  области изображения по текстовым эмбеддингам из энкодера в стиле CLIP.
  LibreYOLO подключает его как семейство только для предсказания на своём уровне
  детекторов с открытым словарём.
keywords:
  - OWLv2
  - OWL-ViT
  - детекция с открытым словарём
  - zero-shot детекция объектов
  - детекция объектов по текстовому запросу
  - owlv2 python
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Словарь по умолчанию
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        # Если не вызывать set_classes(), останется стандартный словарь COCO-80
        этого уровня.

        model = LibreOpenVocab("owlv2-l14")

        result = model.predict(SAMPLE_IMAGE, conf=0.1)

        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## Установка

OWLv2 загружается через уровень детекторов с открытым словарём в LibreYOLO,
которому нужен extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Этот extra подтягивает `transformers` и `timm` — библиотеки Hugging Face, к
которым обращается этот уровень.

## Предсказание

OWLv2 — не тот чекпойнт, который LibreYOLO загружает через `LibreYOLO()`. Он
загружается через соседнюю фабрику `LibreOpenVocab`: она скачивает снапшот с
Hugging Face при первом запуске и кэширует его в `weights/`.

<code-tabs name="predict" />

`set_classes()` задаёт постоянный текстовый словарь: вызовите его ещё раз,
чтобы заменить список, или не вызывайте вовсе, чтобы остались стандартные
метки COCO-80. Каждая метка оборачивается в фиксированный шаблон промпта,
прежде чем попасть в текстовую башню, — так же, как обучался
`Owlv2ForObjectDetection` из `transformers`.

У OWLv2 нет порога по текстовым токенам: детекции фильтрует только `conf`, а
передача `text_threshold` приводит к ошибке. `iou` принимается ради
совместимости API, но выдаёт предупреждение и ничего не делает, потому что
подавления немаксимумов здесь нет. `imgsz` и `augment=True` отклоняются сразу:
изменением размера занимается процессор `transformers`, а аугментация на этапе
предсказания выходит за рамки этого уровня. `predict()` на одном изображении
возвращает один `Results`, а не список; чтобы получить несколько, передайте
директорию, список изображений или `stream=True` для видеоисточника. CLI-пути
у этого семейства нет, `libreyolo predict` загружает только чекпойнты `.pt`
через `LibreYOLO()`, поэтому семейства `LibreOpenVocab` запускаются из Python.
Про типы источников и стриминг см. [предсказание](/docs/predict).

## Варианты

Два чекпойнта, `b16` (base, размер патча 16) и `l14` (large, размер патча 14).
`b16` — размер по умолчанию для этого уровня, когда ничего не задано. Оба
повторяют официальный релиз Google Research через `Owlv2ForObjectDetection` из
`transformers` и один раз скачиваются в размещённый LibreYOLO снапшот
Hugging Face, где сохранены исходные файлы. Ни точности, ни задержек для этого
семейства пока не опубликовано.

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
