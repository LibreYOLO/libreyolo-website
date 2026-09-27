---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo в LibreYOLO: zero-shot детекция в реальном времени'
description: >-
  Используйте OMDet-Turbo в LibreYOLO для детекции с открытым словарём в
  реальном времени. Установите extra openvocab и предсказывайте со словарём из
  свободного текста.
lead: >-
  OMDet-Turbo — детектор объектов с открытым словарём, работающий в реальном
  времени; он разработан в Om AI Lab и отделяет эмбеддинги классов от языкового
  промпта задачи. LibreYOLO подключает его как семейство только для предсказания
  на своём уровне детекторов с открытым словарём.
keywords:
  - OMDet-Turbo
  - OmDet
  - детекция с открытым словарём
  - детекция объектов в реальном времени
  - zero-shot детекция объектов
  - детекция объектов по тексту
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Свой порог NMS
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["traffic light", "bicycle"])

        # OMDet-Turbo — единственное семейство этого уровня, которое учитывает
        # iou=: его собственная постобработка принимает порог подавления как
        # аргумент, а если iou= не задан, берётся 0.5.
        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)
        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Установка

OMDet-Turbo загружается через уровень детекторов с открытым словарём в
LibreYOLO, которому нужен extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Этот extra подтягивает `transformers` и `timm` — библиотеки Hugging Face, к
которым обращается этот уровень; Swin-бэкбон OMDet-Turbo загружается через
обёртку `TimmBackbone` из `transformers`.

## Предсказание

OMDet-Turbo — не тот чекпойнт, который LibreYOLO загружает через `LibreYOLO()`.
Он загружается через соседнюю фабрику `LibreOpenVocab`: она скачивает снапшот с
Hugging Face при первом запуске и кэширует его в `weights/`.

<code-tabs name="predict" />

`set_classes()` задаёт постоянный текстовый словарь: вызовите его ещё раз,
чтобы полностью заменить список, или не вызывайте вовсе, чтобы остались
стандартные метки COCO-80, и пустой результат здесь — допустимый исход, а не
ошибка. В отличие от Grounding DINO, OMDet-Turbo отделяет эмбеддинги классов от
языкового промпта задачи, поэтому постобработка `transformers` возвращает
метки, которые напрямую сопоставляются с запрошенным списком классов, без шага
разрешения неоднозначности фраз.

Порога по текстовым токенам у OMDet-Turbo нет: детекции фильтрует только
`conf`, а передача `text_threshold` вызывает ошибку. Это единственное семейство
этого уровня, которое выполняет собственное подавление немаксимумов внутри
`post_process_grounded_object_detection`, поэтому `iou` здесь учитывается, а не
приводит к предупреждению. `imgsz` и `augment=True` отклоняются сразу:
изменением размера занимается процессор `transformers`, а аугментация на этапе
теста выходит за рамки этого уровня. `predict()` на одном изображении
возвращает один `Results`, а не список; чтобы получить несколько, передайте
директорию, список изображений или `stream=True` для видеоисточника. CLI-пути у
этого семейства нет, `libreyolo predict` загружает только чекпойнты `.pt` через
`LibreYOLO()`, поэтому семейства `LibreOpenVocab` запускаются из Python. Про
типы источников и стриминг см. [предсказание](/docs/predict).

## Варианты

Один чекпойнт, `t`, единственный размер на этом уровне. Он повторяет
`omlab/omdet-turbo-swin-tiny-hf` на зафиксированной ревизии upstream через
`OmDetTurboForObjectDetection` из `transformers`; файл весов в зеркале
побайтово идентичен этому снапшоту upstream. Ни точности, ни задержек для этого
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
