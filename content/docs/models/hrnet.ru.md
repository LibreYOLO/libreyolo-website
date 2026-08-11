---
title: HRNet
families:
  - hrnet
seo_title: 'HRNet: оценка позы по схеме top-down в LibreYOLO'
description: >-
  Запуск HRNet в LibreYOLO для оценки позы COCO-17 по схеме top-down. Установка,
  предсказание, валидация и экспорт чекпойнтов W32 и W48 под лицензией MIT.
lead: >-
  HRNet — свёрточная сеть, которая держит поток признаков высокого разрешения на
  всём протяжении сети за счёт повторяющегося многомасштабного слияния, вместо
  того чтобы восстанавливать разрешение после понижения. LibreYOLO оборачивает
  официальный top-down вариант для оценки позы — для инференса и валидации.
keywords:
  - HRNet
  - оценка позы человека
  - ключевые точки человека python
  - coco-17 ключевые точки
  - top-down оценка позы
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Источник рамок человека не задан: HRNet сам подбирает лёгкий
        # детектор LibreYOLO9t и один раз сообщает об этом выборе.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Источник рамок человека
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # Полностью пропустить детекцию: всё изображение — один человек.
        result = model(SAMPLE_IMAGE, cropped=True)

        # Или передать HRNet рамки от уже запущенного детектора.
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # Или подключить конкретный детектор LibreYOLO вместо
        # LibreYOLO9t по умолчанию.
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Использование экспортированного файла
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # Экспортированный граф — только голова тепловых карт на фиксированном
        # холсте: принимает батч уже вырезанных и нормализованных кропов
        # человека и возвращает сырые тепловые карты. Детекция человека,
        # геометрия кропа, декодирование карт и подавление по OKS в граф не
        # входят; вне LibreYOLO шаг декодирования придётся написать самому.
        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")
        name = session.get_inputs()[0].name
        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## Установка

HRNet не требует ничего сверх базового пакета.

```bash
pip install libreyolo
```

Его детектор человека по умолчанию, лёгкий чекпойнт LibreYOLO9t, скачивается
автоматически, когда HRNet впервые с ним связывается.

## Предсказание

Веса скачиваются с Hugging Face при первом запуске и кэшируются локально.

<code-tabs name="predict" />

HRNet оценивает позу по схеме top-down: перед запуском головы позы ему нужна
рамка человека, поэтому каждый вызов её получает. Если ничего не указывать, он
при первом запуске подбирает детектор LibreYOLO9t и сообщает об этом выборе.
`cropped=True` пропускает детекцию и считает всё изображение одним человеком;
`person_boxes` принимает рамки от детектора, который вы уже запустили;
`person_detector` принимает `"auto"`, `"rfdetr"`, любую модель детекции
LibreYOLO или обычный вызываемый объект. `flip_test=True` дополнительно
прогоняет модель на горизонтально отражённом кропе и усредняет две тепловые
карты — это собственная аугментация HRNet на этапе теста; общий `augment=True`
здесь не определён. Источники из нескольких изображений обрабатываются
последовательно: детектор HRNet и переменное число людей на каждом изображении
не поддерживают предсказание батчем. Про источники, стриминг и работу с
результатами — [предсказание](/docs/predict).

## Варианты

Два размера, `w32` и `w48`, оба предсказывают стандартный набор ключевых точек
COCO-17 из кропа человека фиксированного разрешения; `w48` — более широкий из
двух бэкбонов.

Каталог моделей апстрима приводит точность позы для каждого размера со своим
детектором человека, своей настройкой flip-теста и официальным протоколом
оценки COCO. Связка LibreYOLO по умолчанию использует другой детектор, поэтому
прогон валидации здесь измеряет именно эту комбинацию, а не апстримную; чтобы
сойтись с цифрами апстрима, нужны те же рамки людей, те же оценки детектора и
та же настройка отражения, что и в исходной оценке.

## Валидация

`val()` считает keypoint OKS-AP в стиле COCO и принимает либо `data.yaml` в
формате YOLO-pose, либо COCO-разметку ключевых точек в JSON вместе с каталогом
изображений. Бэкенд метрик по умолчанию — faster-coco-eval, а `pycocotools`
подключается автоматически, если faster-coco-eval не установлен;
`faster_coco_eval=False` принудительно включает путь через `pycocotools`.

<code-tabs name="val" />

Внутри валидация запускает собственный `predict()` HRNet, поэтому использует
тот детектор человека, с которым модель была создана или вызвана. Задавайте
`person_detector=` явно при создании модели, чтобы этот источник оставался
одним и тем же от прогона к прогону, а не разрешался заново на каждом вызове.

## Экспорт

<export-matrix />

Контракт экспорта HRNet покрывает только ONNX, TorchScript, OpenVINO и
TensorRT; любой другой формат вызывает ошибку до начала трассировки. Любой
экспорт — это только голова тепловых карт на фиксированном холсте, батч из
одного элемента и FP32: она принимает кроп человека и возвращает сырые
тепловые карты. Аффинная геометрия кропа перед ней, а также декодирование
тепловых карт, восстановление после отражения и подавление по OKS после неё
остаются в Python, поэтому полному пайплайну «изображение на входе — ключевые
точки на выходе» всё равно нужен LibreYOLO на другом конце.

<code-tabs name="export" />

## Чекпойнты

Все опубликованные файлы весов этого семейства.

<checkpoint-table />

## Лицензирование

<provenance-box></provenance-box>

## Цитирование

<citation-block />
