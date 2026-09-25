---
title: "RT-DETR без Ultralytics: RT-DETR, v2 и v4 под Apache-2.0"
description: "Запускайте и дообучайте RT-DETR, RT-DETRv2 и RT-DETRv4 с весами Apache-2.0 в библиотеке под MIT: единый API для predict, train, val и export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "Можно ли использовать RT-DETR бесплатно в коммерческих целях?"
    a: "Да, если выбрать реализацию с разрешительной лицензией. Исходный код RT-DETR и RT-DETRv2 (lyuwenyu/RT-DETR), а также RT-DETRv4 (RT-DETRs/RT-DETRv4) распространяется под Apache-2.0, а LibreYOLO запускает все три версии с весами Apache-2.0 в библиотеке под MIT. Реализация RT-DETR в пакете ultralytics распространяется под AGPL-3.0; для закрытого ПО альтернативой служит платная Enterprise License. Это общая информация, а не юридическая консультация."
  - q: "Под какой лицензией распространяется RT-DETRv4?"
    a: "Apache-2.0. Файл LICENSE в github.com/RT-DETRs/RT-DETRv4 содержит лицензию Apache-2.0. RT-DETRv4 использует DINOv3 как учителя только во время обучения; в опубликованных весах студента нет параметров DINOv3, поэтому лицензия DINOv3 от Meta на них не распространяется."
  - q: "Можно ли дообучать RT-DETR без Ultralytics?"
    a: "Да. LibreYOLO дообучает детекционные чекпойнты RT-DETR, RT-DETRv2 и RT-DETRv4 с помощью model.train() или команды libreyolo train в командной строке, начиная с опубликованных весов COCO. Модели RT-DETRv2 с повёрнутыми рамками доступны только для инференса."
  - q: "Какие веса RT-DETR поставляются с LibreYOLO?"
    a: "RT-DETR в вариантах r18, r34, r50, r50m, r101, l и x; RT-DETRv2 в вариантах r18, r34, r50, r50m и r101; RT-DETRv4 в вариантах s, m, l и x — все для детекции COCO при 640 px. Для RT-DETRv2 также есть модели с повёрнутыми рамками n, s, m, l и x, обученные на DOTA v1.0 при 1024 px. Все файлы распространяются под Apache-2.0."
---

RT-DETR — трансформер для детекции в реальном времени от Baidu. Он декодирует фиксированный набор запросов вместо плотной сетки, поэтому настраивать этап NMS не нужно. Если поискать RT-DETR, одна из первых найденных реализаций, скорее всего, будет частью Python-пакета `ultralytics`. Тут и возникает вопрос о лицензии, хотя к самой модели он не относится.

## Лицензия RT-DETR зависит от репозитория

Лицензия относится к репозиторию, а не к модели. У RT-DETR несколько репозиториев, и лицензии в них различаются:

| Репозиторий | Содержимое | Лицензия |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR и RT-DETRv2, PyTorch и Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR в пакете `ultralytics` | AGPL-3.0 или Enterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2 и RT-DETRv4 | код под MIT, веса под Apache-2.0 |

Реализация Ultralytics сделана качественно. На [странице RT-DETR](https://docs.ultralytics.com/models/rtdetr/) перечислены предобученные `rtdetr-l.pt` и `rtdetr-x.pt`, а также обучение, валидация, инференс и экспорт. Модель входит в пакет под лицензией AGPL-3.0; Ultralytics продаёт Enterprise License командам, которые не хотят открывать исходный код всего проекта. Если не подходит ни один вариант, архитектура доступна под Apache-2.0 в репозитории её авторов. В [объяснении лицензий YOLO](/articles/yolo-licenses-explained) разобран тот же принцип для семейства YOLO, а в [руководстве по коммерческой лицензии](/articles/yolo-commercial-license) объясняется, что AGPL-3.0 означает для закрытого продукта.

## Запуск RT-DETR с LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

Эту же команду можно выполнить в командной строке:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

Параметры `conf` и `max_det` фильтруют декодирование top-k по запросам и классам. Параметр `iou` принимается, но не используется, поскольку NMS нет. Возвращается объект `Results`, общий для всех моделей LibreYOLO, поэтому для замены детектора достаточно изменить одну строку.

## RT-DETR, RT-DETRv2 и RT-DETRv4 в одном загрузчике

Версия указана в имени файла, а `LibreYOLO()` выбирает реализацию по чекпойнту, поэтому все три версии загружаются одинаково:

* **RT-DETR:** `LibreRTDETR` в вариантах r18, r34, r50, r50m, r101 (бэкбоны ResNet) и l, x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` в вариантах r18, r34, r50, r50m и r101. Архитектура версии 1 сохранена, но изменён способ выборки в deformable attention.
* **RT-DETRv4:** `LibreRTDETRv4` в вариантах s, m, l и x. В ней повторно используется архитектура D-FINE, а веса получены дистилляцией от учителя DINOv3 к студенту на HGNetv2.

Все детекционные веса обучены на COCO и работают при 640 px. Для сравнения: в README исходных проектов указано 46.5 AP для RT-DETR-R18 и 53.0 AP для RT-DETR-L на COCO, а для RT-DETRv4 — от 49.8 AP у RT-DETRv4-S до 57.0 AP у RT-DETRv4-X. В [документации RT-DETR](/docs/models/rt-detr) приведена собственная таблица бенчмарков LibreYOLO для каждого чекпойнта.

RT-DETRv2 также поддерживает повёрнутые рамки. `LibreRTDETRv2n-obb.pt`–`LibreRTDETRv2x-obb.pt` — официальные чекпойнты для DOTA v1.0, 15 классов аэрофотоснимков при 1024 px, преобразованные в формат LibreYOLO:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

В разделе [детекция с повёрнутыми рамками](/docs/tasks/oriented-detection) описаны формат меток и метрики.

## Дообучение RT-DETR на своих данных

Обучение начинается с опубликованного чекпойнта:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Для настоящего запуска укажите в `data` YAML-файл своего датасета. Для каждой версии задана своя скорость обучения по умолчанию; в версиях 1 и 2 скорость обучения бэкбона равна одной двадцатой `lr0`, как и в исходном рецепте. Для нескольких GPU в CLI укажите `device=0,1`, а для дообучения с адаптерами [LoRA](/docs/train/lora) задайте `lora=True` и установите extra `lora`. Модели RT-DETRv2 с повёрнутыми рамками доступны только для инференса. См. раздел [обучение](/docs/train), где описаны датасеты, аугментация и логгеры.

## Валидация и экспорт

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` возвращает обычный словарь. Доступны форматы экспорта ONNX, TorchScript, ExecuTorch, TensorRT и OpenVINO; в матрице экспорта на [странице модели](/docs/models/rt-detr) указано, какие из них проверены для каждой задачи. Экспортированные файлы `.onnx` и `.engine` можно снова загрузить через `LibreYOLO()`; они возвращают тот же объект `Results`. См. [руководства по экспорту](/docs/export) для каждого формата.

## Когда лучше использовать исходные репозитории

Если нужно воспроизвести результаты статьи с точными конфигурациями авторов, использовать версию Paddle или самостоятельно запустить дистилляцию учителя DINOv3 для RT-DETRv4, используйте исходные репозитории. LibreYOLO дообучает опубликованного студента v4, но не запускает учителя. Если проект уже распространяется под AGPL-3.0 или на него распространяется Ultralytics Enterprise License, причин переходить из-за лицензии нет.

## Примечание о лицензии

Код LibreYOLO распространяется под MIT. Каждый файл весов RT-DETR — под Apache-2.0, а в каждом репозитории весов на Hugging Face есть исходные файлы LICENSE и NOTICE, которые Apache-2.0 требует сохранять при распространении весов. Веса, обученные на собственных данных, принадлежат вам. RT-DETRv4 использует DINOv3 только как учителя во время обучения; в опубликованных моделях-студентах нет параметров DINOv3. Это общая информация, а не юридическая консультация. Перед выпуском проверьте актуальные файлы LICENSE.

## Попробуйте

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO распространяется под MIT, работает на Linux, Mac и Windows, а также на GPU, Apple Silicon и обычном CPU без изменений кода.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Документация RT-DETR](/docs/models/rt-detr)
