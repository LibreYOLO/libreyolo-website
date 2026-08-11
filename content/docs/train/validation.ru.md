---
title: Валидация и метрики
seo_title: Валидация и метрики в LibreYOLO
description: >-
  Запуск val() на любой модели, ключи метрик, которые возвращает каждая задача,
  выбор бэкенда для оценки и включение функции потерь на валидации рядом с
  метрикой качества.
lead: >-
  Валидация прогоняет модель по сплиту датасета через val() и возвращает плоский
  словарь: ключи метрик и значения float. Ключи — буквальные строки, и набор
  зависит от задачи, а не от семейства.
keywords:
  - map50-95
  - оценка модели на coco
  - метрики валидации yolo
  - faster-coco-eval
  - pycocotools
  - функция потерь на валидации
  - miou семантическая сегментация
  - panoptic quality pq
  - top-1 accuracy классификация
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: На другом сплите
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Запись предсказаний в формате COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Запуск валидации

`val()` принимает датасет и возвращает метрики.

<code-tabs name="val" />

Возвращается обычный `dict[str, float]`. Каждый ключ буквальный, поэтому читать
его нужно по имени, а не по позиции.

Основные аргументы — `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` и `verbose`. По умолчанию `conf`
равен `0.001`, а `iou` — `0.6`; оба заметно свободнее, чем значения по
умолчанию при предсказании, потому что для расчёта mAP по всем порогам нужен
хвост с низкой уверенностью. `imgsz` по умолчанию берёт собственный входной
размер модели, а не фиксированное число. `split` принимает `val`, `test` или
`train` и ничего больше.

Любое другое поле конфига валидации передаётся как именованный аргумент,
включая `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache` и
`save_plots`.

## Ключи метрик по задачам

Детекция возвращает набор чисел из COCO:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Два из них — ловушка. `metrics/precision` и `metrics/recall` — псевдонимы,
оставленные для обратной совместимости: в них лежат значения mAP 50-95 и AR@100,
а не пара точности и полноты. Используйте именованные ключи.

Сегментация экземпляров возвращает те же числа mAP и AR, что выше, но по
маскам — под ключами без суффикса; версии по рамкам идут с суффиксом `(B)`, а
версии по маскам продублированы с `(M)`. Точность и полнота у этой задачи
существуют только в суффиксной форме, как `metrics/precision(B)`/`metrics/recall(B)`
и `metrics/precision(M)`/`metrics/recall(M)`, и обе пары несут те же
значения-псевдонимы, что и у detect: пара `(B)` — это mAP50-95 и AR@100 по
рамкам, пара `(M)` — mAP50-95 и AR@100 по маскам.

| Задача | Ключи |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, плюс перечисленные выше разбивки по размерам и по полноте |
| segment | версии перечисленных выше ключей detect по маскам (ключи без суффикса — по маскам); `precision`/`recall` есть только как `(B)`/`(M)`, обе пары с тем же псевдонимом |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L` и соответствующие ключи `keypoints_AR` |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, плюс копии с суффиксом `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, плюс ключ с mAP по диапазону порогов |

У OBB `metrics/precision` и `metrics/recall` — не псевдонимы: это настоящие
точность и полнота при IoU 0.50, взятые в самой свободной рабочей точке (каждое
предсказание, прошедшее `conf`, по умолчанию `0.001`). Копии с суффиксом
`(OBB)` повторяют те же четыре значения под специфичным для задачи именем — то
же соглашение, что и у `(B)` и `(M)` выше.

`accuracy_top5` на самом деле считается как top-`min(5, num_classes)`, поэтому
на датасете с тремя классами это top-3 — условие, которому удовлетворяет каждый
пример и которое поэтому читается как 1.0.

Ключ с mAP по диапазону у задачи point собирается из порогов по расстоянию,
поэтому со значениями по умолчанию он выглядит как `metrics/mAP@[0.01:0.10]`, а
ключ по одному порогу — как `metrics/mAP@0.01`. Передача `dist_thresholds`
меняет обе строки.

Большинство задач возвращают ещё и ключ `fitness` — единственное число, по
которому по умолчанию выбирается лучший чекпойнт. У детекции, сегментации и OBB
его нет: их семейства отбираются по `metrics/mAP50-95`, который в их словарях
как раз есть. Задача pose не возвращает ни `fitness`, ни `metrics/mAP50-95`; её
обучающие классы вместо этого выставляют `best_metric_key` в
`metrics/keypoints_mAP50-95`.

## Ключи скорости

Каждый валидатор добавляет тайминги:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Это миллисекунды на изображение, усреднённые по всему прогону. Они описывают
машину и настройки, на которых был запуск, поэтому взятое из них число имеет
смысл только вместе с железом, размером батча и точностью вычислений.

## Бэкенд для оценки

Метрики детекции и сегментации считаются через оценщик COCO, и
`faster_coco_eval=True`, значение по умолчанию, выбирает C++-бэкенд, если пакет
`faster-coco-eval` установлен. Если нет, прогон откатывается на pycocotools с
одним предупреждением на процесс:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Какой бэкенд отработал на самом деле, записывается в модель как
`last_eval_backend`, а CLI сообщает это в своём выводе для задач детекционного
типа. Задайте `LIBREYOLO_FASTER_COCO_EVAL`, чтобы переопределить значение из
конфига через окружение.

`iou_thresholds` учитывается только на пути OBB. Путь COCO считает по
собственному фиксированному набору порогов от 0.50 до 0.95 и это значение
игнорирует.

## Функция потерь на валидации

По умолчанию валидация сообщает только метрики качества. `val_loss=True`
дополнительно считает обучающую целевую функцию семейства на батчах валидации.

<code-tabs name="valloss" />

Выдаётся `metrics/loss` плюс по одному `metrics/loss/<component>` на каждое
слагаемое, с ровно теми же весами, что и при обучении, поэтому компоненты в
сумме дают общее значение. Через логгер они видны как `val/loss` и
`val/loss/<component>`, а `libreyolo monitor` накладывает `metrics/loss` на
`train/loss`.

Компоненты у каждого семейства свои:

| Задача | Семейства | Компоненты |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

По умолчанию это выключено, потому что назначение таргетов добавляет валидации
время и память. Валидатор переиспользует вывод модели, уже полученный для
метрики качества, вместо второго прямого прохода, работает под `no_grad` на
оценочной или EMA-модели, а при обучении на нескольких GPU считается локально
на rank 0 без коллективных операций. Выбор лучшего чекпойнта остаётся по метрике
качества.

Три вещи он намеренно не делает. Он никогда не включает слагаемые contrastive
denoising, потому что им нужна эталонная разметка (ground truth) на прямом
проходе, а валидация делает проход без неё. Он сообщает числа для модели в
режиме оценки, поэтому там, где прямые проходы семейства в обучении и в оценке
действительно различаются — статистикой BatchNorm или stochastic depth, — число
отражает режим оценки; именно такое сравнение и задумано. И задача, для которой
семейство это не реализовало, вызывает ошибку конфигурации на этапе подготовки,
а не тихо пропускается:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO — исключение, которое ничего не меняет: его валидатор всегда считал эту
функцию потерь, и `val_loss=True` влияет только на то, под какими ключами она
публикуется.

Валидацию с аугментациями и функцию потерь на валидации совмещать нельзя, и
запрос обоих сразу вызывает ошибку.

## Файлы, которые записывает валидация

`val()` всегда пишет `config.yaml` в свой каталог сохранения; если `save_dir`
не задан, это `runs/val/<model>_<size>_<timestamp>`.

<code-tabs name="json" />

`save_json=True` пишет `predictions.json` для детекции и `predictions_bbox.json`
вместе с `predictions_masks.json` для сегментации. OBB это не поддерживает и
прямо об этом сообщает.

`save_plots=True` пишет в подкаталог `plots/`. Для детекции это
`box_metrics.png`, графики AP и полноты по классам, кривые precision-recall и
уверенности, матрица ошибок, а при установленном OpenCV — ещё и размеченные
примеры изображений. Сегментация добавляет к каждому из них копию по маскам, а у
pose свой набор метрик и кривых. Остальные валидаторы графики не реализуют:
классификация, semantic, panoptic, depth, normal, edge, restore, matte, OCR, OBB
и point не пишут туда ничего. Сбой при построении графиков выдаёт предупреждение
и никогда не прерывает прогон.

## Валидация во время обучения

Обучение запускает валидацию каждые `eval_interval` эпох на сплите `val`
датасета, и именно её метрики управляют выбором `best.pt`, ранней остановкой по
`patience` и ключами `val/` в каждом логгере. Если EMA включена, валидация идёт
по весам EMA.

Про `eval_interval`, `patience` и `save_plots` — в разделе
[Гиперпараметры](/docs/train/hyperparameters), а про то, куда уходят числа, — в
разделе [Логгеры экспериментов](/docs/train/loggers).

## Смотрите также

- [Датасеты](/docs/train/datasets) — про ключи сплитов и форматы, которые читают валидаторы.
