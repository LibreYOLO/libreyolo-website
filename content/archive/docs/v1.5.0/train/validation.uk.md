---
title: Валідація та метрики
seo_title: Валідація та метрики в LibreYOLO
description: >-
  Запускайте val() для будь-якої моделі, читайте ключі метрик кожної задачі,
  вибирайте backend оцінювання та вмикайте функцію втрат валідації разом із
  метрикою правильності.
lead: >-
  Валідація запускає модель на частині датасету через val() і повертає плоский
  словник ключів метрик та чисел із рухомою крапкою. Ключі є буквальними
  рядками, а їхній набір залежить від задачі, а не від сімейства.
keywords:
  - map50-95
  - coco evaluation
  - метрики валідації
  - faster-coco-eval
  - pycocotools
  - функція втрат валідації
  - miou
  - panoptic quality
  - top1 accuracy
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
    - label: На іншій частині
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
    - label: Записати передбачення у форматі COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Запуск валідації

`val()` приймає датасет і повертає метрики.

<code-tabs name="val" />

Повертається звичайний `dict[str, float]`. Кожен ключ є буквальним, тому читайте
його за назвою, а не позицією.

Основними аргументами є `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` і `verbose`. `conf` типово дорівнює
`0.001`, а `iou` дорівнює `0.6`, що значно слабше за типові значення передбачення,
оскільки для перебору mAP потрібен хвіст низької впевненості. `imgsz` типово дорівнює
власному розміру вхідних даних моделі, а не фіксованому числу. `split` приймає лише
`val`, `test` або `train`.

Будь-яке інше поле конфігурації валідації передається як іменований аргумент, зокрема
`save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache` і `save_plots`.

## Ключі метрик для кожної задачі

Виявлення повертає сімейство показників COCO:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Два з них можуть ввести в оману. `metrics/precision` і `metrics/recall` є псевдонімами,
збереженими для зворотної сумісності: вони містять значення mAP 50-95 і AR@100,
а не пару точності й повноти. Використовуйте іменовані ключі.

Сегментація екземплярів повертає наведені вище значення mAP і AR для масок під
ключами без суфікса, версії рамок під суфіксом `(B)` і повторені версії масок
під суфіксом `(M)`. Для цієї задачі точність і повнота існують лише у формах
із суфіксом, `metrics/precision(B)`/`metrics/recall(B)` і
`metrics/precision(M)`/`metrics/recall(M)`, а обидві пари містять ті самі типи
псевдонімів, що й для виявлення: пара `(B)` є mAP50-95 рамок і AR@100 рамок,
а пара `(M)` є mAP50-95 масок і AR@100 масок.

| Задача | Ключі |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, а також наведені вище розподіли за розміром і повнотою |
| segment | версії наведених вище ключів detect для масок (ключі без суфікса стосуються масок); `precision`/`recall` існують лише як `(B)`/`(M)`, обидві пари мають ті самі типи псевдонімів |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L` і відповідні ключі `keypoints_AR` |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, а також копії із суфіксом `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, а також ключ перебору mAP |

`metrics/precision` і `metrics/recall` для OBB не є псевдонімами: це справжні точність
і повнота за IoU 0.50, узяті в найслабшій робочій точці (кожне передбачення, що пройшло
`conf`, типово `0.001`). Копії із суфіксом `(OBB)` повторюють ті самі чотири значення
під назвою конкретної задачі за тією самою домовленістю, що й `(B)` та `(M)` вище.

`accuracy_top5` насправді означає top-`min(5, num_classes)`, тому в датасеті з трьома
класами це top-3, якому відповідає кожен зразок, а отже значення дорівнює 1.0.

Ключ перебору задачі point формується з порогів відстані, тому за типових значень
він має назву `metrics/mAP@[0.01:0.10]`, а ключ одного порога має назву
`metrics/mAP@0.01`. Передавання `dist_thresholds` змінює обидва рядки.

Більшість задач також повертає ключ `fitness`, одне число, яке типово використовує
вибір найкращої контрольної точки. Виявлення, сегментація та OBB не мають цього ключа;
їхні сімейства вибираються за `metrics/mAP50-95`, який словники цих задач повертають.
Поза не повертає ані `fitness`, ані `metrics/mAP50-95`; натомість її засоби навчання
встановлюють `best_metric_key` у `metrics/keypoints_mAP50-95`.

## Ключі швидкості

Кожен валідатор додає часові показники:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Це мілісекунди на зображення, усереднені за запуском. Вони описують машину й використані
налаштування, тому значення має сенс лише разом з обладнанням, розміром батча й точністю.

## Backend оцінювання

Метрики виявлення й сегментації обчислюються через оцінювач COCO, а типове значення
`faster_coco_eval=True` вибирає backend C++, коли встановлено пакет `faster-coco-eval`.
Якщо його немає, запуск переходить до pycocotools з одним попередженням на процес:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Фактично використаний backend записується в модель як `last_eval_backend`, а CLI
повідомляє його у виводі для задач у стилі виявлення. Установіть
`LIBREYOLO_FASTER_COCO_EVAL`, щоб перевизначити значення конфігурації із середовища.

`iou_thresholds` враховується лише на шляху OBB. Шлях COCO оцінює за власним
фіксованим набором від 0.50 до 0.95 та ігнорує це значення.

## Функція втрат валідації

Типово валідація повідомляє лише правильність. `val_loss=True` також обчислює
навчальну цільову функцію сімейства на батчах валідації.

<code-tabs name="valloss" />

Вона породжує `metrics/loss` і один ключ `metrics/loss/<component>` для кожного
доданка, зваженого так само, як під час навчання, тому сума компонентів дорівнює
загальному значенню. Через логер вони відображаються як `val/loss` і
`val/loss/<component>`, а `libreyolo monitor` накладає `metrics/loss` на `train/loss`.

Компоненти належать самим сімействам:

| Задача | Сімейства | Компоненти |
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

Вона типово вимкнена, оскільки призначення цілей додає до валідації витрати часу
й пам'яті. Валідатор повторно використовує результат моделі, уже створений для метрики
правильності, замість другого прямого проходу, працює в контексті `no_grad` на моделі
оцінювання або EMA, а під час навчання на кількох GPU обчислюється локально на ранзі 0
без колективних операцій. Вибір найкращої контрольної точки й далі використовує
метрику правильності.

Три дії навмисно не виконуються. Функція ніколи не містить доданків контрастного
шумопригнічення, бо їм потрібна еталонна розмітка під час прямого проходу, а прямий
прохід валідації виконується без неї. Вона повідомляє результат моделі в режимі
оцінювання, тому там, де прямі проходи навчання й оцінювання сімейства справді
відрізняються статистикою BatchNorm або стохастичною глибиною, число відображає режим
оцінювання; саме таке порівняння й передбачено. Якщо сімейство не реалізувало цю
можливість для задачі, під час налаштування породжується помилка конфігурації замість
непомітного пропуску:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO є винятком, для якого нічого не змінюється: його валідатор завжди обчислював
цю функцію втрат, а `val_loss=True` впливає лише на ключі, під якими її опубліковано.

Аугментовану валідацію не можна поєднувати з функцією втрат валідації; одночасний
запит обох можливостей породжує помилку.

## Файли, які записує валідація

Метод `val()` завжди записує `config.yaml` до каталогу результатів, який типово має
назву `runs/val/<model>_<size>_<timestamp>`, якщо `save_dir` не задано.

<code-tabs name="json" />

За `save_json=True` для виявлення записується `predictions.json`, а для сегментації
записуються `predictions_bbox.json` і `predictions_masks.json`. OBB не підтримує
цю можливість і повідомляє про це.

За `save_plots=True` файли записуються до підкаталогу `plots/`. Для виявлення
створюються `box_metrics.png`, графіки AP і повноти для кожного класу, криві
точність-повнота й залежності від упевненості, матриця плутанини та анотовані
зображення-зразки, якщо встановлено OpenCV. Сегментація додає копії кожного графіка
для масок, а поза отримує власний набір метрик і кривих. Інші валідатори не реалізують
графіки; класифікація, семантична й паноптична сегментація, глибина, нормалі, краї,
відновлення, matting, OCR, OBB і point нічого туди не записують. Помилка побудови
графіка породжує попередження й ніколи не перериває запуск.

## Валідація під час навчання

Навчання виконує валідацію кожні `eval_interval` епох на частині `val` датасету,
а отримані метрики керують вибором `best.pt`, ранньою зупинкою `patience` і ключами
`val/` у кожному логері. За ввімкненого EMA валідація виконується на вагах EMA.

Опис `eval_interval`, `patience` і `save_plots` наведено в розділі
[Гіперпараметри](/docs/train/hyperparameters), а місця надсилання чисел описано
в розділі [Логери експериментів](/docs/train/loggers).

## Пов'язані матеріали

- [Датасети](/docs/train/datasets) описують ключі частин і формати, які читають валідатори.
