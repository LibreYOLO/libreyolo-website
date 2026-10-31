---
title: Пороги и фильтрация
seo_title: 'conf, iou и max_det в LibreYOLO'
description: >-
  Что делают conf, iou, max_det, classes и agnostic_nms во время предсказания и
  какие семейства игнорируют iou, потому что не запускают собственный NMS.
lead: >-
  Какие предсказания выживут, решают четыре аргумента: conf, iou, max_det и
  classes. Только два из них применимы к каждому семейству, потому что предиктор
  множества декодирует фиксированный набор запросов и никогда не запускает NMS.
keywords:
  - порог уверенности yolo
  - conf yolo python
  - порог iou nms
  - max_det yolo
  - фильтрация классов детекция python
  - agnostic nms
  - detr без nms
  - фильтрация классов при инференсе
last_verified: 1.6.0
verification: >-
  Значения по умолчанию взяты из InferenceRunner.__call__ в
  libreyolo/models/base/inference.py. Поведение NMS по семействам прочитано из
  всех модулей libreyolo/postprocess/ и сверено с _is_nms_free_family в
  libreyolo/backends/base.py. Фильтрация классов — из
  InferenceRunner._apply_classes_filter и _wrap_results. agnostic_nms — из
  check_agnostic_nms и postprocess_max_det в libreyolo/utils/predict_args.py и
  шага agnostic в InferenceRunner._wrap_results. Работа с открытым словарём — из
  NMS_THRESHOLD в libreyolo/models/openvocab/base.py. Значения по умолчанию для
  валидации — из BaseModel.val.
snippets:
  basic:
    - label: Четыре аргумента
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # оставить предсказания с этой оценкой и выше
            iou=0.45,       # порог перекрытия NMS — там, где NMS запускается
            max_det=300,    # предел на изображение
            classes=None,   # или список id классов
        )
        print(len(result.boxes))
    - label: Перебор conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Фильтрация по конкретным классам
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Id классов индексируют model.names. В COCO 0 — это person.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Поиск id по имени
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: 'iou в семействе, которое не запускает NMS'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # RF-DETR декодирует фиксированный набор запросов, поэтому iou здесь
        ничего не меняет.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # Количество одинаковое в обоих случаях. Работают именно conf и max_det.

        print(len(loose.boxes), len(tight.boxes))
  agnostic:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        per_class = model(SAMPLE_IMAGE)

        agnostic = model(SAMPLE_IMAGE, agnostic_nms=True)


        # Перекрывающиеся рамки разных классов сворачиваются в одну с наибольшей
        оценкой.

        print(len(per_class.boxes), len(agnostic.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt agnostic_nms=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: fa0e6cee02b3e336
---

## Четыре аргумента

| Аргумент | По умолчанию | Применяется |
|---|---|---|
| `conf` | `0.25` | К каждому семейству |
| `iou` | `0.45` | К семействам, которые запускают немаксимальное подавление (NMS) |
| `max_det` | `300` | К каждому семейству |
| `classes` | `None` | К каждому семейству |

<code-tabs name="basic" />

Два из них универсальны, а два нет — это самое полезное, что стоит знать до
того, как что-то настраивать.

У валидации значения по умолчанию другие, и это сделано намеренно: `val()`
работает при `conf=0.001` и `iou=0.6`, потому что средняя точность считается по
полной кривой точности и полноты, а отсечка на 0.25 её обрезала бы.

## conf

`conf` — это оценка, ниже которой предсказание отбрасывается. Он применяется к
каждому семейству, включая те, что никогда не запускают NMS, и это первая ручка,
за которую стоит браться, когда детекций слишком много или слишком мало.

Значение по умолчанию `0.25` подходит, чтобы просто смотреть на картинки. Если
предсказания уходят в следующую систему, обычно нужно значение выше; если
измеряется качество — гораздо ниже.

Валидация детекции предоставляет `metrics/best_conf` и пороги по классам на основе F1 при IoU 0.50. Используйте их как измеренные отправные значения порогов при развёртывании; см. [валидацию](/docs/train/validation). Сегментация не предоставляет эти ключи.

## iou

`iou` — это перекрытие, выше которого немаксимальное подавление убирает ту из двух
рамок одного класса, у которой оценка ниже. Он что-то значит, только если
семейство вообще запускает подавление.

Предиктор множества декодирует фиксированное число запросов и берёт те, у
которых оценка выше. Дубликаты подавляются внутри архитектуры во время обучения,
а не на шаге постобработки, поэтому никакого порога здесь нет. Эти семейства
принимают `iou` ради совместимости API и игнорируют его:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR и end-to-end голова YOLOv9.
Варианты, построенные на этих декодерах, наследуют это поведение.

<code-tabs name="nmsfree" />

Большинство из них сообщают об этом в строках документации к постобработке, но
во время выполнения предупреждение не выдаётся, поэтому перебор `iou` на RF-DETR
даёт ровную линию, а не ошибку. Faster R-CNN и Mask R-CNN — случай чуть другой:
обе уже запустили NMS внутри модели, с фиксированным порогом из исходной
реализации, изменить который через `iou` штатным способом нельзя.

А эти семейства его используют: от YOLOv1 до YOLOv4, YOLOv7, YOLOv9, YOLOX,
YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet и SSD.

Три параметра на этапе предсказания делают `iou` значимым даже для предиктора
множества, потому что все три действуют на рамки уже после того, как модель
отработала:

- `tiling=True` согласует перекрывающиеся тайлы поклассовым NMS с порогом `iou`
- `augment=True` объединяет отражённые виды поклассовым NMS с порогом `iou`
- `agnostic_nms=True` подавляет перекрывающиеся рамки между классами с порогом
  `iou`, см. [ниже](#agnostic_nms)

Первые два разобраны в разделе [Производительность инференса](/docs/predict/performance).

У детекторов с открытым словарём своё правило. Семейство, чей процессор
запускает NMS, объявляет собственный порог по умолчанию и учитывает `iou` — так
устроен OMDet-Turbo. Семейства, которые ничего не подавляют, — Grounding DINO,
OWLv2 и OV-DEIM — выдают предупреждение, если передать `iou`. Это единственное
предупреждение такого рода во всей библиотеке.

## max_det

`max_det` ограничивает, сколько предсказаний возвращается для одного
изображения. Он работает везде, но через разные механизмы: семейство с NMS
обрезает список после подавления, а предиктор множества использует его как
размер выборки top-k.

Некоторые семейства ограничивают результат сильнее, чем вы просите, потому что
так устроена их исходная эталонная конфигурация. SSD останавливается на 200, а
FCOS — на собственном лимите детекций на изображение. Поднимать `max_det` выше
этих значений бесполезно.

`max_det` применяется централизованно, а не в каждом семействе, в двух случаях:
при потайловом инференсе, где объединённый список обрезается после согласования
тайлов, и при вызовах с `classes` или `agnostic_nms=True`, где обрезка идёт после
фильтра или подавления.

## Фильтрация классов

<code-tabs name="classes" />

`classes` принимает список id классов и оставляет только те предсказания, чей
класс есть в списке. Id индексируют `result.names`, и надёжнее всего взять
нужный, прочитав `names` из результата, а не предполагая порядок классов в
датасете.

Фильтрация происходит централизованно, после постобработки каждого семейства, в
единой точке, через которую проходит любой путь предсказания. Из этого следуют
две вещи, которые стоит знать. Она работает на каждом семействе, включая те, где
NMS нет. И заодно она фильтрует данные, привязанные к рамкам, так что маски,
ключевые точки и повёрнутые рамки урезаются вместе с ними, а не остаются
рассогласованными.

В командной строке `classes` принимает голое целое число, список или строку со
значениями через запятую:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Фильтрация не даёт качества бесплатно. Модель всё равно тратит свой бюджет на
предсказание классов, которые вы потом выбрасываете. Если задан `classes`,
семейство держит бюджет кандидатов в 300 рамок или `max_det`, если он больше, а
обрезка до `max_det` идёт после фильтра, поэтому малый `max_det` не вытесняет ваш
класс. Экспортированные бэкенды используют тот же бюджет, включая батчевые
движки TensorRT. Изображение, где больше 300 рамок ненужных классов с более
высокой оценкой, всё равно может заполнить бюджет раньше, чем дойдёт до вашего
класса; если так вышло, поднимите `max_det` выше 300.

## agnostic_nms

NMS каждого семейства учитывает класс, поэтому по умолчанию две перекрывающиеся
рамки разных классов выживают обе, при любом `iou`. `agnostic_nms=True` добавляет
один шаг без учёта классов: из рамок, перекрывающихся сильнее `iou`, выживает
только рамка с наибольшей оценкой, независимо от их классов.

<code-tabs name="agnostic" />

Шаг работает на готовых детекциях семейства, после фильтра `classes` и до
обрезки до `max_det`. Поэтому он ведёт себя одинаково на предикторах множества,
которые не запускают собственный NMS, на экспортированных бэкендах и при
`augment=True` и `tiling=True`. Маски и ключевые точки отбрасываются вместе со
своей рамкой, а повёрнутые рамки сравниваются по повёрнутому IoU. Он охватывает
задачи, которые возвращают рамки: детекцию, сегментацию, позу и obb.

Для шага семейство держит бюджет кандидатов в 300 рамок или `max_det`, если он
больше. При `max_det` ниже 300 кандидаты с меньшей оценкой заполняют места,
которые освободило подавление. При 300 и больше, включая значение по умолчанию,
эти места остаются пустыми.

Где он не применяется:

- На стандартном пути predict задача без рамок вызывает `ValueError`.
- `LibreEnsemble` и [HRNet](/docs/models/hrnet) вызывают `NotImplementedError`.
- Модели `LibreSAM` и [3D-детекторы](/docs/tasks/3d-object-detection),
  WildDet3D, 3D-MOOD, FCOS3D и DetAny3D, не принимают такого аргумента и
  вызывают `TypeError`.
- [L2CS-Net](/docs/models/l2cs), [PP-OCRv5](/docs/models/pp-ocrv5),
  [SAM 3D Body](/docs/models/sam-3d-body) и
  [LibreFaceRec](/docs/models/librefacerec) идут по собственному пути predict и
  игнорируют его.

`val(agnostic_nms=True)` применяет тот же шаг к валидации детекции и сегментации;
см. [Валидация и метрики](/docs/train/validation).

## Что predict отвергает

Два аргумента не предупреждают, а выбрасывают исключение: `visualize` и `embed` —
оба с `NotImplementedError`. Для эмбеддингов загрузите модель с
`task="embed"` и вызывайте `predict` или `embed` как обычно. `agnostic_nms=True`
вызывает исключение там, где не применяется, как перечислено выше.

Всё нераспознанное выбрасывает `TypeError` с перечислением поддерживаемых опций,
так что опечатка сразу приводит к ошибке, а не игнорируется молча.

Эти аргументы принимаются, вызывают предупреждение и отбрасываются: `boxes`, `dnn`, `half`,
`line_width`, `retina_masks`, `show_conf`, `show_labels` и `verbose`.
