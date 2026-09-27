---
title: Пороги и фильтрация
seo_title: 'conf, iou и max_det в LibreYOLO'
description: >-
  Что на самом деле делают conf, iou, max_det и classes во время предсказания,
  какие семейства игнорируют iou, потому что не запускают NMS, и почему
  agnostic_nms ничего не делает.
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
last_verified: 1.5.0
verification: >-
  Значения по умолчанию взяты из InferenceRunner.__call__ в
  libreyolo/models/base/inference.py. Поведение NMS по семействам прочитано из
  всех модулей libreyolo/postprocess/ и сверено с _is_nms_free_family в
  libreyolo/backends/base.py. Фильтрация классов — из
  InferenceRunner._apply_classes_filter и _wrap_results. Статус agnostic_nms —
  из NOOP_PREDICT_KWARGS в libreyolo/utils/predict_args.py. Работа с открытым
  словарём — из NMS_THRESHOLD в libreyolo/models/openvocab/base.py. Значения по
  умолчанию для валидации — из BaseModel.val.
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
source_hash: 0b978963c356027d
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

Два параметра на этапе предсказания делают `iou` значимым даже для предиктора
множества, потому что оба объединяют рамки уже после того, как модель отработала:

- `tiling=True` согласует перекрывающиеся тайлы поклассовым NMS с порогом `iou`
- `augment=True` объединяет отражённые виды поклассовым NMS с порогом `iou`

Оба разобраны в разделе [Производительность инференса](/docs/predict/performance).

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
так устроена их исходная эталонная конфигурация. SSD останавливается на 200,
сегментация экземпляров RTMDet — на 100, а FCOS — на собственном лимите детекций
на изображение. Поднимать `max_det` выше этих значений бесполезно.

Единственное место, где `max_det` применяется централизованно, а не в каждом
семействе, — потайловый инференс: объединённый список обрезается после
согласования тайлов.

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
предсказание классов, которые вы потом выбрасываете, а `max_det` применяется
семейством до фильтра, поэтому изображение, забитое ненужными классами, может
упереться в предел раньше, чем дойдёт до вашего класса. Если так вышло, снизьте
`conf` или поднимите `max_det`.

## agnostic_nms

`agnostic_nms` принимается и ничего не делает. При передаче выдаётся
предупреждение о том, что это заглушка ради совместимости с командной строкой,
и сам аргумент отбрасывается.

Режима подавления без учёта классов нет. Каждый вызов NMS в библиотеке учитывает
класс, поэтому две перекрывающиеся рамки разных классов выживают обе, при любом
`iou`. Если это мешает, сначала отфильтруйте через `classes` или подавите
пересечения между классами сами, на `result.boxes`.

## Что predict отвергает

Два аргумента не предупреждают, а выбрасывают исключение: `visualize` и `embed` —
оба с `NotImplementedError`. Для эмбеддингов загрузите модель с
`task="embed"` и вызывайте `predict` или `embed` как обычно.

Всё нераспознанное выбрасывает `TypeError` с перечислением поддерживаемых опций,
так что опечатка сразу приводит к ошибке, а не игнорируется молча.

Эти аргументы принимаются, вызывают предупреждение и отбрасываются: `agnostic_nms`,
`boxes`, `dnn`, `half`, `line_width`, `retina_masks`, `show_conf`, `show_labels`
и `verbose`.
