---
title: API сегментации по промптам
seo_title: 'LibreSAM API: промпты, алиасы и сигнатуры'
description: >-
  Фабрика LibreSAM, её алиасы размеров, промпты точкой, рамкой и
  концепт-текстом, жизненный цикл set_image с однократным кодированием и то,
  чего уровень не поддерживает.
lead: >-
  LibreSAM — фабрика для сегментации по промптам. Прямому проходу нужен промпт,
  который задаётся для каждого изображения в момент вызова, поэтому у уровня
  собственный интерфейс predict, а не общий раннер инференса, который промпты не
  принимает.
keywords:
  - LibreSAM
  - сегментация по промптам python
  - SAM промпт точкой
  - SAM промпт рамкой
  - set_image
  - segment everything
  - libreyolo sam extra
last_verified: 1.5.0
verification: >-
  Алиасы фабрики, размеры и репозитории прочитаны из
  libreyolo/models/sam/model.py, sam2.py, edgetam.py, sam3.py,
  libreyolo/models/mobilesam/model.py и libreyolo/models/picosam3/model.py.
  Контракт промптов и значения по умолчанию прочитаны из
  libreyolo/models/sam/base.py. Замысел из docs/adr/0007-libresam-contract.md,
  всё на версии v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Промпты точкой и рамкой
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Одно кодирование, много промптов'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Установка

Этому уровню нужен extra `sam`.

<code-tabs name="install" />

## Фабрика

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` — это алиас размера, а не путь. `**kwargs` передаётся в конструктор
семейства, который принимает `device` и `multimask`. Неизвестный алиас вызывает
`ValueError`, и в сообщении перечислены все известные алиасы.

<code-tabs name="usage" />

## Алиасы

| Семейство | Алиасы | Размеры | Веса |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, а также короткие формы `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

По умолчанию берётся `base`. SAM-1, SAM-2, EdgeTAM и MobileSAM работают на
номинальном холсте в 1024 пикселя, SAM 3 — на 1008, PicoSAM3 — на 96.

Доступ к весам SAM 3 ограничен. Они скачиваются с `facebook/sam3` под
собственной лицензией Meta SAM License — это не MIT и не Apache-2.0, и
LibreYOLO эти веса не перераспространяет. Примите условия на странице
репозитория и авторизуйтесь в Hugging Face перед загрузкой; загрузчик сначала
выводит в лог соответствующее уведомление.

Классы семейств тоже экспортируются, поэтому `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` и `LibrePicoSAM3` можно создавать
напрямую с `size=`.

## predict

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Параметр | По умолчанию | Смысл |
|---|---|---|
| `source` | `None` | Изображение для сегментации; `None` переиспользует изображение, закэшированное `set_image()` |
| `points` | `None` | Промпт точкой в пиксельных координатах |
| `bboxes` | `None` | Промпт рамкой в виде `[x1, y1, x2, y2]` или список рамок, по одной маске на рамку |
| `labels` | `None` | Метки точек, `1` — положительная, `0` — отрицательная, по форме совпадают с `points`; если не задать, все положительные |
| `masks` | `None` | Зарезервировано; передача вызывает `NotImplementedError` |
| `text` | `None` | Промпт-концепт; только SAM 3 |
| `conf` | `None` | Нижняя граница предсказанного mask-IoU |
| `multimask` | `None` | Возвращать на каждый промпт все маски неоднозначности; по умолчанию берётся настройка, заданная при создании |
| `max_det` | `300` | Предел числа возвращаемых масок |
| `device` | `None` | Переносит модель для этого и последующих вызовов, сбрасывая закэшированные эмбеддинги |
| `color_format` | `"auto"` | Подсказка о цветовом формате для массивов в памяти |
| `points_per_side` | `None` | Плотность сетки для режима «сегментировать всё»; по умолчанию 32 |

Возвращается обычный `Results` с `masks` и плотно прилегающими `boxes`,
выведенными из этих масок; класс `0` называется `"object"`.

## Формы промптов

`points` принимает вложенные формы `[x, y]` для одного объекта,
`[[x, y], ...]` для N объектов и `[[[x, y], ...], ...]` для точек,
сгруппированных по объектам. Массивы numpy работают везде, где работает список.
Координаты — обычные пиксели на исходном изображении.

Если не задать ни одного пространственного промпта, запускается режим
«сегментировать всё» — автоматический генератор масок по сетке с порогом по
предсказанному IoU и дедупликацией по box-IoU. Значение `points_per_side` по
умолчанию, 32, даёт примерно 1024 прохода декодера, что медленно на CPU; для
интерактивной работы его стоит понизить. В генераторе нет фильтрации по
stability score, multi-crop и дедупликации по mask-IoU, поэтому он — приближение
к пути с промптом, а не точное соответствие ему.

## Уверенность

`conf` фильтрует по предсказанному mask-IoU — это оценка качества маски, а не
уверенность детекции. `None` оставляет все маски на пути с промптом и применяет
сеточный порог семейства в режиме «сегментировать всё». `0.0` отключает
фильтрацию в обоих режимах.

На текстовом пути SAM 3 `conf` вместо этого — оценка детекции Promptable
Concept Segmentation. `None` здесь означает стандартный порог 0.3, а `0.0`
оставляет всех кандидатов.

## Текстовые промпты

`text=` работает только у SAM 3; у всех семейств с пространственными промптами
он вызывает `NotImplementedError`. Текст несовместим с точками и рамками.
Возвращаемый `names` сопоставляет класс `0` с запрошенным концептом. Вызов с
текстом и `source=None` заново кодирует закэшированное изображение, потому что
трекер и энкодер концептов не используют общий кэш.

Ключевое слово `exemplars=` зарезервировано под будущее расширение с
образцами-изображениями и не реализовано.

## Жизненный цикл с однократным кодированием

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` один раз прогоняет тяжёлый энкодер изображения и кэширует
эмбеддинги, поэтому каждый последующий `predict()` с `source=None` обходится
дёшево. Оба метода возвращают модель, так что вызовы можно объединять в
цепочку. Передача `device=` в `predict` переносит модель и сбрасывает кэш.

## PicoSAM3

PicoSAM3 принимает только `bboxes=`. Промпты точкой, текстом, маской, промпт с
multimask и режим «сегментировать всё» вызывают исключение. Рамка расширяется
на 10 процентов и прогоняется через ROI-сеть на 96 пикселей, и PicoSAM3 —
единственное семейство уровня, которое экспортируется, причём только в ONNX.

## Что не поддерживается

`train()`, `val()` и `track()` вызывают `NotImplementedError` у каждого
семейства уровня. У масок по промпту нет фиксированного набора классов, по
которому можно считать метрику, поэтому mAP здесь не имеет смысла. `export()`
вызывает исключение для SAM-1, SAM-2, SAM 3, EdgeTAM и MobileSAM.

Видео- и memory-пути SAM-2, SAM 3 и EdgeTAM выходят за рамки этой версии, как и
образцы-изображения SAM 3 и промпты масками.
