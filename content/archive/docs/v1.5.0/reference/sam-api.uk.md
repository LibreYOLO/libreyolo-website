---
title: API сегментації за підказками
seo_title: 'API LibreSAM: підказки, псевдоніми та сигнатури'
description: >-
  Фабрика LibreSAM, її псевдоніми розмірів, типи точкових, рамкових і текстових
  концептуальних підказок, життєвий цикл set_image з одноразовим кодуванням та
  обмеження цього рівня.
lead: >-
  LibreSAM є фабрикою сегментації за підказками. Для прямого проходу потрібна
  підказка для кожного зображення, яку надають під час виклику, тому цей рівень
  має власний інтерфейс передбачення замість маршрутизації через засіб інференсу
  без підказок.
keywords:
  - LibreSAM
  - сегментація за підказками
  - SAM точкова підказка
  - SAM підказка рамкою
  - set_image
  - сегментувати все зображення
  - libreyolo sam extra
last_verified: 1.5.0
verification: >-
  Псевдоніми фабрики, розміри й репозиторії звірено з
  libreyolo/models/sam/model.py, sam2.py, edgetam.py, sam3.py,
  libreyolo/models/mobilesam/model.py і libreyolo/models/picosam3/model.py.
  Контракт підказок і типові значення звірено з libreyolo/models/sam/base.py.
  Проєктні рішення взято з docs/adr/0007-libresam-contract.md, усе для версії
  v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Точкові та рамкові підказки
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Одноразове кодування, багато підказок'
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

## Встановлення

Цей рівень потребує додаткового набору залежностей `sam`.

<code-tabs name="install" />

## Фабрика

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` є псевдонімом розміру, а не шляхом. `**kwargs` передається конструктору
сімейства, який приймає `device` і `multimask`. Невідомий псевдонім спричиняє
`ValueError`, а повідомлення містить усі відомі псевдоніми.

<code-tabs name="usage" />

## Псевдоніми

| Сімейство | Псевдоніми | Розміри | Ваги |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large` і короткі форми `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

Типово використовується `base`. SAM-1, SAM-2, EdgeTAM і MobileSAM працюють на
номінальному полотні 1024 пікселі, SAM 3 на 1008, а PicoSAM3 на 96.

Доступ до ваг SAM 3 обмежено. Їх завантажують із `facebook/sam3` за спеціальною
ліцензією SAM від Meta, яка не є ані MIT, ані Apache-2.0; LibreYOLO не
розповсюджує ці ваги. Прийміть умови на сторінці репозиторію та автентифікуйтеся
в Hugging Face перед завантаженням; завантажувач спочатку записує це
повідомлення до журналу.

Класи сімейств також експортуються, тому `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` і `LibrePicoSAM3` можна
створювати безпосередньо з `size=`.

## Передбачення

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

| Аргумент | Типове значення | Значення |
|---|---|---|
| `source` | `None` | Зображення для сегментації; `None` повторно використовує зображення, кешоване через `set_image()` |
| `points` | `None` | Точкова підказка в піксельних координатах |
| `bboxes` | `None` | Рамкова підказка у вигляді `[x1, y1, x2, y2]` або їх список для отримання однієї маски на рамку |
| `labels` | `None` | Мітки точок, `1` для додатних і `0` для від'ємних, із формою відповідно до `points`; якщо їх пропущено, усі точки додатні |
| `masks` | `None` | Зарезервовано; передавання значення спричиняє `NotImplementedError` |
| `text` | `None` | Концептуальна підказка; лише SAM 3 |
| `conf` | `None` | Нижня межа передбаченого IoU маски |
| `multimask` | `None` | Повернути всі неоднозначні маски для кожної підказки; типово використовується налаштування конструктора |
| `max_det` | `300` | Обмеження кількості повернених масок |
| `device` | `None` | Перемістити модель для цього й наступних викликів, скинувши кешовані ембединги |
| `color_format` | `"auto"` | Підказка щодо формату кольорів для масивів у пам'яті |
| `points_per_side` | `None` | Щільність сітки для сегментації всього зображення; типове значення 32 |

Повертається звичайний об'єкт `Results` із `masks` і щільно припасованими
`boxes`, отриманими з цих масок, де клас `0` має назву `"object"`.

## Форми підказок

`points` приймає вкладені форми `[x, y]` для одного об'єкта, `[[x, y], ...]`
для N об'єктів і `[[[x, y], ...], ...]` для точок, згрупованих за об'єктами.
Масиви Numpy працюють скрізь, де працюють списки. Координати задано звичайними
пікселями на початковому зображенні.

Якщо пропустити всі просторові підказки, запускається сегментація всього
зображення: автоматичний генератор масок за сіткою з порогом передбаченого IoU
і усуненням дублікатів за IoU рамок. Типове значення `points_per_side`, що
дорівнює 32, запускає приблизно 1024 проходи декодера, тому на CPU це повільно;
для інтерактивного використання зменште значення. Генератор не застосовує
фільтрацію за оцінкою стабільності, багаторазове кадрування та усунення
дублікатів за IoU масок, тому він наближує шлях із підказками, але не збігається
з ним.

## Впевненість

`conf` фільтрує за передбаченим IoU маски, який є оцінкою якості маски, а не
впевненістю виявлення. `None` зберігає всі маски на шляху з підказками й
застосовує поріг сітки сімейства під час сегментації всього зображення. `0.0`
вимикає фільтрацію в обох режимах.

На текстовому шляху SAM 3 параметр `conf` натомість є оцінкою виявлення
Promptable Concept Segmentation. Тут `None` означає стандартний поріг 0.3, а
`0.0` зберігає всіх кандидатів.

## Текстові підказки

`text=` підтримує лише SAM 3; для нього кожне сімейство з просторовими
підказками спричиняє `NotImplementedError`. Текст не можна поєднувати з точками
та рамками. Повернене поле `names` зіставляє клас `0` із запитаною концепцією.
Текстовий виклик із `source=None` повторно кодує кешоване зображення, оскільки
трекер і кодувальник концепцій не мають спільного кешу.

Ключове слово `exemplars=` зарезервовано для майбутнього розширення з
прикладами зображень і ще не реалізовано.

## Життєвий цикл з одноразовим кодуванням

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` один раз запускає ресурсомісткий кодувальник зображення та кешує
ембединги, тому кожен наступний виклик `predict()` із `source=None` потребує
мало ресурсів. Обидва методи повертають модель, тож виклики можна об'єднувати в
ланцюжок. Передавання `device=` до `predict` переміщує модель і скидає кеш.

## PicoSAM3

PicoSAM3 приймає лише `bboxes=`. Підказки точками, текстом, маскою чи
мультимаскою, а також сегментація всього зображення спричиняють помилку. Рамка
розширюється на 10 відсотків і обробляється мережею ROI на 96 пікселів;
PicoSAM3 є єдиним сімейством цього рівня, яке підтримує експорт, і лише у
формат ONNX.

## Непідтримувані можливості

`train()`, `val()` і `track()` спричиняють `NotImplementedError` для кожного
сімейства цього рівня. Маски за підказками не мають фіксованого набору класів
для оцінювання, тому mAP тут не має змісту. `export()` спричиняє помилку для
SAM-1, SAM-2, SAM 3, EdgeTAM і MobileSAM.

Шляхи відео та пам'яті для SAM-2, SAM 3 і EdgeTAM не входять до цієї версії,
як і приклади зображень SAM 3 та підказки масками.
