---
title: Виявлення країв
seo_title: Виявлення країв у LibreYOLO
description: >-
  Передбачте щільну карту ймовірностей країв за одним зображенням у LibreYOLO.
  Перетворіть контрольну точку, застосуйте поріг до карти, виконайте валідацію
  за ODS і OIS та експортуйте модель.
lead: >-
  Виявлення країв передбачає ймовірність того, що кожен піксель лежить на межі
  об'єкта. LibreYOLO надає його як задачу edge, що повертає щільну карту
  ймовірностей на полотні початкового зображення замість набору відрізків.
keywords:
  - виявлення країв python
  - нейромережа для меж об'єктів
  - карта ймовірностей країв
  - ODS OIS F-міра
  - передбачення країв зображення
last_verified: 1.5.0
snippets:
  predict:
    - label: Передбачити карту країв
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # LibreYOLO не постачає контрольну точку країв; спочатку перетворіть її
        (нижче).

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)          # (H, W) float32 у [0, 1]

        print(edges.binary(0.5).sum())    # кількість пікселів краю за порога
        0.5
    - label: Вибрати власний поріг
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # Безперервна карта зберігається, щоб ви самі вибрали поріг.
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Зберегти візуалізацію
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot() візуалізує карту; його визначено для країв і нормалей.
        result.plot().save("edges.png")
  val:
    - label: Виконати валідацію та прочитати ключі метрик
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Змінити діапазон порогів і допуск зіставлення
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Експорт
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Запустити експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файла, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        Results.

        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## Визначення

Задача `edge` передбачає одну ймовірність для кожного пікселя одного зображення
RGB: `0` означає не край, а `1` означає край. Карта залишається безперервною,
тому поріг перетворення на бінарне зображення меж вибирає код виклику, а
правильний поріг залежить від датасету та подальшого використання.

Передбачення заповнює `result.edges`, об'єкт даних `EdgeMap` із масивом float32
у формі `(H, W)` у діапазоні `[0, 1]` на полотні початкового зображення.
`.array` повертає цю карту як NumPy, а `.binary(threshold)` повертає булеву
маску. Поле `result.boxes` залишається порожнім, тому `conf`, `iou` і `max_det`
не мають впливу. `Results.plot()` підтримує цю задачу й безпосередньо
візуалізує карту.

## Моделі

Задачу `edge` виконують три сімейства.

[DexiNed](/docs/models/dexined), Dense Extreme Inception Network, об'єднує
кілька бічних виходів в одну карту ймовірностей і працює з нативним розміром
352 пікселі.

[TEED](/docs/models/teed), Tiny and Efficient Edge Detector, є малою мережею з
тим самим нативним розміром 352 пікселі та кроком зменшення роздільної здатності
4 проти 16 у DexiNed, тому приймає більше значень `imgsz`.

[LibreMODUS](/docs/models/libremodus) створює краї в стилі Canny як одну з цілей
моделі перетворення довільної модальності на довільну. Для нього потрібний
набір `modus` і ваш автентифікований обліковий запис Hugging Face; він не
підтримує ні `val()`, ні `export()`, тому не бере участі в наведених нижче
розділах валідації та експорту.

## Передбачення

LibreYOLO не публікує контрольних точок країв. Офіційні ваги DexiNed і TEED
навчено на BIPED, опубліковані умови якого обмежують використання датасету
некомерційними цілями, тому LibreYOLO не створює їх дзеркальних копій.
Перетворіть контрольну точку, яку ви маєте право використовувати, а потім
завантажте перетворений файл за шляхом:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

Назва файла має містити суфікс задачі `-edge`, щоб завантажувач розпізнав її.
`imgsz` має ділитися на крок зменшення роздільної здатності мережі, і в разі
невідповідності LibreYOLO спричиняє зрозумілу помилку з назвою дільника.
Джерела, потокове оброблення й роботу з результатами описано в розділі
[передбачення](/docs/predict).

## Формат датасету

Валідація країв зіставляє кожне зображення RGB з одноканальною картою такої
самої роздільної здатності й з тією самою основою назви, а також із
необов'язковою маскою коректності.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Цільовими даними є одноканальний PNG або TIF, а не візуалізація RGB.
Цілочислові карти діляться на максимальне значення їх типу даних; карти з
рухомою комою вже мають містити скінченні значення в діапазоні `[0, 1]`.
Пікселі маски вважаються коректними, якщо мають ненульове значення, а доповнені
пікселі ніколи не враховуються в метриці. `edge_invert: true` підтримує джерела,
які зберігають чорні краї на білому тлі. Повний контракт наведено в розділі
[форматів датасетів](/docs/reference/dataset-formats).

## Навчання

Жодне сімейство країв у LibreYOLO не має реалізації навчання: `train()`
спричиняє `NotImplementedError` в усіх трьох. На сторінці кожної моделі названо
скрипт перетворення, який створює придатну для LibreYOLO контрольну точку з
контрольної точки, навченої в іншому місці.

## Валідація

`val()` повідомляє F-міри в стилі BSDS. Безперервні передбачення спочатку
стоншуються чотиринапрямленим пригніченням немаксимумів градієнта, після чого
передбачені й еталонні пікселі країв зіставляються один до одного в межах
допуску відстані.

<code-tabs name="val" />

`metrics/ODS` є F-мірою оптимального масштабу датасету: кількості збігів
об'єднуються за всім датасетом для кожного порога, після чого повідомляється
найкраща з цих об'єднаних F-мір. Вона також є `fitness`, числом для вибору
найкращої контрольної точки. `metrics/OIS` є F-мірою оптимального масштабу
зображення, середнім за зображеннями значенням найкращої F-міри кожного
зображення, тому кожне зображення може вибрати власний поріг.
`metrics/best_threshold` є єдиним порогом, що створив ODS, саме його слід
повторно використовувати в `edges.binary()` під час інференсу.

Два аргументи визначають перебір. `edge_thresholds` є набором перевірених
порогів, типовий діапазон від 0.01 до 0.99 із кроком одна сота.
`edge_max_dist` є допуском зіставлення як часткою діагоналі зображення, типовим
значенням є `0.0075`; пара з більшою відстанню не вважається збігом.

## Експорт

Експортована модель країв завантажується назад через `LibreYOLO()` за суфіксом
файла, тому файл `.onnx` поводиться як контрольна точка й повертає той самий
об'єкт `Results`.

<code-tabs name="export" />

Експорт країв використовує контракт середовища виконання з фіксованою
роздільною здатністю та розміром батча 1: `dynamic` і значення `batch`, відмінне
від 1, відхиляються, а експортований граф створює одну об'єднану карту
ймовірностей. Покриття окремих форматів наведено на сторінках
[DexiNed](/docs/models/dexined) і [TEED](/docs/models/teed), а також у
[повній матриці експорту](/docs/reference/export-matrix). У розділі
[експорту](/docs/export) наведено аргументи, які приймає кожен формат.
