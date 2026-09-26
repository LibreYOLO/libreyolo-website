---
title: Виявлення точок
seo_title: Виявлення точок і підрахунок у LibreYOLO
description: >-
  Визначайте розташування об'єктів як окремі точки замість рамок у LibreYOLO.
  Передбачайте центроїди, підраховуйте об'єкти, навчайте FOMO та аналізуйте
  метрики точок.
lead: >-
  Виявлення точок повертає для кожного об'єкта одне положення x, y замість
  обмежувальної рамки. LibreYOLO надає цю можливість як задачу point, а
  передбачення містить для кожного об'єкта один рядок зі значеннями x, y, класом
  і впевненістю.
keywords:
  - виявлення точок python
  - підрахунок об'єктів python
  - детекція центроїдів
  - локалізація точок FOMO
  - підрахунок об'єктів на зображенні
  - точкова локалізація
last_verified: 1.5.0
snippets:
  predict:
    - label: Передбачити точки й підрахувати їх
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Ваги LibreFOMO не завантажуються автоматично. Спочатку отримайте

        # контрольну точку з https://huggingface.co/LibreYOLO і завантажте її з
        локального шляху.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        points = result.points

        print(len(points))     # кількість об'єктів

        print(points.xy)       # центри (N, 2) у пікселях початкового зображення

        print(points.cls, points.conf)
    - label: Нормалізовані координати й кількість за класами
      language: python
      code: >
        from collections import Counter


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE)


        points = result.points.numpy()

        print(points.xyn)                          # ті самі центри в діапазоні
        [0, 1]

        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Навчити FOMO на датасеті YOLO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Передбачити за допомогою навченої контрольної точки
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        results = model.train(data="my-dataset.yaml", epochs=40)


        # train() повторно завантажує найкращу контрольну точку в той самий
        об'єкт, тому

        # після повернення з виклику модель передбачає за допомогою навчених
        ваг.

        print(results["best_checkpoint"])

        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Провалідувати й переглянути ключі метрик
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")

        metrics = model.val(data="my-dataset.yaml")


        print(metrics["metrics/precision"], metrics["metrics/recall"])

        print(metrics["metrics/f1"])

        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness

        print(metrics["metrics/MLE"])               # середня похибка
        локалізації

        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # похибка
        підрахунку
    - label: Змінити пороги відстані
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # Межі перебору є частиною тексту ключа, тому власний перебір

        # перейменовує створені ним ключі mAP.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Експорт
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Запустити експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика визначає маршрут за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Визначення

Задача `point` визначає розташування кожного об'єкта за однією координатою x, y і класом
без ширини, висоти чи маски. Оскільки передбачення є плоским списком об'єктів,
кількість рядків дорівнює кількості об'єктів, тому ця задача придатна для підрахунку.

Передбачення заповнює `result.points`, корисне навантаження `Points`, яке обгортає масив
`(N, 4)` із рядками `x, y, class, confidence` у пікселях початкового зображення. `.xy`
повертає координати, `.xyn` повертає ті самі координати, поділені на розмір зображення,
`.cls` повертає індекси класів, а `.conf` повертає оцінки; `len()` повертає кількість
точок. `result.boxes` залишається порожнім, тому `iou` і `max_det` не мають на що діяти.

## Моделі

Задачу `point` підтримують три сімейства, які не є взаємозамінними.

[FOMO](/docs/models/fomo) є варіантом із фіксованим словником: сітковим класифікатором,
який позначає кожну комірку сітки низької роздільної здатності як фон або центр об'єкта.
Це єдине сімейство точкових моделей, яке LibreYOLO може навчати й експортувати.

[LocateAnything](/docs/models/locate-anything) приймає текст замість індексу класу,
тож словником може бути будь-яка введена фраза. Для нього потрібне доповнення `vlm`,
а створювати його слід як `LibreLocateAnything`, а не через фабрику `LibreYOLO()`.
Його ваги дозволено використовувати лише в некомерційних цілях. Точні умови та ще
дві ліцензії, які поєднує контрольна точка, наведено на сторінці моделі.

[SenseNova-Vision](/docs/models/sensenova-vision) підтримує `point` через ту саму
контрольну точку з генерацією за підказкою, яку використовує для шести інших задач,
завантажену за допомогою `LibreVLM("sensenova-vision", task="point")`. Для неї потрібне
доповнення `sensenova`, а кожне передбачення виконує прохід генерації через модель 7B,
тому затримка на зображення помітно вища, ніж у спеціалізованого детектора. Її ваги
дозволено використовувати лише в некомерційних цілях; ліцензію наведено на сторінці моделі.

## Передбачення

Ваги LibreFOMO є єдиним винятком з автоматичного завантаження на цьому сайті.
`LibreYOLO("LibreFOMOs-point.pt")` шукає цей файл на диску й замість завантаження
породжує `ValueError` із його назвою. Спочатку завантажте контрольну точку з
[організації LibreYOLO](https://huggingface.co/LibreYOLO) на Hugging Face
і відкрийте її за локальним шляхом або навчіть власну модель.

<code-tabs name="predict" />

Щоб завантажувач розпізнав задачу, назва файлу має містити суфікс `-point`.
Параметр `predict(..., nms_radius=1)` визначає, на відстані скількох комірок сітки
мають бути два виявлення FOMO, щоб збереглися обидва. Докладніше про джерела,
потокову обробку та роботу з результатами див. у розділі
[передбачення](/docs/predict).

## Формат датасету

Задача `point` не має власного формату міток. Сімейства точкових моделей читають
стандартну структуру виявлення YOLO і виводять один центр із кожного рядка рамки,
тому `cx cy` задають точку, а `w h` лише визначають дійсність рядка.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Кожен файл міток містить один рядок на об'єкт із нормалізованими координатами:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

Відсутній або порожній файл міток означає, що об'єктів немає. Повний контракт
наведено в розділі [форматів датасетів](/docs/reference/dataset-formats).

## Навчання

FOMO є єдиним сімейством точкових моделей із реалізованим навчанням. Виклик `train()`
для LocateAnything і SenseNova-Vision породжує `NotImplementedError`; донавчайте ці
моделі в їхніх початкових проєктах і завантажуйте результат.

<code-tabs name="train" />

Для FOMO значення `imgsz` не можна вибирати довільно: типово використовується
власна роздільна здатність завантаженої контрольної точки, а передавання іншого
значення породжує `ValueError` із зазначенням очікуваного розміру. Відомості про
датасети, логери й роботу з кількома GPU див. у розділі [навчання](/docs/train),
а типові значення цього сімейства наведено на [сторінці FOMO](/docs/models/fomo).

## Валідація

Метод `val()` зіставляє передбачені точки з еталонними точками один до одного
за допомогою угорського алгоритму для набору порогів відстані. Порогом є евклідова
відстань у нормалізованих координатах зображення, а типовий набір містить десять
значень від 0.01 до 0.10.

<code-tabs name="val" />

Значення `metrics/precision`, `metrics/recall` і `metrics/f1` усереднюються макро
за класами на найсуворішому порозі набору, типово 0.01. `metrics/mAP@0.01` є
середньою точністю на тому самому порозі, а `metrics/mAP@[0.01:0.10]` є середнім
значенням для всього набору. Значення для набору також слугує `fitness`, яке
використовує вибір найкращої контрольної точки. Обидва ключі mAP формуються
з використаних порогів, тому передавання `dist_thresholds=` перейменовує їх.

`metrics/MLE` є середньою відстанню між зіставленими парами на найсуворішому
порозі в тих самих нормалізованих одиницях. `metrics/MAE` і `metrics/RMSE`
є метриками підрахунку, а не локалізації: вони вимірюють для кожного зображення
різницю між кількістю передбачених і еталонних точок.

FOMO додає до них другу групу на рівні сітки. Вона перебирає впевненість і
`nms_radius` та публікує комбінацію з найкращим F1 як `metrics/grid_F1`,
`metrics/grid_precision`, `metrics/grid_recall`, `metrics/grid_mean_distance`,
`metrics/grid_TP`, `metrics/grid_FP` і `metrics/grid_FN`, а налаштування, що дали
цей результат, розміщує в `decode/threshold` і `decode/nms_radius`.

## Експорт

FOMO експортується через спільний шлях експорту, а експортований артефакт знову
завантажується через `LibreYOLO()` за суфіксом файлу, тому файл `.onnx` або `.engine`
поводиться як контрольна точка й повертає той самий об'єкт `Results`.

<code-tabs name="export" />

Підтримку окремих форматів наведено на [сторінці FOMO](/docs/models/fomo)
і в [повній матриці експорту](/docs/reference/export-matrix). LocateAnything і
SenseNova-Vision не підтримують експорт: `export()` породжує помилку для обох,
оскільки генеративна модель не має придатного до трасування графа виявлення.
