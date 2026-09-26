---
title: Паноптична сегментація
seo_title: Паноптична сегментація в LibreYOLO
description: >-
  Призначте кожному пікселю один сегмент у LibreYOLO: сімейства для цієї задачі,
  формат датасету COCO-panoptic і виклики передбачення та валідації.
lead: >-
  Паноптична сегментація призначає кожен піксель рівно одному сегменту без
  перекриття, поєднуючи злічувані екземпляри об'єктів з аморфними областями тла.
  Ключ задачі має назву panoptic.
keywords:
  - паноптична сегментація python
  - panoptic quality
  - сегментація things stuff
  - формат COCO panoptic
  - карта id сегментів
  - метрика PQ
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Суфікс -panoptic у назві файла вибирає задачу, тому аргумент
        # task не потрібний.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) ідентифікатори сегментів
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: По одному сегменту
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # булевий (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Менша контрольна точка
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() повертає звичайний словник, а не об'єкт.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## Визначення

Паноптична сегментація є об'єднанням двох інших задач сегментації. Кожен
піксель отримує рівно один сегмент, сегменти ніколи не перекриваються, а сегмент
є або об'єктом, злічуваним екземпляром, або фоном, аморфною областю на кшталт
неба чи дороги. Тому вона строгіша за
[сегментацію екземплярів](/docs/tasks/instance-segmentation), яка залишає
пікселі тла непризначеними й дозволяє маскам перекриватися, і за
[семантичну сегментацію](/docs/tasks/semantic-segmentation), яка позначає кожен
піксель, але об'єднує дотичні екземпляри одного класу.

`panoptic` є канонічним ключем задачі, а суфікс `-panoptic` у назві файла
контрольної точки вибирає її, тому під час завантаження опублікованих ваг
`task=` не потрібний.

`predict()` заповнює `result.panoptic`. `.data` є цілочисловою картою
ідентифікаторів сегментів у формі `(H, W)` на полотні початкового зображення.
`.segments_info` є списком словників, по одному для кожного сегмента, кожен із
яких містить щонайменше `{"id", "category_id"}`, де `id` відповідає значенню
на карті, а `category_id` індексує `result.names`. `.segment_ids` перелічує
наявні ідентифікатори у відсортованому порядку, а `.segment_mask(id)` повертає
булевий вибір одного сегмента у формі `(H, W)`. Ідентифікатор сегмента `0` є
порожнім значенням: немарковані пікселі, вилучені з метрики та зі
`.segment_ids`.

Поділ на об'єкти й фон є властивістю категорії, а не окремого сегмента. Він
зберігається в метаданих категорій набору міток, а дані передбачення можуть
для зручності копіювати його в кожен сегмент як `"isthing"`, але метадані
категорій залишаються авторитетними.

## Моделі

[EoMT](/docs/models/eomt) є сімейством, яке виконує цю задачу через
`LibreYOLO()`. Воно працює з базовим пакетом і постачає паноптичні контрольні
точки трьох розмірів, s, b і l, навчені на COCO.

[SenseNova-Vision](/docs/models/sensenova-vision) також створює паноптичні
карти. Це генеративна модель із підказками, власною фабрикою `LibreVLM` і
власним набором залежностей; без заданого словника вона використовує паноптичні
категорії COCO, на яких її налаштовано. Її ваги призначено лише для
некомерційного використання. Затримка для кожного зображення значно вища, ніж
у спеціалізованого сегментатора, оскільки кожне передбачення виконує дифузійне
декодування.

## Передбачення

Ваги завантажуються з Hugging Face під час першого використання та кешуються
локально.

<code-tabs name="predict" />

`conf` фільтрує вибір запитів. Джерела, потокове оброблення й роботу з
результатами описано в розділі [передбачення](/docs/predict).

## Формат датасету

LibreYOLO дослівно використовує формат COCO-panoptic від Kirillov et al.,
CVPR 2019. Окремої паноптичної структури LibreYOLO немає.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Кожне зображення зіставляється з одним PNG RGB такої самої роздільної
здатності, де колір кожного пікселя кодує ідентифікатор сегмента, якому він
належить:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Ідентифікатор сегмента `0`, чорний RGB, є порожнім значенням: немаркованими
пікселями, які не винагороджують і не штрафують передбачення. Кожен інший
піксель належить рівно одному сегменту.

JSON перелічує для кожного зображення PNG з ідентифікаторами сегментів і
сегменти всередині нього:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` називає PNG у паноптичному каталозі, а
`segments_info[].id` відповідає значенню в цьому PNG. `iscrowd` позначає
групові області: вони ніколи не враховуються як хибнонегативні результати, а
передбачення, яке переважно покриває одну з них, не є хибнопозитивним.
`isthing` належить до `categories`, а не до окремого сегмента.

YAML указує на обидва:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` і `panoptic_dir` приймають один шлях або відповідність для
кожного розбиття. Початкові ідентифікатори категорій COCO зазвичай не
послідовні, тоді як моделі передбачають послідовні `0..nc-1`, тому
ідентифікатори перепризначаються через `names` за назвою категорії. Відсутня в
`names` категорія JSON спричиняє помилку замість непомітного відкидання,
оскільки відкидання оцінювалося б як постійний хибнонегативний результат.

Канонічним завантажувачем є `libreyolo.data.PanopticDataset`.

## Навчання

Сьогодні жодне сімейство не навчає паноптичну сегментацію в LibreYOLO:
`train()` моделі EoMT спричиняє `NotImplementedError`, тому паноптичні
контрольні точки використовуються як опубліковано.

## Валідація

`val()` повертає звичайний словник ключів `metrics/`, обчислених із роздільною
здатністю еталонних даних на розбитті, яке названо `val` у YAML датасету.
Передбачений та еталонний сегменти однієї категорії збігаються, коли їхній IoU
перевищує 0.5, і цей збіг є унікальним.

<code-tabs name="val" />

`metrics/PQ` є Panoptic Quality, головним показником. У межах однієї категорії
він є добутком двох множників. Якість сегментації є середнім IoU за зіставленими
сегментами й показує, наскільки добре вирівняно форми збігів. Якість
розпізнавання дорівнює `TP / (TP + 0.5 FP + 0.5 FN)`, F1-оцінці самого
зіставлення, і показує, скільки сегментів узагалі знайдено. Потім усі три
показники усереднюються за наявними категоріями й повідомляються як
`metrics/PQ`, `metrics/SQ` і `metrics/RQ`, тому повідомлене PQ є середнім
добутків для кожної категорії, а не добутком двох повідомлених середніх.

`metrics/PQ_things` і `metrics/PQ_stuff` окремо усереднюють те саме PQ для
кожної категорії за категоріями об'єктів та фону, а `metrics/categories`
містить кількість наявних категорій, за якими виконано усереднення. Словник
також містить `fitness`, копію значення PQ.

## Експорт

Паноптичні контрольні точки не експортуються. `export()` спричиняє
`NotImplementedError` для цієї задачі, оскільки вихід масок запитів ще не має
контракту експорту для середовища виконання. Семантична задача EoMT підтримує
експорт; дивіться [семантичну сегментацію](/docs/tasks/semantic-segmentation)
і [експорт та розгортання](/docs/export).
