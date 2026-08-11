---
title: Виявлення об'єктів
seo_title: "Виявлення об'єктів у LibreYOLO"
description: "Виявляйте об'єкти як вирівняні за осями рамки в LibreYOLO: сімейства для цієї задачі, формат міток і виклики передбачення, навчання, валідації та експорту."
lead: "Виявлення об'єктів локалізує кожен екземпляр об'єкта на зображенні й повертає для нього вирівняний за осями прямокутник, мітку класу та оцінку. Ключ задачі має назву detect."
keywords: [виявлення об'єктів python, детекція об'єктів на зображенні, bounding box detection, MIT бібліотека детекції, альтернатива YOLO, навчити детектор об'єктів]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Інше сімейство, той самий виклик
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика маршрутизує за контрольною точкою, а кожен детектор повертає
        # той самий Results, тому для зміни сімейства достатньо одного рядка.
        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy.shape)
    - label: Відео та потоки
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Будь-яке джерело, яке приймає бібліотека: файл, каталог, URL-адреса,
        # індекс вебкамери, потік RTSP або список .streams.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yaml завантажує вибірку зі 128 зображень під час першого використання.
        # Для справжнього запуску вкажіть у data YAML власного датасету.
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Кілька GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() повертає звичайний словник, а не об'єкт.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Використати експортований файл
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Фабрика маршрутизує за суфіксом файла, тому експортований артефакт
        # завантажується як контрольна точка й повертає той самий Results.
        model = LibreYOLO("LibreYOLO9t.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Визначення

Виявлення об'єктів відповідає, де розташований і чим є кожен об'єкт. На вході
одне зображення, на виході один рядок для кожного екземпляра: чотири числа
прямокутника, індекс класу й оцінка. Дані про попіксельну форму, орієнтацію чи
частини відсутні, чим ця задача відрізняється від
[сегментації екземплярів](/docs/tasks/instance-segmentation),
[орієнтованих рамок](/docs/tasks/oriented-detection) та
[пози](/docs/tasks/pose-estimation).

`detect` є канонічним і типовим ключем задачі: контрольна точка, назва файла
якої не має суфікса задачі, завантажується як детектор.

`predict()` заповнює `result.boxes`. `.xyxy` повертає піксельні координати
кутів на полотні початкового зображення, `.conf` оцінку, а `.cls` індекс класу
в `result.names`. `.xywh`, `.xyxyn` і `.xywhn` є похідними поданнями тих самих
рядків, а `.id` містить ідентифікатор відстеження після підключення трекера.
Перебір об'єкта `Boxes` повертає зрізи з одним рядком, тому `box.cls`,
`box.conf` і `box.xyxy` працюють для кожного виявлення.

## Моделі

Дванадцять сімейств підтримують і навчання, і передбачення:
[YOLOv9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr),
[D-FINE](/docs/models/d-fine), [DEIM](/docs/models/deim),
[Dome-DETR](/docs/models/dome-detr), [YOLO-NAS](/docs/models/yolo-nas),
[YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7),
[RTMDet](/docs/models/rtmdet) і [PicoDet](/docs/models/picodet). YOLOv9 та
RF-DETR є двома флагманськими сімействами, і функції спочатку додаються до них.
RF-DETR потребує власного набору залежностей
`pip install "libreyolo[rfdetr]"`; решта працює з базовим пакетом.

Ще одинадцять підтримують передбачення, валідацію та експорт, але їхній
`train()` спричиняє `NotImplementedError`: [LW-DETR](/docs/models/lw-detr),
[DETR](/docs/models/detr),
[Deformable DETR](/docs/models/deformable-detr),
[DINO-DETR](/docs/models/dino-detr),
[Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) і
[EfficientDet](/docs/models/efficientdet).

Лінію Darknet, [YOLOv1](/docs/models/yolov1),
[YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) і
[YOLOv4](/docs/models/yolov4), збережено як заморожений експонат: передбачення,
валідація та експорт працюють, навчання ні.

Окрема група приймає список класів під час виконання, а не з контрольної точки,
тому виявляє назви, яких не бачила під час навчання:
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) і [OV-DEIM](/docs/models/ov-deim),
а також візуально-мовні сімейства
[Florence-2](/docs/models/florence-2), [Kosmos-2](/docs/models/kosmos-2),
[Qwen3-VL](/docs/models/qwen3-vl), [SmolVLM2](/docs/models/smolvlm2),
[InternVL3](/docs/models/internvl3), [LFM2-VL](/docs/models/lfm2-vl),
[LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) і
[LibreMODUS](/docs/models/libremodus). Вони завантажуються через власну фабрику
та набори залежностей; точний виклик наведено на сторінці кожної моделі.

## Передбачення

Ваги завантажуються з Hugging Face під час першого використання та кешуються
локально.

<code-tabs name="predict" />

`conf` задає поріг упевненості, а `max_det` обмежує кількість рядків. `iou` є
порогом NMS, тому впливає лише на сімейства, які виконують NMS; RF-DETR і
наскрізна голова YOLOv9 декодують фіксований набір передбачень та ігнорують
його. Джерела, потокове оброблення й роботу з результатами описано в розділі
[передбачення](/docs/predict).

## Формат датасету

Для кожного зображення використовується один файл міток `.txt`, знайдений
шляхом заміни `images` на `labels` у шляху зображення та зміни розширення.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Кожен рядок містить рівно п'ять полів, індекс класу, за яким іде нормалізована
рамка у форматі центру й розміру:

```text
<class_id> <cx> <cy> <w> <h>
```

Координати є числами з рухомою комою в діапазоні `[0, 1]` відносно ширини й
висоти початкового зображення. `w` і `h` мають бути додатними. Відсутній або
порожній файл міток означає, що зображення не має об'єктів. Рядки не містять
ані впевненості, ані ідентифікатора відстеження.

YAML називає розбиття й класи:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` і `val` можуть бути каталогами зображень, файлами `.txt` зі списками
зображень або списками будь-якого з цих типів. `nc` є необов'язковим і за
наявності має відповідати `names`. Нативний JSON COCO також працює: додайте
відповідність `annotations` між назвою розбиття та файлом JSON, після чого шлях
розбиття задає кореневий каталог зображень. Якщо наявний `names`, він визначає
ідентифікатори міток, тому назви категорій JSON мають йому відповідати.

## Навчання

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` і `lr0` є аргументами, які змінюють першими. `lr0`
не переноситься між сімействами: швидкість, яку витримує згортковий детектор,
спричинить розходження трансформерного, тому беріть значення зі сторінки моделі,
а не з прикладу іншого сімейства. Сімейство також може повністю ігнорувати
аргумент, і його сторінка містить відповідний список. Датасети, аугментацію,
кілька GPU й засоби журналювання описано в розділі [навчання](/docs/train).

## Валідація

`val()` повертає звичайний словник ключів `metrics/`, обчислених оцінюванням
COCO на розбитті, яке названо `val` у YAML датасету.

<code-tabs name="val" />

`metrics/mAP50-95` є середньою average precision за порогами IoU від 0.50 до
0.95 і головним показником. `metrics/mAP50` і `metrics/mAP75` є версіями для
одного порога. `metrics/mAP_small`, `metrics/mAP_medium` і
`metrics/mAP_large` ділять те саме середнє за площею об'єкта, а `metrics/AR1`,
`metrics/AR10`, `metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium` і
`metrics/AR_large` є відповідними показниками середньої повноти.
`metrics/AR_max_det` і `metrics/max_det` записують обмеження кількості виявлень,
використане під час запуску.

Уважно читайте `metrics/precision` і `metrics/recall` для цієї задачі. Їх
збережено для зворотної сумісності, і вони є псевдонімами, а не робочою точкою:
`metrics/precision` містить те саме значення, що й `metrics/mAP50-95`, а
`metrics/recall` те саме, що й `metrics/AR100`. Графік їх як пари точність-
повнота повідомляє одне число двічі. Чотири ключі також повторюються із
суфіксом `(B)` для рамок, щоб ключ виявлення однаково читався в моделі, яка
також передбачає маски: `metrics/mAP50-95(B)`, `metrics/mAP50(B)`,
`metrics/precision(B)` і `metrics/recall(B)`.

## Експорт

<code-tabs name="export" />

Експортований артефакт завантажується назад через `LibreYOLO()` за суфіксом
файла, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий об'єкт `Results`. Покриття форматів залежить від сімейства; матриця
на сторінці кожної моделі створюється з валідованого набору, а не вводиться
вручну. Формати, їхні додаткові набори залежностей та обмеження описано в
розділі [експорту й розгортання](/docs/export).
