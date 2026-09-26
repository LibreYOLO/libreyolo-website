---
title: CLIP
families:
  - clip
seo_title: 'CLIP у LibreYOLO: класифікація без прикладів та ембединги'
description: >-
  Використовуйте CLIP у LibreYOLO для класифікації зображень без прикладів та
  ембедингів зображень/тексту. Без навчання: set_classes() визначає набір міток
  під час виконання.
lead: >-
  CLIP є двовежовою моделлю, яка оцінює зображення відносно текстових запитів
  замість фіксованого набору міток. LibreYOLO підтримує її для класифікації без
  прикладів та ембедингів зображень/тексту без етапу навчання.
keywords:
  - CLIP
  - OpenCLIP
  - zero-shot класифікація
  - ембединг зображення
  - текстовий ембединг
  - відкритий словник
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Без виклику set_classes() передбачення CLI використовує 1,000 назв

        # класів ImageNet, з якими модель завантажується типово.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Ембединги зображення й тексту
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Обидва L2-нормалізовані, тому звичайний скалярний добуток є косинусною
        подібністю.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data є коренем ImageFolder із поділом train/; назви папок стають
        # запитами класів без прикладів для цього запуску.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Поточні мітки set_classes() і вхідна роздільна здатність вбудовуються
        # у граф. Після зміни будь-якого з них експортуйте повторно.
    - label: CLI
      language: bash
      code: |
        # Тут немає виклику set_classes(), тому вбудовуються типові 1,000
        # класів ImageNet, з якими завантажується модель.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Експорт ембедингів
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" трасує лише вежу зображення; класи не потрібні.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Встановлення

CLIP потребує власного набору додаткових залежностей, який установлює пакети
для включеного токенізатора BPE, щоб точно відтворювати ідентифікатори токенів.

```bash
pip install "libreyolo[clip]"
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

`set_classes()` є базовою операцією, яка робить цей класифікатор відкритим:
вона підставляє кожну мітку в усі шаблони запитів, кодує й усереднює
результати та кешує отриману матрицю `[K, D]` як голову класифікатора, тому
її не обчислюють повторно для кожного зображення. Викличте метод знову, щоб
будь-коли змінити класи. Без виклику LibreCLIP завантажується з уже заданими
1,000 назвами класів ImageNet-1k.

З `task="embed"` передбачення повертає один L2-нормалізований вектор
зображення на вхід замість імовірностей класів, а `embed_text()` повертає
нормалізовані рядки тексту в тому самому векторному просторі, тому звичайний
скалярний добуток між ними є косинусною подібністю. `iou` не впливає на
жодне завдання; етапу NMS немає. Типи джерел, потокове передбачення та обробку
результатів описано в розділі [передбачення](/docs/predict).

## Валідація

`val()` читає назви папок класів у поділі `train/` ImageFolder, викликає з
ними `set_classes()`, а потім вимірює правильність top-1 і top-5 без
прикладів. Правильність залежить від того, як назви класів читаються як
запити, а не від оновлення ваг, оскільки навчання не виконується. Валідація
охоплює лише `task="classify"`; `task="embed"` не має валідатора датасету.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експорт вбудовує поточний стан моделі у фіксований граф. Для
`task="classify"` мітки, востаннє задані `set_classes()`, і роздільна
здатність під час експорту вбудовуються у завершальний лінійний шар. Тому
експортований граф ONNX або TensorRT є звичайним класифікатором зображень
`[B, K]` без текстової вежі й токенізатора; після зміни класів або розміру
експортуйте повторно. Експорт `task="embed"` трасує лише вежу зображення.
Для обох потрібен opset ONNX 14 або новіший, який експортер задає типово.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства. Обидві перетворено з контрольних
точок OpenCLIP, навчених на LAION-2B (`ViT-B-32` і `ViT-B-16`), а не з
будь-якого запуску навчання COCO.

<checkpoint-table />

Навчальні дані LAION-2B мають задокументовану історію вмісту CSAM (Stanford
Internet Observatory, грудень 2023 року). Відтоді LAION випустила Re-LAION,
очищене повторне видання. Якщо далі розміщуєте ці ваги, за можливості
віддавайте перевагу контрольним точкам на основі Re-LAION.

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
