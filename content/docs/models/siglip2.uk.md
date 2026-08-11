---
title: SigLIP2
families:
  - siglip2
seo_title: 'SigLIP2 у LibreYOLO: zero-shot класифікація та ембединги'
description: >-
  Використовуйте SigLIP2 у LibreYOLO для zero-shot класифікації зображень та
  ембедингів зображень і тексту з багатомітковим оцінюванням sigmoid. Навчання
  не потрібне.
lead: >-
  SigLIP2, це двобаштова модель, що оцінює зображення відносно текстових
  підказок незалежною sigmoid для кожного класу, а не спільною softmax для
  фіксованого набору міток. LibreYOLO підтримує її для zero-shot класифікації та
  ембедингів зображень і тексту без етапу навчання.
keywords:
  - SigLIP2
  - SigLIP 2
  - zero-shot класифікація
  - ембединг зображення
  - ембединг тексту
  - відкритий словник
  - багатомовна модель
  - sigmoid loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Без виклику set_classes() команда CLI predict використовує 1 000 назв

        # класів ImageNet, які модель стандартно завантажує.

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Багатоміткове оцінювання sigmoid
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)
        r = model(SAMPLE_IMAGE)

        # Незалежні ймовірності для класів: одночасно високий бал можуть
        # мати кілька класів або жоден. Натомість Softmax (стандартний режим)
        # нормалізує їх у розподіл для однієї мітки, як у LibreCLIP.
        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Ембединги зображень і тексту
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

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

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        # data, це корінь ImageFolder із поділом train/; назви його каталогів
        # стають підказками класів zero-shot для цього запуску.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Поточні мітки set_classes() і роздільна здатність входу вбудовуються
        # в граф. Після зміни будь-чого з них виконайте експорт знову. Під час
        # експорту multi_label має бути False (стандартне значення).
    - label: CLI
      language: bash
      code: |
        # Тут немає виклику set_classes(), тому вбудовуються стандартні 1 000
        # класів ImageNet, які завантажує модель.
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Експорт ембедингів
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" трасує лише башту зображень; класи не потрібні.
        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: f992655747fd8819
---

## Встановлення

SigLIP2 потребує власної додаткової залежності, яка встановлює пакет
SentencePiece для її багатомовного токенізатора.

```bash
pip install "libreyolo[siglip2]"
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

`set_classes()`, це основна операція, яка робить класифікатор відкритим:
вона підставляє кожну мітку в усі шаблони підказок, кодує й усереднює
результати та кешує отриману матрицю `[K, D]` як голову класифікатора, тому
вона не обчислюється знову для кожного зображення. Викличте її ще раз, щоб
будь-коли змінити класи. Без виклику LibreSigLIP2 завантажується з уже
заданими 1 000 назвами класів ImageNet-1k.

SigLIP оцінює кожен клас незалежно:
`logit = scale * (image . text) + bias`. Стандартно цей набір логітів усе
одно проходить через softmax, утворюючи одномітковий розподіл, що відповідає
поведінці `top1` і `top5` LibreCLIP. Передавання `multi_label=True`
до `set_classes()` або конструктора натомість перемикає на незалежні
ймовірності sigmoid, тому на одному зображенні високий бал можуть мати кілька
класів або жоден. Токенізатор є багатомовною моделлю SentencePiece зі
словником Gemma, тому назви класів іншими мовами працюють так само.

З `task="embed"` передбачення повертає один L2-нормалізований вектор
зображення на кожен вхід замість імовірностей класів, а `embed_text()`
повертає нормалізовані рядки тексту в тому самому векторному просторі, тому
звичайний скалярний добуток між ними дорівнює косинусній подібності. `iou`
не впливає на жодну задачу, оскільки етапу NMS немає. Джерела, потокову
обробку й роботу з результатами описано в розділі
[передбачення](/docs/predict).

## Валідація

`val()` читає назви каталогів класів у поділі `train/` ImageFolder,
викликає з ними `set_classes()`, а потім вимірює точність zero-shot top-1
і top-5 з оцінюванням softmax. Точність залежить від того, як назви класів
сприймаються як підказки, а не від оновлення ваг, оскільки навчання не
відбувається. Валідація охоплює лише `task="classify"`; для
`task="embed"` немає валідатора датасету.

<code-tabs name="val" />

## Експорт

<export-matrix />

Експорт вбудовує поточний стан моделі у фіксований граф. Для
`task="classify"` останні мітки, задані через `set_classes()`, і
роздільна здатність під час експорту вбудовуються в кінцевий лінійний шар
разом із навченими масштабом і зміщенням. Тому експортований граф є звичайним
класифікатором зображень `[B, K]` без текстової башти й токенізатора;
після зміни класів або розміру експортуйте знову. Експорт у режимі
`multi_label=True` не реалізовано, спочатку поверніть значення `False`.
Експорт `task="embed"` трасує лише башту зображень. Обом варіантам потрібен
ONNX opset 14 або новіший, який експортер задає стандартно.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг для цього сімейства. Обидва перетворено з
контрольних точок Google `siglip2-base-patch16-256` і
`siglip2-so400m-patch14-384` за ліцензією Apache-2.0, а не з будь-якого
навчання на COCO.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />

