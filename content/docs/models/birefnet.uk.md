---
title: BiRefNet
families:
  - birefnet
seo_title: 'BiRefNet: видалення тла та matting у LibreYOLO'
description: >-
  Використовуйте BiRefNet у LibreYOLO для видалення тла й дихотомічної
  сегментації зображень. Установлюйте, виконуйте передбачення, валідацію та
  експорт контрольної точки general.
lead: >-
  Мережа двостороннього зіставлення передбачає м'яке alpha matte, що відокремлює
  об'єкт від тла. LibreYOLO постачає інференс і валідацію завдання matte моделі
  BiRefNet.
keywords:
  - BiRefNet
  - видалення фону
  - видалення тла з фото
  - дихотомічна сегментація зображень
  - alpha matte
  - image matting
  - вирізання об'єкта
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Вирізання
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: RGB джерела та matte як альфа-канал.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreBiRefNetl-matte.pt")


        # Замість YAML датасету також працює каталог із images/ і автоматично

        # виявленим каталогом matte (mattes/, matte/, gt/, masks/, mask/ або
        alpha/).

        metrics = model.val(data="my-matte-dataset/")


        print(metrics["metrics/MAE"])

        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreBiRefNetl-matte.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## Встановлення

BiRefNet не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Результат matte не містить рамок; `result.matte` є щільним масивом float32
`(H, W)` у діапазоні `[0, 1]`, де 1 означає повністю передній план, а 0
повністю тло. На відміну від бінарної маски, м'яке matte зберігає згладжені
деталі країв, як-от волосся та хутро. `result.cutout()` об'єднує початкове
зображення з цим альфа-каналом у масив RGBA, а `result.save(path)` (або
`save=True` у виклику передбачення) записує його безпосередньо до PNG із
прозорим тлом. Модель працює на фіксованому початковому полотні 1024x1024;
інша роздільна здатність не підтримується, оскільки таблиці відносних позицій
бекбона Swin прив'язані до неї, а невідповідність спричиняє їх погану
інтерполяцію замість помилки. Типи джерел, потокове передбачення та обробку
результатів описано в розділі [передбачення](/docs/predict).

## Варіанти

Опубліковано одну контрольну точку `l`, модель BiRefNet-general рівня Swin-L
і типовий варіант якості початкового проєкту. Код сімейства також підтримує
легкий рівень Swin-T `t`, але його перетворення LibreYOLO ще не опубліковано.

## Валідація

`val()` повідомляє дві метрики для парної папки зображень і matte. Обидві
належать до діапазону `[0, 1]` і не залежать від роздільної здатності: MAE
є середньою абсолютною похибкою відносно еталонної альфи (менше значення є
кращим), а S-measure (Fan et al., ICCV 2017) є структурною подібністю, що
враховує збереження форми об'єкта й отворів, які не помічає сама попіксельна
MAE (більше значення є кращим). Валідація викликає власний метод `predict`
моделі, тому використовує точну попередню обробку сімейства.

<code-tabs name="val" />

Валідація призначена лише для інференсу; донавчання є задокументованим
наступним кроком, а не наявною функцією (точне обмеження роздільної здатності,
яке успадкує будь-який майбутній тренер, див. у розділі «Передбачення»).

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` поводиться як контрольна точка й повертає
той самий об'єкт `Results`. Перевіреним шляхом є TorchScript; перетворення
ONNX виконується, але ще не пройшло той самий рівень перевірки відповідності.
У розділі [експорту](/docs/export) наведено аргументи, які приймає кожен
формат, і додаткові параметри деяких форматів.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
