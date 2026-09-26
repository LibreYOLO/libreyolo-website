---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: видалення тла в LibreYOLO'
description: >-
  Використовуйте FeyNobg у LibreYOLO для видалення тла й alpha matting. Це
  поглиблений варіант BiRefNet від Feyn Inc. Установлюйте, виконуйте
  передбачення та валідацію.
lead: >-
  Модель видалення тла від Feyn Inc. поглиблює архітектуру BiRefNet і повторно
  навчає її. LibreYOLO постачає інференс і валідацію завдання matte моделі
  FeyNobg.
keywords:
  - FeyNobg
  - видалення фону
  - видалення тла з фото
  - дихотомічна сегментація зображень
  - alpha matte
  - image matting
  - вирізання об'єкта
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Вирізання
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: RGB джерела та matte як альфа-канал.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreFeyNobgl-matte.pt")


        # Замість YAML датасету також працює каталог із images/ і автоматично

        # виявленим каталогом matte (mattes/, matte/, gt/, masks/, mask/ або
        alpha/).

        metrics = model.val(data="my-matte-dataset/")


        print(metrics["metrics/MAE"])

        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Встановлення

FeyNobg не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання контрольна точка завантажується з організації
LibreYOLO на Hugging Face і кешується локально, як для будь-якого іншого
сімейства, хоча її ще не наведено в таблиці контрольних точок на цій сторінці.

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

Опубліковано один розмір `l` із бекбоном рівня Swin-L. FeyNobg бере
архітектуру BiRefNet і поглиблює її третій етап Swin із 18 до 24 блоків перед
повторним навчанням, тому порт LibreYOLO повторно використовує прямий шлях,
попередню обробку та контракт виходу з одним логітом від BiRefNet. Поведінка
передбачення, валідації та обробки контрольних точок збігається із сімейством
`birefnet`.

## Валідація

`val()` повідомляє дві метрики для парної папки зображень і matte. Обидві
належать до діапазону `[0, 1]` і не залежать від роздільної здатності: MAE
є середньою абсолютною похибкою відносно еталонної альфи (менше значення є
кращим), а S-measure (Fan et al., ICCV 2017) є структурною подібністю, що
враховує збереження форми об'єкта й отворів, які не помічає сама попіксельна
MAE (більше значення є кращим). Валідація викликає власний метод `predict`
моделі, тому використовує точну попередню обробку сімейства.

<code-tabs name="val" />

Валідація призначена лише для інференсу. Початкова бібліотека `nobg` містить
код навчання під ліцензією Apache-2.0. Наразі для донавчання потрібно навчити
модель там і перетворити результат власним скриптом LibreYOLO, а не викликати
`train()` для цього сімейства, оскільки цей метод спричиняє помилку замість
запуску часткового тренера.

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
