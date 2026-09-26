---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: надрозрізнення зображень у LibreYOLO'
description: >-
  Використовуйте Real-ESRGAN у LibreYOLO для практичного надрозрізнення
  зображень у 4x, 2x і швидкому рівні 4x. Установлюйте, виконуйте передбачення,
  валідацію та експорт.
lead: >-
  Практичний засіб сліпого надрозрізнення, навчений на синтетичних деградаціях,
  а не лише на бікубічному зменшенні. LibreYOLO постачає інференс і валідацію
  його контрольних точок 4x, 2x і швидкої 4x.
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - надрозрізнення зображень
  - відновлення зображень
  - сліпе надрозрізнення
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Тайли для великих зображень
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")

        # tile ділить прямий прохід на тайли з перекриттям і змішує стики;
        # tile_pad є ореолом навколо кожного тайла, який потім обрізається.
        # Обидва є аргументами лише Python, а не прапорцями CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # Якщо imgsz пропущено, типовим є малий внутрішній розмір патча, а не

        # робоча роздільна здатність, тому передайте розмір фактичного входу
        розгортання.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Встановлення

Real-ESRGAN не потребує додаткових залежностей. Усе, що вона імпортує, входить
до базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Результат відновлення не містить рамок; `result.restored` є щільним
RGB-зображенням uint8 `(H, W, 3)` на полотні, яке у
`Results.restore_scale` разів більше за вхід у кожному вимірі. `save=True`
записує безпосередньо це зображення, а не анотований графік. Вхід
перетворюється на RGB, а альфа-канал видаляється. Джерело, більше за доступну
пам'ять, можна поділити за допомогою `tile` і `tile_pad`, які змішують
стики тайлів у виході. Типи джерел, потокове передбачення та обробку
результатів описано в розділі [передбачення](/docs/predict).

## Варіанти

Доступні три контрольні точки, названі за коефіцієнтом збільшення. `x4` є
RRDBNet (`RealESRGAN_x4plus`) із 23 щільними блоками
residual-in-residual і типовою якістю 4x. `x2` має ту саму архітектуру
RRDBNet у 2x. `x4t` є SRVGGNetCompact (`realesr-general-x4v3`), меншим і
швидшим генератором для відео та роботи з меншою затримкою в 4x. Початкова
модель загального призначення також постачає парну мережу сили шумозаглушення,
яка змішується під час інференсу. Це налаштування сили не входить до цього
порту, який запускає базовий генератор `x4t`.

## Валідація

`val()` вимірює PSNR і SSIM між відновленим виходом і чистим цільовим
зображенням. Обидві метрики обчислюються в RGB на початковому полотні без
обрізання меж і зміни розміру. SSIM використовує гаусове вікно 11x11 із
сигмою 1.5, усереднене за трьома колірними каналами.

<code-tabs name="val" />

Аргумент датасету є YAML, який поєднує каталог деградованих вхідних зображень
із каталогом чистих цільових зображень відповідної роздільної здатності. Точні
ключі описано у [форматах датасетів](/docs/reference/dataset-formats).

## Експорт

<export-matrix />

Експортований артефакт знову завантажується через `LibreYOLO()` відповідно до
суфікса файлу, тому файл `.onnx` або `.engine` поводиться як контрольна
точка й повертає той самий об'єкт `Results`. У розділі
[експорту](/docs/export) наведено аргументи, які приймає кожен формат, і
додаткові параметри деяких форматів.

<code-tabs name="export" />

## Контрольні точки

Усі опубліковані файли ваг цього сімейства.

<checkpoint-table />

## Ліцензування

<provenance-box></provenance-box>

## Цитування

<citation-block />
