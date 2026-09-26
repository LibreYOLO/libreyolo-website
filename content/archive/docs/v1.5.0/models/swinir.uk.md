---
title: SwinIR
families:
  - swinir
seo_title: 'SwinIR: 4-кратне надрозрізнення зображень у LibreYOLO'
description: >-
  Використовуйте SwinIR у LibreYOLO для 4-кратного надрозрізнення зображень.
  Установлюйте, виконуйте передбачення, валідацію та експорт контрольних точок
  lightweight, medium і large.
lead: >-
  Мережа Swin Transformer для відновлення зображень. LibreYOLO постачає інференс
  і валідацію її контрольних точок 4-кратного надрозрізнення: офіційного легкого
  генератора, а також генераторів medium і large для реальних зображень.
keywords:
  - SwinIR
  - Swin Transformer
  - надрозрізнення зображень
  - відновлення зображень
  - residual Swin Transformer block
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Тайли для великих зображень
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRl-restore.pt")

        # tile ділить прямий прохід на тайли з перекриттям і змішує стики;
        # tile_pad є ореолом навколо кожного тайла, який потім обрізається.
        # Обидва є аргументами лише Python, а не прапорцями CLI.
        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwinIRm-restore.pt")


        # Якщо imgsz пропущено, типовим є малий внутрішній розмір патча, а не

        # робоча роздільна здатність, тому передайте розмір фактичного входу
        розгортання.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Використання експортованого файлу
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика маршрутизує за суфіксом файлу, тому експортований артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results.

        model = LibreYOLO("LibreSwinIRm-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## Встановлення

SwinIR не потребує додаткових залежностей. Усе, що вона імпортує, входить до
базового встановлення.

```bash
pip install libreyolo
```

## Передбачення

Під час першого використання ваги завантажуються з Hugging Face і кешуються
локально.

<code-tabs name="predict" />

Результат відновлення не містить рамок; `result.restored` є щільним
RGB-зображенням uint8 `(H, W, 3)` на полотні, яке в 4 рази більше за вхід у
кожному вимірі. `save=True` записує безпосередньо це зображення, а не
анотований графік. Вхід доповнюється до кратного 8 замість зміни розміру, тому
передбачення працює з власною роздільною здатністю фотографії. Джерело,
більше за доступну пам'ять, можна поділити за допомогою `tile` і `tile_pad`,
які змішують стики тайлів у виході. Типи джерел, потокове передбачення та
обробку результатів описано в розділі
[передбачення](/docs/predict).

## Варіанти

Доступні три розміри, усі з фіксованим 4-кратним збільшенням. `s` є
офіційним легким генератором із чотирма етапами залишкових блоків Swin
Transformer (RSTB) і прямим підвищенням роздільної здатності через
pixel-shuffle. `m` і `l` є генераторами medium і large для реальних
зображень із шістьма та дев'ятьма етапами RSTB і модулем підвищення роздільної
здатності на основі найближчого сусіда зі згорткою, створеним для реальних
деградацій, а не лише бікубічного зменшення.

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
точка й повертає той самий об'єкт `Results`. ExecuTorch і всі формати,
позначені в матриці як заблоковані, недоступні для цього сімейства; доступні
ONNX, TorchScript, TensorRT, OpenVINO та TFLite. У розділі
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
