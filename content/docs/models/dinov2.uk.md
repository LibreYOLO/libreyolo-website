---
title: DINOv2
families:
  - dinov2
seo_title: 'DINOv2 у LibreYOLO: семантична сегментація, класифікація та ембединг'
description: >-
  Використання DINOv2 у LibreYOLO для семантичної сегментації, класифікації та
  ембедингу всього зображення на бекбоні DINOv2-with-Registers. Повністю за
  ліцензією Apache-2.0.
lead: >-
  DINOv2 є самокерованим візуальним трансформером, який Meta AI навчила
  створювати універсальні ознаки зображення без міток. LibreYOLO обгортає його
  бекбон DINOv2-with-Registers для трьох завдань: семантичної сегментації,
  класифікації та ембедингу всього зображення.
keywords:
  - DINOv2
  - DINOv2 with registers
  - самокероване навчання
  - візуальний трансформер
  - семантична сегментація
  - ембединг зображення
  - виділення ознак
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Семантична сегментація
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.dinov2.model import LibreDINOv2


        # Для цього сімейства немає контрольної точки на хостингу LibreYOLO: цей
        код

        # завантажує бекбон DINOv2-with-Registers-small за ліцензією Apache-2.0
        з

        # організації Meta на Hugging Face. Щільна голова має випадкову

        # ініціалізацію, доки ви не навчите її (див. розділ «Навчання» нижче).

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)

        result = model(SAMPLE_IMAGE)


        mask = result.semantic_mask

        print(mask.data.shape, mask.classes)
    - label: Класифікація
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.dinov2.model import LibreDINOv2


        # nb_classes= визначає кількість класів вашого датасету; лінійна голова
        має

        # випадкову ініціалізацію, доки ви не навчите її.

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1, result.probs.top1conf)
    - label: Ембединг
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Обходить усі голови завдань: достатньо самого бекбона, тому
        # корисний результат не потребує донавчання.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), нормалізовано за L2
    - label: Ембединг батча
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Зручна обгортка: запускає predict() і складає кожен рядок в один
        # тензор (N, D).
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Семантична сегментація
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Класифікація
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Кілька GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Семантична сегментація
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Класифікація
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Семантична сегментація
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Класифікація
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Ембединг
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Використати експортований файл
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Фабрика виконує маршрутизацію за суфіксом файлу, тому експортований
        артефакт

        # завантажується як будь-яка контрольна точка й повертає той самий
        об'єкт Results. Експорт

        # називає файл за завданням, тут LibreDINOv2s-sem.onnx.

        model = LibreYOLO("LibreDINOv2s-sem.onnx")

        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## Встановлення

LibreDINOv2 реєструється, лише якщо встановлено `transformers`, ту саму
необов'язкову залежність, яка потрібна RF-DETR для бекбона DINOv2, тому потрібен
той самий додатковий пакет.

```bash
pip install "libreyolo[rfdetr]"
```

## Передбачення

LibreYOLO не публікує контрольну точку LibreDINOv2. Створіть обгортку
безпосередньо замість завантаження файлу: `model_path=None` (типове значення)
під час першого використання завантажує з Hugging Face бекбон Meta
`facebook/dinov2-with-registers-small` за ліцензією Apache-2.0. Параметр `task=`
вибирає те, що запускається поверх нього.

<code-tabs name="predict" />

Завдання `task="semantic"` і `task="classify"` додають поверх бекбона щільну або
лінійну голову; ця голова має випадкову ініціалізацію й стає корисною лише після
навчання (див. [Навчання](#train)). Завдання `task="embed"` пропускає всі голови
та повертає кінцевий нормалізований токен CLS бекбона як один рядок для всього
зображення у `result.embeddings`, тому взагалі не потребує навчання.
`result.boxes` завжди дорівнює `None`: жодне з трьох завдань не створює виявлень
окремих екземплярів. Джерела, потокове оброблення та роботу з результатами
описано на сторінці [передбачення](/docs/predict).

## Варіанти

`size` вибирає ширину проєктора в стилі RF-DETR, розташованого поверх бекбона,
а не сам бекбон: усі розміри використовують спільний кодер DINOv2-S (small).
Семантична сегментація працює на нативній квадратній сітці патчів DINOv2;
класифікація та ембединг працюють із меншою роздільною здатністю класифікації,
використаною для навчання лінійного пробника.

## Навчання

Обидва завдання `task="semantic"` і `task="classify"` навчаються; `task="embed"`
не має залежної від класів голови для підлаштування та спричиняє
`NotImplementedError`, якщо викликати для нього `train()`.

<code-tabs name="train" />

Основними іменованими аргументами тут є `batch_size` і `lr`, а не `batch` і
`lr0`, які використовує більшість інших сімейств; `batch` і `lr0` усе одно
приймаються та зіставляються з ними, але передавання обох спричиняє помилку
конфлікту. `output_dir=` (типове значення `"runs/train"`) замінює
`project=`/`name=` як основний спосіб розміщення запуску, хоча пряме передавання
`project=`/`name=` усе одно працює. Датасети, аугментацію, кілька GPU та системи
журналювання описано на сторінці [навчання](/docs/train).

## Валідація

Метод `val()` повертає словник ключів `metrics/`: mIoU і правильність за пікселями
для `task="semantic"`, правильність top-1 і top-5 для `task="classify"`.
Завдання `task="embed"` не має еталонних даних для оцінювання та спричиняє
`NotImplementedError`, якщо викликати для нього `val()`.

<code-tabs name="val" />

## Експорт

<export-matrix />

Кожне завдання підтримує власну підмножину форматів, наведену вище.
Експортований артефакт повторно завантажується через `LibreYOLO()` за суфіксом
файлу, тому файл `.onnx` або `.engine` поводиться як контрольна точка й повертає
той самий `Results`. На сторінці [Експорт](/docs/export) наведено аргументи, які
приймає кожен формат.

<code-tabs name="export" />

## Ліцензування

<provenance-box>

У наведеному вище рядку «Weights» зазначено застосовну ліцензію Apache-2.0, але
для цього сімейства організація LibreYOLO у Hugging Face насправді нічого не
публікує: LibreYOLO не розміщує власної контрольної точки LibreDINOv2. Виклик
`LibreDINOv2(model_path=None)` завантажує без змін власний репозиторій Meta
`facebook/dinov2-with-registers-small`.

</provenance-box>

## Цитування

<citation-block />

