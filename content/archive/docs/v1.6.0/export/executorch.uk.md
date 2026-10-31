---
title: ExecuTorch
seo_title: Експорт до ExecuTorch з LibreYOLO
description: >-
  Експорт моделі LibreYOLO у програму .pte для ExecuTorch з делегуванням
  XNNPACK: фіксована форма, батч 1, FP32 і потрібний для неї sidecar-файл
  метаданих.
lead: >-
  ExecuTorch виконує програми PyTorch на edge-пристроях. LibreYOLO захоплює
  модель за допомогою torch.export у строгому режимі, виконує lowering до
  XNNPACK і фіксує програму .pte разом із sidecar-файлом метаданих JSON як одне
  ціле.
keywords:
  - експорт yolo executorch
  - програма .pte
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - інференс pytorch на периферії
last_verified: 1.5.0
meta:
  - label: Прапорець
    value: export(format="executorch")
    mono: true
  - label: Записує
    value: Одну програму .pte плюс sidecar-файл метаданих .pte.json
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Зворотне завантаження
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: Форми
    value: Фіксовані. dynamic=True і batch != 1 відхиляються.
  - label: Точність
    value: Лише FP32. half=True та int8=True відхиляються.
  - label: Делегат
    value: 'XNNPACK, CPU. Єдине прийнятне значення: delegate=''xnnpack''.'
verification: >-
  Прочитано з libreyolo/export/executorch.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/executorch.py і pyproject.toml
  у гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: |
        # Навмисно поза libreyolo[all]: ExecuTorch обмежує, з якою
        # версією Torch його можна поєднувати.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записує weights/LibreYOLO9t.pte і weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int або (висота, ширина)
            batch=1,               # будь-яке інше значення викликає ValueError
            dynamic=False,         # True викликає ValueError
            delegate="xnnpack",    # єдине прийнятне значення
            device="cpu",          # будь-який інший пристрій викликає ValueError
            output_path=None,      # None записує weights/<stem>.pte
        )
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Чисте середовище виконання ExecuTorch
      language: python
      code: >
        import json

        from pathlib import Path


        import torch

        from executorch.runtime import Runtime


        runtime = Runtime.get()

        print(runtime.backend_registry.is_available("XnnpackBackend"))


        program =
        runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())

        method = program.load_method("forward")


        # На цьому шляху попередня і подальша обробка лишаються за вами.

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Перевірити одне сімейство і задачу перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## Встановлення

<code-tabs name="install" />

Цей extra навмисно винесено за межі `libreyolo[all]`, бо ExecuTorch фіксує версію
Torch, з якою він працює, і його встановлення потягнуло б усе середовище на цю
пару. Встановлюйте його в середовище, яке ви готові обмежити.

У Windows крок lowering викликає виконуваний файл `flatc`, який постачається
разом з ExecuTorch. Якщо його немає в `PATH`, експорт викидає `RuntimeError` з
відповідним повідомленням, а розв'язанням є запуск із Visual Studio 2022
Developer PowerShell.

## Експорт

<code-tabs name="export" />

Захоплення виконує `torch.export.export(..., strict=True)`, тобто справжнє
захоплення графа з перевірками (guards), а не записаний трейс. Читання скалярів
на хості та потік керування, залежний від даних, відхиляються, замість того щоб
мовчки зафіксуватися в графі, тож кілька сімейств зазнають невдачі тут, хоча в
інших місцях трасуються успішно; причини записано для кожної комбінації в
матриці підтримки.

Lowering запускає `to_edge_transform_and_lower` з партиціонером XNNPACK. Якщо
результат не містить жодного делегованого розділу, експорт завершується
помилкою, замість того щоб позначити як XNNPACK програму, зібрану лише з
портативних ядер.

Програму і sidecar-файл фіксують разом. Обидва готуються проміжно, обидва
підставляються на місце, а збій відкочує все до попереднього стану, тож неповна
пара ніколи не потрапляє на диск.

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` розпізнає суфікс `.pte` і повертає той самий об'єкт `Results`, що й
контрольна точка. Sidecar-файл обов'язковий під час завантаження: без
`<program>.pte.json` бекенд викидає `FileNotFoundError`, бо сама програма не
несе ні назв класів, ні задачі, ні власного розміру входу. Перед завантаженням
бекенд також перевіряє, що встановлене середовище виконання надає
`XnnpackBackend`, і читає програму з байтів, а не відображає файл у пам'ять, що
дозволяє не тримати файлове блокування Windows протягом усього часу життя
бекенда.

Другий фрагмент показує шлях через чисте середовище виконання. Попередня
обробка, декодування, NMS і масштабування координат стають там вашою
відповідальністю.

## Обмеження

Батч 1, фіксована форма, FP32, CPU. І `batch != 1`, і `dynamic=True` викидають
`ValueError` ще до того, як експорт щось змінить, `half=True` та `int8=True`
відхиляються під час перевірки, а пристрій, відмінний від CPU, не приймається.

У цій версії `delegate` приймає `"xnnpack"` і нічого іншого.

Експорти для класифікації несуть два додаткові ключі метаданих, `crop_pct` та
`interpolation`, щоб середовище виконання могло відтворити політику зміни
розміру та центрального кадрування, властиву сімейству.

Заблоковані записи називають конкретний збій, а не категорію. Виявлення та
сегментація в моделі D-FINE натрапляють на непідтримуване читання `ContextVar` у
deformable attention при строгому захопленні, а примусовий ручний шлях через
grid-sample серіалізується, але потім падає під час виконання через некоректний
порядок вимірів делегованого тензора. Моделі DEIM і DEIMv2 захоплюються,
проходять lowering і серіалізуються, а потім зазнають збою під час виконання.
Семантична сегментація в моделі EoMT падає на символьному виразі, залежному від
даних, у шляху масок. Matting у моделі BiRefNet захоплюється на 1024 на 1024,
але не має out-варіанта для `torchvision::deform_conv2d`. Відновлення в моделі
SwinIR перезавантажується, а потім падає в `aten::alias_copy.out` через
невідповідність порядків вимірів.

Повну сітку сімейств і задач дивіться в
[матриці експорту](/docs/reference/export-matrix). Для однієї комбінації:

<code-tabs name="support" />
