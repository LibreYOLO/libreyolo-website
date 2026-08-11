---
title: TorchScript
seo_title: Експорт до TorchScript з LibreYOLO
description: >-
  Експортуйте модель LibreYOLO до TorchScript: трасований архів .torchscript із
  метаданими LibreYOLO всередині, який можна завантажити з Python або libtorch.
lead: >-
  TorchScript є власним форматом серіалізованого графа PyTorch. LibreYOLO трасує
  модель за допомогою torch.jit.trace і зберігає результат разом із додатковим
  файлом libreyolo_metadata.json, тому архів містить сімейство, завдання, назви
  класів і розмір входу.
keywords:
  - експорт yolo torchscript
  - torch.jit.trace
  - torch.jit.load
  - розгортання libtorch
  - метадані torchscript
  - extra_files
last_verified: 1.5.0
meta:
  - label: Прапорець
    value: export(format="torchscript")
    mono: true
  - label: Створює
    value: Один архів .torchscript із додатковим файлом libreyolo_metadata.json
  - label: Додатково
    value: Нічого. TorchScript постачається з PyTorch.
  - label: Повторне завантаження
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Форми
    value: Фіксовані. Граф трасується для однієї форми входу.
  - label: Точність
    value: 'FP32, FP16 (half=True). Без INT8.'
verification: >-
  Перевірено за файлами libreyolo/export/torchscript.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py і
  libreyolo/backends/torchscript.py у гілці dev.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Створює weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Аргументи
      language: python
      code: >
        model.export(
            format="torchscript",
            imgsz=640,        # ціле число або (висота, ширина)
            batch=1,
            half=False,       # ваги й активації FP16
            device=None,      # для цього формату None трасує на CPU
            output_path=None, # None створює weights/<stem>.torchscript
        )


        # dynamic приймається, але архів завжди містить трасування з фіксованою
        формою,

        # а вбудовані метадані в будь-якому разі записують dynamic=False.
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Безпосередньо через PyTorch
      language: python
      code: |
        import json

        import torch

        extra_files = {"libreyolo_metadata.json": ""}
        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )
        module.eval()

        metadata = json.loads(extra_files["libreyolo_metadata.json"])
        print(metadata["model_family"], metadata["task"], metadata["imgsz"])

        # На цьому шляху попередня та подальша обробка покладаються на вас.
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Перевірити сімейство й завдання перед експортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Встановлення

<code-tabs name="install" />

Для TorchScript не потрібно нічого, крім базового встановлення, оскільки
`torch.jit` постачається з PyTorch. Це єдина ціль експорту без додаткової
залежності та зовнішнього конвертера, тому вона корисна для першої перевірки,
коли складніший ланцюжок інструментів дає збій.

## Експорт

<code-tabs name="export" />

Трасування виконується на CPU, якщо пристрій не вказано. Коли `output_path`
пропущено, архів записується в `weights/` під основою назви контрольної точки.

Повторну перевірку трасування, яку зазвичай виконує `torch.jit.trace`, вимкнено.
Деякі обгортки експорту кешують залежні від форми якорі під час першого прямого
проходу, тому друге трасування спостерігає інший шлях Python, хоча записаний граф
із фіксованою формою правильний. Натомість тести паритету перевіряють безпосередньо
збережений модуль.

Метадані не зберігаються в супровідному файлі. `torch.jit.save` записує
`libreyolo_metadata.json` усередину архіву, а `torch.jit.load` повертає його через
`_extra_files`.

## Запуск артефакту

<code-tabs name="run" />

`LibreYOLO()` визначає маршрут за суфіксом `.torchscript` і повертає той самий
об'єкт `Results`, що й початкова контрольна точка. З `device="auto"` модуль
зіставляється спочатку з CUDA, якщо вона доступна, потім із MPS, а тоді з CPU.

Другий фрагмент призначено для читача без установленої LibreYOLO та для
розгортання на C++ через libtorch, де той самий архів завантажується за допомогою
`torch::jit::load`. На цьому шляху попередня обробка, декодування, NMS і
масштабування координат покладаються на вас. Додатковий файл метаданих усе ще
можна прочитати, і лише в ньому зберігаються назви класів.

## Обмеження

Граф є трасуванням для однієї форми входу. `dynamic=True` приймається для
симетрії інтерфейсу, але нічого не змінює, а вбудовані метадані повідомляють
`dynamic=False`, щоб бекенд не припускав наявність осі, яку не може використати.
Для іншої роздільної здатності експортуйте другий архів.

`half=True` перетворює модель і вхід трасування на FP16. Шляху INT8 немає:
`int8=True` спричиняє `NotImplementedError` під час валідації.

Прямокутний `imgsz` працює для сімейств YOLO9, HRNet, NAFNet і Real-ESRGAN, але
відхиляється для сімейств із фіксованим контрактом квадратного входу.

Перед трасуванням відхиляються п'ять комбінацій. Сегментація YOLO9, оскільки в
LibreYOLO сімейство YOLO9 підтримує лише виявлення. Сегментація RTMDet-Ins, бо
декодування масок із динамічним ядром не має контракту для експортованого
середовища виконання. Виявлення SSD, Faster R-CNN і RetinaNet, адже їхні графи зі
змінною довжиною або динамічними якорями мають підтвердження паритету лише через
контракт ONNX Runtime.

Повну таблицю сімейств і завдань наведено в
[матриці експорту](/docs/reference/export-matrix). Для однієї комбінації:

<code-tabs name="support" />
