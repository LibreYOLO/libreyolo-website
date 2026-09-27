---
title: TorchScript
seo_title: Экспорт в TorchScript из LibreYOLO
description: >-
  Экспорт модели LibreYOLO в TorchScript: трассированный архив .torchscript с
  метаданными LibreYOLO внутри, который загружается из Python или из libtorch.
lead: >-
  TorchScript — собственный формат сериализованного графа PyTorch. LibreYOLO
  трассирует модель через torch.jit.trace и сохраняет результат вместе с
  дополнительным файлом libreyolo_metadata.json, поэтому архив несёт семейство,
  задачу, имена классов и размер входа.
keywords:
  - экспорт yolo в torchscript
  - torch.jit.trace
  - torch.jit.load
  - libtorch c++ инференс
  - метаданные torchscript
  - extra_files
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="torchscript")
    mono: true
  - label: Записывает
    value: Один архив .torchscript с дополнительным файлом libreyolo_metadata.json
  - label: Дополнительно
    value: Ничего. TorchScript поставляется вместе с PyTorch.
  - label: Загружается обратно
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Формы
    value: Фиксированные. Граф трассируется на одной форме входа.
  - label: Точность
    value: 'FP32, FP16 (half=True). Без INT8.'
verification: >-
  Прочитано из libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py и libreyolo/backends/torchscript.py в ветке dev.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Аргументы
      language: python
      code: >
        model.export(
            format="torchscript",
            imgsz=640,        # int или (height, width)
            batch=1,
            half=False,       # веса и активации в FP16
            device=None,      # None трассирует на CPU для этого формата
            output_path=None, # None записывает weights/<stem>.torchscript
        )


        # dynamic принимается, но архив всегда остаётся трассировкой с
        фиксированной

        # формой, а встроенные метаданные в любом случае записывают
        dynamic=False.
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Голый PyTorch
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

        # Предобработка и постобработка на этом пути — на вас.
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Установка

<code-tabs name="install" />

TorchScript не требует ничего сверх базовой установки, потому что `torch.jit`
поставляется вместе с PyTorch. Это единственная цель экспорта без опциональной
зависимости и без внешнего конвертера, и потому удобная первая проверка, когда
падает более длинная цепочка инструментов.

## Экспорт

<code-tabs name="export" />

Трассировка выполняется на CPU, если устройство не названо явно, а архив пишется
в `weights/` под именем чекпойнта, когда `output_path` опущен.

Повторная проверка трассировки, которую `torch.jit.trace` обычно выполняет,
отключена. Несколько обёрток для экспорта кэшируют зависящие от формы якоря во
время первого прямого прохода, поэтому вторая трассировка видит другой путь по
коду Python, хотя записанный граф с фиксированной формой корректен. Вместо этого
тесты паритета проверяют сохранённый модуль напрямую.

Метаданные не лежат в отдельном sidecar-файле. `torch.jit.save` кладёт
`libreyolo_metadata.json` внутрь архива, а `torch.jit.load` возвращает его
обратно через `_extra_files`.

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` выбирает ветку по суффиксу `.torchscript` и возвращает тот же
объект `Results`, что и чекпойнт, из которого он получен. При `device="auto"`
модуль отображается на CUDA, если она доступна, затем на MPS, затем на CPU.

Второй сниппет — путь для читателя, у которого LibreYOLO не установлен, и для
развёртывания на C++ через libtorch, где тот же архив загружается через
`torch::jit::load`. Предобработка, декодирование, NMS и пересчёт координат там
ложатся на вас. Дополнительный файл метаданных по-прежнему читается, и только в
нём существуют имена классов.

## Ограничения

Граф — это трассировка на одной форме входа. `dynamic=True` принимается ради
симметрии интерфейса, но ничего не меняет, а встроенные метаданные сообщают
`dynamic=False`, чтобы бэкенд никогда не рассчитывал на ось, которой не может
воспользоваться. Для второго разрешения экспортируйте второй архив.

`half=True` приводит модель и вход трассировки к FP16. Пути через INT8 нет:
`int8=True` выбрасывает `NotImplementedError` при валидации.

Прямоугольный `imgsz` работает для семейств YOLO9, HRNet, NAFNet и Real-ESRGAN и
отклоняется для семейств с фиксированным квадратным контрактом.

Пять комбинаций отклоняются ещё до трассировки. Сегментация YOLO9 — потому что в
LibreYOLO YOLO9 умеет только детекцию. Сегментация RTMDet-Ins, у которой
декодирование масок с динамическими ядрами не имеет контракта для
экспортированной среды выполнения. Детекция SSD, Faster R-CNN и RetinaNet, чьи
графы с переменной длиной или с динамическими якорями имеют подтверждение
паритета только через контракт ONNX Runtime.

Полную сетку семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). Для одной комбинации:

<code-tabs name="support" />
