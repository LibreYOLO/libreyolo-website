---
title: ExecuTorch
seo_title: Экспорт в ExecuTorch из LibreYOLO
description: >-
  Экспорт модели LibreYOLO в программу ExecuTorch .pte с делегированием на
  XNNPACK: фиксированная форма, батч 1, FP32 и нужный ей sidecar-файл
  метаданных.
lead: >-
  ExecuTorch запускает программы PyTorch на периферийных устройствах. LibreYOLO
  захватывает модель через torch.export в строгом режиме, понижает её на XNNPACK
  и фиксирует программу .pte вместе с JSON-файлом метаданных как одно целое.
keywords:
  - экспорт yolo в executorch
  - executorch pte
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - pytorch инференс на edge
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="executorch")
    mono: true
  - label: Записывает
    value: Одну программу .pte плюс sidecar-файл метаданных .pte.json
  - label: Дополнительно
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Загружается обратно
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: Формы
    value: Фиксированные. dynamic=True и batch != 1 отклоняются.
  - label: Точность
    value: Только FP32. half=True и int8=True отклоняются.
  - label: Делегат
    value: 'XNNPACK, CPU. delegate=''xnnpack'' — единственное допустимое значение.'
verification: >-
  Прочитано из libreyolo/export/executorch.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/executorch.py и pyproject.toml
  в ветке dev.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        # Намеренно вынесено из libreyolo[all]: ExecuTorch ограничивает, с какой
        # версией Torch его можно сочетать.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает weights/LibreYOLO9t.pte и weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int или (height, width)
            batch=1,               # любое другое значение вызывает ValueError
            dynamic=False,         # True вызывает ValueError
            delegate="xnnpack",    # единственное допустимое значение
            device="cpu",          # любое другое устройство вызывает ValueError
            output_path=None,      # None записывает weights/<stem>.pte
        )
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Голая среда выполнения ExecuTorch
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


        # Предобработка и постобработка на этом пути — на вас.

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## Установка

<code-tabs name="install" />

Этот extra намеренно вынесен за пределы `libreyolo[all]`, потому что ExecuTorch
жёстко фиксирует версию Torch, с которой работает, и его установка утянула бы
всё окружение на эту пару. Ставьте его в окружение, которое вы готовы
ограничить.

На Windows шаг понижения вызывает исполняемый файл `flatc`, который поставляется
вместе с ExecuTorch. Если его нет в `PATH`, экспорт выбрасывает `RuntimeError` с
соответствующим сообщением, а лечится это запуском из Developer PowerShell для
Visual Studio 2022.

## Экспорт

<code-tabs name="export" />

Захват — это `torch.export.export(..., strict=True)`, настоящий захват графа с
guard-проверками, а не записанная трассировка. Чтение скалярных значений на
хосте и зависящий от данных поток управления отклоняются, а не запекаются молча
в граф, поэтому здесь падают несколько семейств, которые в других форматах
трассируются успешно; причины записаны для каждой комбинации в матрице
поддержки.

Понижение выполняет `to_edge_transform_and_lower` с партиционером XNNPACK. Если
в результате ноль делегированных разделов, экспорт выбрасывает ошибку, а не
выдаёт за XNNPACK программу, собранную только из портируемых ядер.

Программа и sidecar-файл фиксируются вместе. Оба сначала пишутся во временные
файлы, оба подменяются разом, а сбой откатывает всё к тому, что было раньше, так
что неполная пара никогда не попадает на диск.

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` выбирает ветку по суффиксу `.pte` и возвращает тот же объект
`Results`, что и чекпойнт. При загрузке sidecar-файл обязателен: без
`<program>.pte.json` бэкенд выбрасывает `FileNotFoundError`, потому что сама
программа не несёт ни имён классов, ни задачи, ни размера входа. Ещё бэкенд
перед загрузкой проверяет, что установленная среда выполнения предоставляет
`XnnpackBackend`, и читает программу из байтов, а не отображает файл в память,
что избавляет от удержания файловой блокировки Windows на всё время жизни
бэкенда.

Второй сниппет — путь через голую среду выполнения. Предобработка,
декодирование, NMS и пересчёт координат там ложатся на вас.

## Ограничения

Батч 1, фиксированная форма, FP32, CPU. `batch != 1` и `dynamic=True` оба
выбрасывают `ValueError` ещё до того, как экспорт что-либо изменит, `half=True`
и `int8=True` отклоняются при валидации, а устройство, отличное от CPU, не
принимается.

`delegate` в этой версии принимает `"xnnpack"` и ничего больше.

Экспорты классификации несут два дополнительных ключа метаданных, `crop_pct` и
`interpolation`, чтобы среда выполнения могла воспроизвести принятые в семействе
правила изменения размера и центрального кропа.

Заблокированные комбинации называют конкретный сбой, а не категорию. Детекция и
сегментация D-FINE упираются в неподдерживаемое чтение `ContextVar` в deformable
attention при строгом захвате, а принудительный ручной путь через grid-sample
сериализуется, но затем падает во время выполнения на недопустимом порядке
размерностей делегированного тензора. DEIM и DEIMv2 захватываются, понижаются и
сериализуются, а потом падают при выполнении. Семантическая сегментация EoMT
падает на зависящем от данных символьном выражении в ветке масок. Маттинг
BiRefNet захватывается при 1024 на 1024, но не имеет out-варианта для
`torchvision::deform_conv2d`. Восстановление SwinIR перезагружается, а затем
падает в `aten::alias_copy.out` из-за несовпадающих порядков размерностей.

Полную сетку семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). Для одной комбинации:

<code-tabs name="support" />
