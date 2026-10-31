---
title: Core AI
seo_title: Экспорт в Apple Core AI из LibreYOLO
description: >-
  Экспорт модели LibreYOLO в ассет .aimodel для Apple Core AI: только macOS,
  фиксированный холст, FP32 и контракт порядка именованных выходов, который
  обязаны соблюдать потребители.
lead: >-
  Core AI — стек инференса на устройстве от Apple. LibreYOLO захватывает модель
  через torch.export, понижает её представление конвертером Core AI и записывает
  ассет .aimodel с метаданными модели и именами экспортированных выходов.
keywords:
  - экспорт libreyolo в core ai
  - aimodel
  - coreai-torch
  - torch.export apple
  - инференс на устройстве apple
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="coreai")
    mono: true
  - label: Записывает
    value: Один ассет .aimodel с прикреплёнными метаданными
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Обратная загрузка
    value: >-
      Не через LibreYOLO. Потребители используют среду выполнения Core AI
      напрямую.
  - label: Формы
    value: Фиксированный холст. dynamic=True вызывает NotImplementedError.
  - label: Точность
    value: Только FP32. half=True и int8=True отклоняются.
  - label: Требуется
    value: >-
      macOS. Тулчейн нигде больше не конвертирует и не запускает модели, а
      coreai-torch фиксирует torch на 2.11.x.
verification: >-
  Прочитано из libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py и pyproject.toml в
  ветке dev.
snippets:
  install:
    - label: 'Установка, на macOS'
      language: bash
      code: |
        # Намеренно не входит ни в один сводный extra: coreai-torch фиксирует
        # torch на 2.11.x и утянул бы всё окружение на эту версию.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int или (высота, ширина); это холст запуска
            batch=1,
            output_path=None, # None записывает weights/<stem>.aimodel
        )

        # dynamic=True вызывает NotImplementedError.
        # half=True и int8=True отклоняются на этапе валидации.
  outputs:
    - label: Чтение порядка выходов перед подключением потребителя
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")

        model.export(format="coreai", imgsz=640)


        # Метаданные ассета записывают имена экспортированных выходов в порядке

        # графа под ключом "coreai_output_names". Сопоставляйте по имени
        словарь,

        # который возвращает Core AI, используя этот список; никогда не
        связывайте

        # его по позиции с кортежем eager-режима.
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## Установка

Этот формат работает только на macOS. У зависимости `coreai-torch` стоит маркер
`sys_platform == 'darwin'`, и тулчейн нигде больше не конвертирует и не
запускает модели.

<code-tabs name="install" />

Этот extra не входит ни в один сводный extra, включая `libreyolo[all]`, потому
что `coreai-torch` фиксирует torch на серии 2.11. Ставьте его в то окружение,
которое вы готовы ограничить этой парой.

## Экспорт

<code-tabs name="export" />

Захват выполняется через `torch.export` — это настоящий захват графа с
guard-условиями, а не одна записанная трассировка. Такой путь строже, чем у
Core ML: чтения скалярных значений на хосте и зависящий от данных поток
управления отклоняются, а не запекаются молча, и именно поэтому несколько
семейств здесь заблокированы с записанной ошибкой захвата.

Три шага подготовки выполняются внутри области видимости, которая восстанавливает
живую модель вызывающей стороны независимо от того, удался экспорт или нет. У
семейств, производных от Darknet, батч-нормализация в режиме инференса точно
сворачивается в предшествующие свёртки, потому что Core AI 0.4.1 не сохраняет
формулу Darknet с эпсилон после квадратного корня. У семейств с сеткой и якорями
якоря замораживаются под фиксированный холст. У RF-DETR позиционный эмбеддинг
перезапекается под запрошенный холст повторным прогоном собственного пути
запекания модели, потому что у конвертера нет понижения для
`aten._upsample_bicubic2d_aa`.

На этапе понижения эталонная декомпозиция PyTorch для `aten.grid_sampler_2d`
добавляется в таблицу декомпозиций, поскольку у конвертера Core AI нет понижения
для сэмплера deformable attention, который используют семейства DETR.

Ассеты объявляют минимальную версию ОС v27 — единственное значение, которое
предлагает тулчейн. Это ограничивает развёртывание, а не конвертацию:
конвертация и запуск на стороне Python работают и на более ранних macOS через
среду выполнения внутри wheel-пакета, но численные результаты различаются между
версиями ОС, поэтому записанный паритет измеряется на macOS 27.

## Запуск артефакта

В `libreyolo/backends` нет записи для Core AI, поэтому `LibreYOLO()` не загружает
`.aimodel`. Потребители используют среду выполнения Core AI напрямую, а
предобработка, декодирование, NMS и пересчёт координат — на их стороне.
Проверенная строка в матрице поддержки означает, что экспортированный граф
считает те же числа, что и эталон, а не то, что его запустит `predict`.

Единственное, что потребитель не может вывести сам, — это порядок выходов:

<code-tabs name="outputs" />

Core AI возвращает именованный словарь, порядок ключей в котором не совпадает ни
с порядком кортежа из eager-прохода вперёд, ни с чем-либо угадываемым. Именно
поэтому экспортированные имена записываются в метаданные ассета как
`coreai_output_names`. Сопоставляйте по имени.

## Ограничения

Фиксированный холст, FP32, батч такой, каким его экспортировали. `dynamic=True`
вызывает `NotImplementedError`, а `half=True` и `int8=True` отклоняются на этапе
валидации.

Покрытие на стороне конвертации широкое. В число проверенных комбинаций входят
детекция в семействах YOLO9, YOLOX, YOLO7, четырёх детекторах эпохи Darknet,
YOLO-NAS, PicoDet, RTMDet, RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2,
EC и RF-DETR; четыре CNN-семейства классификации плюс CLIP и SigLIP2 с
фиксированными классами; Depth Anything V2 и ZipDepth; восстановление NAFNet и
Real-ESRGAN; семантическая сегментация PIDNet и LingBotVision; детекция точек
FOMO. У каждой комбинации свой записанный контекст, который печатает
`libreyolo formats`.

Заблокировано, с причиной, записанной для каждой комбинации:

| Комбинация | Почему |
|---|---|
| Семантическая сегментация EoMT | Строгий захват падает с `GuardOnDataDependentSymNode`: что-то в пути масок читает значение из тензора и ветвится по нему |
| Семантическая сегментация SegFormer | Путь захвата не оценивали, а опубликованные веса всё равно некоммерческие независимо от формата |
| Оценка взгляда L2CS | Сама модель поддерживает только ONNX, TorchScript, ExecuTorch, TensorRT и OpenVINO — это решение на стороне модели |
| Оценка глубины Depth Anything 3 | Семейство отклоняет экспорт во все форматы |

С RF-DETR связана одна оговорка, которую стоит прочитать до сравнения
артефактов. Паритет для него записан относительно графа, который готовит сам
экспортёр Core AI, а не относительно ONNX, и на холсте 640 ONNX-артефакт RF-DETR
расходится с этим подготовленным графом. Перезапекание в Core AI сохраняет ресайз
со сглаживанием, который выполняет eager-модель, тогда как путь ONNX сглаживание
отключает. Поэтому ONNX — некорректный эталон для этого семейства на неродном
холсте.

О более раннем формате Apple читайте в [Core ML](/docs/export/coreml). Полную
таблицу семейств и задач смотрите в [матрице экспорта](/docs/reference/export-matrix).
Для одной комбинации:

<code-tabs name="support" />
