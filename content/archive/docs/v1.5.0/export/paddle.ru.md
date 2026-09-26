---
title: Paddle
seo_title: Экспорт в PaddlePaddle из LibreYOLO
description: >-
  Конвертация детектора LibreYOLO в модель для инференса PaddlePaddle через
  X2Paddle: закреплённый стек инструментов, статические графы FP32 с батчем 1 и
  инференс на CPU.
lead: >-
  Модель для инференса PaddlePaddle — это граф model.pdmodel рядом с файлом
  весов model.pdiparams. LibreYOLO экспортирует статический ONNX-граф с opset
  15, конвертирует его через X2Paddle и упаковывает результат вместе с
  metadata.yaml, поэтому он загружается через ту же фабрику, что и любая другая
  среда выполнения.
keywords:
  - экспорт yolo в paddle
  - paddlepaddle инференс
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="paddle")
    mono: true
  - label: Записывает
    value: 'Директорию с model.pdmodel, model.pdiparams и metadata.yaml'
  - label: Дополнительно
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Загружается обратно
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Бэкенд
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Формы
    value: 'Статические, батч 1, opset 15. Все три требования жёстко проверяются.'
  - label: Точность
    value: 'Только FP32, только CPU.'
  - label: Стек инструментов
    value: >-
      PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 или ниже, проверяется в
      точности
verification: >-
  Прочитано из libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md и
  pyproject.toml в ветке dev.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        # Python 3.10–3.12. Проверенный путь для Windows — WSL2 с Ubuntu 22.04.
        pip install "libreyolo[paddle]"
    - label: Проверка закреплённых версий
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает директорию weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; квадратный холст этого семейства
            batch=1,          # любое другое значение вызывает ValueError
            dynamic=False,    # True вызывает ValueError
            simplify=True,    # False вызывает ValueError
            opset=15,         # любое другое значение вызывает ValueError
            output_path=None, # None записывает weights/<stem>_paddle
        )
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: Бэкенд напрямую
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # То, что LibreYOLO() создаёт для директории Paddle. Тот же объект
        # Results, без маршрутизации через фабрику.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Чистый Paddle
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Предобработка и постобработка на этом пути — на вас.
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Установка

<code-tabs name="install" />

Дополнительная зависимость закрепляет ровно тот стек, на котором измерялся
паритет: PaddlePaddle 2.6.2, X2Paddle 1.6.0 и ONNX 1.17 или ниже. Эти
закрепления проверяются при экспорте, а не только при установке, и другая версия
вызывает `ImportError` с указанием ожидаемой. Новые релизы Paddle отвергают часть
статического кода, который генерирует X2Paddle 1.6.0, поэтому упасть сразу лучше,
чем выдать артефакт, который никто не проверял.

## Экспорт

<code-tabs name="export" />

Четыре аргумента зафиксированы, а не просто имеют значения по умолчанию.
`dynamic` должен быть `False`, `batch` — 1, `simplify` — `True`, чтобы граф
конвертации был полностью статическим, а `opset` — 15: это потолок, который
принимает X2Paddle 1.6.0. Любое другое значение вызывает ошибку ещё до
трассировки.

На промежуточном графе выполняется одна нормализация. ONNX считает пропущенный
dilation у MaxPool равным единице, PyTorch записывает явный атрибут из одних
единиц, а X2Paddle 1.6.0 его отвергает, поэтому экспортёр убирает это избыточное
значение по умолчанию и оставляет заданную операцию без изменений.

Артефакт — это директория: `model.pdmodel`, `model.pdiparams` и
`metadata.yaml`. Python-код, который X2Paddle генерирует во время конвертации, в
неё не входит.

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` распознаёт любую директорию, где лежат и `model.pdmodel`, и
`model.pdiparams`, читает `metadata.yaml` и возвращает тот же объект `Results`,
что и чекпойнт. Устройство, отличное от `auto` или `cpu`, вызывает ошибку: этот
бэкенд работает только на CPU.

Фабрика создаёт `PaddleBackend` — он экспортируется из `libreyolo` и импортируется
как `libreyolo.backends.paddle.PaddleBackend`. Создавайте его сами, когда нужен
бэкенд без маршрутизации по суффиксам, которую делает фабрика, — например, чтобы
явно передать `task=` для директории, чей `metadata.yaml` писали не вы. Его
`predict()` принимает те же источники и возвращает те же результаты.

Сниппет с чистой средой выполнения повторяет то, что настраивает бэкенд, а три
отключённые опции выключены намеренно. Пайплайн CPU-фьюзинга в Paddle 2.6 может
падать при оптимизации больших графов gather и scatter, которые порождает
deformable attention, поэтому паритет измерялся на переносимом статическом графе
без фьюзинга. Предобработка, декодирование, NMS и пересчёт координат на этом пути
ложатся на вас.

## Ограничения

Ни динамических форм, ни FP16, ни INT8, ни встроенного NMS, ни GPU-среды
выполнения.

Проверенные комбинации: детекция YOLO9, детекция YOLO9-E2E и YOLO9-P2, детекция,
оценка позы и сегментация EC, детекция RT-DETRv4, D-FINE, DEIM и DEIMv2, а также
детекция и оценка позы YOLO-NAS. Для каждой проверены конвертация, повторная
загрузка в CPU-среде выполнения, паритет сырых выходов и совпадение с
опубликованными результатами.

Заблокированы, с причиной, записанной для каждой комбинации:

| Комбинация | Причина |
|---|---|
| RF-DETR, все задачи | Нужны ONNX opset 17 и GridSample; X2Paddle 1.6.0 принимает opset 15 или ниже и не имеет маппера для GridSample |
| Детекция RT-DETR и RT-DETRv2 | Обученным графам нужен GridSample на opset 16 или новее |
| Сегментация D-FINE | Конвертируется и загружается обратно, но относительная RMS-ошибка логитов маски — 3.52%, а минимальный IoU по сопоставленным маскам — 0.582 |
| Сегментация YOLO9 | В LibreYOLO YOLO9 — только детекция |
| Сегментация RTMDet-Ins | У декодирования маски с динамическим ядром нет контракта для экспортированных сред выполнения |

Всё, что не указано как проверенное или заблокированное, отклоняется с
примечанием, что комбинация не проверялась на пути конвертации из ONNX в Paddle.

Полная таблица семейств и задач — [матрица экспорта](/docs/reference/export-matrix).
Для одной комбинации:

<code-tabs name="support" />
