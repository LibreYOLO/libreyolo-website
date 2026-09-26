---
title: ncnn
seo_title: Экспорт в ncnn из LibreYOLO
description: >-
  Экспорт модели LibreYOLO в ncnn через PNNX: пара param и bin, фиксированный
  холст экспорта, замена слоя Focus в YOLOX и какие семейства конвертируются.
lead: >-
  ncnn — библиотека инференса на CPU от Tencent для мобильных устройств.
  LibreYOLO конвертирует через PNNX, записывая граф model.ncnn.param рядом с
  файлом весов model.ncnn.bin и файлом metadata.yaml, который несёт семейство,
  задачу и имена классов.
keywords:
  - экспорт yolo в ncnn
  - pnnx
  - model.ncnn.param
  - инференс на мобильном cpu
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: Флаг
    value: export(format="ncnn")
    mono: true
  - label: Записывает
    value: 'Директорию с model.ncnn.param, model.ncnn.bin и metadata.yaml'
  - label: Дополнительно
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Загружается обратно
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: Формы
    value: >-
      Фиксированные. В метаданных записывается dynamic=False независимо от
      флага.
  - label: Точность
    value: Только FP32. half=True и int8=True отклоняются.
verification: >-
  Прочитано из libreyolo/export/ncnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/ncnn.py и pyproject.toml в
  ветке dev.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        # pnnx конвертирует, ncnn запускает результат.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Записывает директорию weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Аргументы
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int или (высота, ширина)
            batch=1,
            simplify=True,    # действует только на запасном пути через ONNX
            opset=None,       # авто; действует только на запасном пути через ONNX
            output_path=None, # None записывает weights/<stem>_ncnn
        )

        # half=True и int8=True отклоняются при валидации.
  run:
    - label: Через LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Голый ncnn
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnn принимает одно изображение CHW, а не батч.
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Предобработка и постобработка на этом пути на вас.
  support:
    - label: Проверка одного семейства и задачи перед экспортом
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## Установка

<code-tabs name="install" />

Дополнительная зависимость подтягивает обе половины инструментария: `pnnx`
выполняет конвертацию, а `ncnn` исполняет результат. На основном пути ни один из
них не проходит через ONNX.

## Экспорт

<code-tabs name="export" />

Артефакт — это директория. В `weights/LibreYOLO9t_ncnn` лежат
`model.ncnn.param`, `model.ncnn.bin` и `metadata.yaml`; все три — один артефакт,
и переносить их нужно вместе.

Конвертация сначала пробует PNNX напрямую из PyTorch. Если это не срабатывает,
она экспортирует статический ONNX-граф во временную директорию и вызывает на нём
консольную утилиту `pnnx`, а ошибку экспорт поднимает только тогда, когда
провалились оба пути, и сообщает обе ошибки. Поэтому `opset` и `simplify` влияют
только на запасной путь.

Чтобы YOLOX вообще сконвертировался, нужна одна замена. Его слой Focus использует
нарезку с шагом, которую PNNX не умеет опустить до своих операций, поэтому
экспорт меняет его на `pixel_unshuffle` и переставляет входные каналы следующей
свёртки, компенсируя другой порядок каналов. Выход численно идентичен, а исходные
веса восстанавливаются после экспорта.

## Запуск артефакта

<code-tabs name="run" />

`LibreYOLO()` распознаёт любую директорию, в которой лежат `model.ncnn.param` и
`model.ncnn.bin`, читает `metadata.yaml` и возвращает тот же объект `Results`,
что и чекпойнт.

Второй сниппет — путь через голую среду выполнения, и две детали отличаются от
всех остальных форматов здесь. ncnn работает с одним изображением CHW, а не с
батчем, поэтому ведущей оси батча нет. Имена блобов берутся из файла `.param`;
PNNX по соглашению пишет `in0` и `out0`, но бэкенд разбирает файл, а не полагается
на это. Предобработка, декодирование, NMS и пересчёт координат на этом пути на
вас.

## Ограничения

FP32 на фиксированном холсте. `half=True` и `int8=True` отклоняются при
валидации, а в экспортированных метаданных записывается `dynamic=False`
независимо от того, что говорил флаг, чтобы ни один бэкенд не рассчитывал на ось,
которой в графе нет.

Каждое семейство в стиле DETR отклоняется на предварительной проверке: `detr`,
`deformable_detr`, `dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rfdetr` и `ec`. Сообщение у всех одинаковое: модели
нужны операции декодера или сэмплирования, которых в ncnn нет, и оно указывает
вместо этого на ONNX, OpenVINO, TorchScript или TensorRT.

Со стороны свёрточных моделей конвертируется многое: YOLO9 и YOLO9-E2E, YOLOX,
PicoDet, детекция и оценка позы в YOLO-NAS, более старые детекторы YOLO1, YOLO3,
YOLO4 и YOLO7, четыре семейства CNN-классификации, семантическая сегментация
PIDNet, детекция точек FOMO на фиксированных 96 на 96, ZipDepth, NAFNet и
Real-ESRGAN.

Заблокированные записи называют конкретную причину сбоя. Графы трансформеров
обычно оставляют после себя неподдерживаемые узлы `pnnx.Expression`, из-за чего
получается сеть без исполняемого входного блоба, — именно это останавливает
DINOv2, CLIP, SigLIP2 и SegFormer. BiRefNet нужна деформируемая свёртка из
torchvision, которую PNNX не умеет опустить до своих операций. Сконвертированный
граф YOLO2 завершает среду выполнения ncnn на Windows нативным целочисленным
делением на ноль во время извлечения выхода.

Полную сетку семейств и задач смотрите в
[матрице экспорта](/docs/reference/export-matrix). Для одной комбинации:

<code-tabs name="support" />
