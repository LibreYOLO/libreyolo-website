---
title: Hailo
seo_title: Запуск моделей LibreYOLO на ускорителях Hailo
description: >-
  Развёртывание модели LibreYOLO на Hailo-8 или Hailo-8L: статический экспорт в
  ONNX, этап Dataflow Compiler, который вы запускаете сами, и какие архитектуры
  компилируются.
lead: >-
  Ускорители Hailo программируются через Hailo Dataflow Compiler — проприетарный
  SDK, который распространяется через Developer Zone компании Hailo. Со стороны
  LibreYOLO в этом процессе только обычный статический экспорт в ONNX; парсинг,
  квантизация и компиляция в HEF происходят уже потом, внутри DFC.
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - hailo dataflow compiler
  - запуск yolo на hailo
  - компиляция hef
  - hailortcli
last_verified: 1.5.0
meta:
  - label: Шаг LibreYOLO
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: Не формат
    value: Формата format="hef" нет. DFC не может быть pip-зависимостью.
  - label: Дополнительно
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Хост для компиляции
    value: 'Linux x86_64, включая WSL2 Ubuntu 22.04. На ARM компиляция не запускается.'
  - label: Компилируется
    value: >-
      Чисто свёрточные графы с фиксированными формами. Attention, динамические
      формы и архитектуры на LayerNorm — нет.
  - label: Статус
    value: >-
      Ни одно семейство LibreYOLO ещё не проведено через DFC от начала до конца
      до работающего HEF.
verification: >-
  Прочитано из skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py
  и libreyolo/cli/commands/export.py в ветке dev. Ограничения DFC — те, что
  записаны в этом skill-файле; ни один HEF для LibreYOLO не скомпилирован и не
  измерен.
snippets:
  install:
    - label: Со стороны LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Со стороны Hailo, устанавливаете сами'
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Hailo нужен батч 1, фиксированное разрешение и никаких динамических
        осей.

        # В Python API по умолчанию dynamic=True, поэтому выключите его явно.

        model = LibreYOLO("LibreYOLOXs.pt")

        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # В CLI по умолчанию уже статические формы.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Проверка статичности графа перед компиляцией
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 'Парсинг, квантизация и компиляция'
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # Для YOLOX один раз выполните трансляцию без end_node_names: DFC

        # напечатает в логе end-узлы, которые он предлагает. Перезапустите с
        ними.

        runner.translate_onnx_model(ONNX)


        # Нормализация должна совпадать с предобработкой LibreYOLO. YOLOX и
        YOLO9

        # не нужны ни среднее, ни стандартное отклонение, только масштаб 0-255 в
        0-1.

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # Необязательно: отдать NMS на сторону Hailo. Конфигурация зависит и от

        # числа классов, и от размера входа, поэтому конфиг для COCO-80 не
        подойдёт

        # дообученной модели на три класса. Без этой строки HEF выдаёт сырые

        # тензоры головы, а декодирует их приложение.

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # Калибровочные изображения должны быть представительны для тех данных,

        # на которых модель будет работать. Случайные картинки скомпилируются

        # и молча уничтожат точность.

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: End-узлы YOLO9
      language: python
      code: >
        # В графах LibreYOLO используется префикс "/head/...", а не префикс

        # "model.N" из конфигураций, написанных под другие экспорты.
        Скопированный

        # конфиг не совпадёт. Если парсинг падает, проверьте имена в своём
        графе.

        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]

        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 с AI Kit или AI HAT+
      language: bash
      code: >
        sudo apt install dkms hailo-all

        hailortcli fw-control identify       # проверка устройства, заодно
        назовёт архитектуру

        hailortcli run libreyoloxs.hef       # smoke-тест и пропускная
        способность
source_hash: 33b077f1c23d5535
---

## Установка

В LibreYOLO нет `format="hef"` и не будет. Hailo Dataflow Compiler — проприетарный
SDK, который распространяется как приватный wheel и требует регистрации в
Developer Zone, поэтому он не может быть ни зависимостью, ни extra. Развёртывание
идёт в два этапа: LibreYOLO записывает статический ONNX-файл, а вы прогоняете по
нему DFC.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Экспорт

<code-tabs name="export" />

Не передавайте `half=True`. DFC принимает ONNX в FP32 и делает собственную
INT8-квантизацию. `nms=True` тоже не передавайте: NMS берёт на себя либо Hailo
через `nms_postprocess`, либо приложение, а подграф NMS за end-узлами — мёртвый
груз. Опсет по умолчанию работает; если парсер DFC возражает, переэкспортируйте с
`opset=11`.

DFC обрезает граф по тем end-узлам, которые вы указали, — это свёртки
детекционной головы, — и отбрасывает всё, что ниже. Поэтому обычный
декодированный ONNX от LibreYOLO подходит на вход: хвост с декодированием парсер
просто игнорирует.

## Компиляция

<code-tabs name="compile" />

Выберите `hw_arch` под целевое устройство: `hailo8` — для Hailo-8, AI HAT+ на
26 TOPS и модулей M.2 и PCIe; `hailo8l` — для Hailo-8L, Raspberry Pi AI Kit и
AI HAT+ на 13 TOPS; `hailo10h` — для Hailo-10H, которому нужны более новые DFC и
Model Zoo. Если вы не уверены, `hailortcli fw-control identify` на устройстве
отвечает на этот вопрос.

Два семейства ложатся на мета-архитектуру NMS в HailoRT, поэтому подавление Hailo
может взять на себя внутри скомпилированного пайплайна: YOLOX — через
`meta_arch=yolox`, а YOLO9 — через мета-архитектуру Hailo с разделённой головой, у
которой раскладка головы точно такая же. Возьмите подходящую конфигурацию
`nms_postprocess` из Hailo Model Zoo и поправьте её под своё число классов и размер
входа. Все остальные свёрточные детекторы компилируются как граф без подходящей
мета-архитектуры: HEF выдаёт сырые тензоры головы, а декодирование и NMS
приложение выполняет на CPU.

Сохраняйте лог компиляции, когда что-то падает. Любое исправление держится на
точном имени слоя или оператора, на котором всё сломалось.

## Запуск артефакта

<code-tabs name="device" />

Инференс в приложении идёт через Python API `hailo_platform`. Если
`nms_postprocess` вкомпилирован, выход имеет форму
`(batch, num_classes, max_dets, 5)` и несёт `[y1, x1, y2, x2, score]` в
координатах модели, которые вы сами масштабируете обратно к исходному
изображению. Пайплайн `Results` из LibreYOLO во время выполнения не участвует:
HEF — самостоятельный артефакт, а предобработка и постобработка остаются на
стороне приложения.

## Ограничения

Может ли модель работать на Hailo-8 или Hailo-8L — свойство её архитектуры, а не
названия, поэтому правило ниже применимо и к семействам, добавленным после того,
как эта страница была написана.

Модель не скомпилируется, если содержит что-либо из этого:

- Attention в любом виде: self, cross, deformable или оконный. Это исключает все
  детекторы в стиле DETR, все детекторы с открытым словарём и с текстовым
  условием, все бэкбоны ViT и любые языковые и визуально-языковые башни. В
  каталоге моделей самой Hailo есть несколько вручную подогнанных HEF с
  трансформерами; это штучная работа вендора, и она не доказывает, что
  скомпилируется произвольный граф с attention.
- Динамические формы или поток управления, зависящий от данных. DFC компилирует
  одну фиксированную форму входа и статический граф, поэтому переменное число
  запросов, текстовые промпты, динамический top-k, `NonZero`, `Gather` или `TopK`
  с динамическими индексами и `grid_sample` отпадают.
- Архитектура, построенная в основном на LayerNorm или GELU. BatchNorm аккуратно
  сворачивается в свёртки; поддержка LayerNorm слабая, а GELU — не нативная
  активация, поэтому стек в стиле ConvNeXt подходит плохо, хотя формально он
  свёрточный.
- Работа image-to-image в нативном разрешении. Модели восстановления работают на
  полном входном разрешении и выходят за практические бюджеты SRAM у Hailo.

Семейство становится кандидатом, когда оно состоит только из свёрток, использует
BatchNorm с ReLU или SiLU и имеет фиксированный размер входа. В этой библиотеке
под это подпадают одностадийные CNN-детекторы, где основные цели — YOLOX и YOLO9;
другие свёрточные детекторы, такие как PicoDet, YOLO-NAS и RTMDet, с
декодированием на стороне приложения; CNN-классификаторы ResNet, MobileNetV4-conv
и EfficientNetV2, из которых лучше всего поддержан ResNet, потому что для него в
Hailo Model Zoo есть готовые рецепты; и небольшие свёрточные головы под
конкретные задачи, такие как детекция точек FOMO и оценка направления взгляда
L2CS на бэкбоне ResNet, которые в принципе компилируются, но рецепта под Hailo для
них нет.

Одна оговорка про статус, из-за которой ничто на этой странице не подаётся как
поддерживаемое: ни одно семейство LibreYOLO ещё не проведено через DFC от начала
до конца до работающего HEF. Правила выше предсказывают компилируемость по
архитектуре. Поведение парсера, квантизация и точность остаются непроверенными,
пока HEF не скомпилирован и не измерен, поэтому считайте, что каждому кандидату
нужны собственные зафиксированные доказательства: скомпилированный HEF из точного
чекпойнта с записанными версиями DFC, Model Zoo и HailoRT, документированная
калибровка и сравнение точности на устройстве с базовой моделью в FP32, а не
число пропускной способности.

Если модель не проходит, альтернативы — среды выполнения с зафиксированным
паритетом: [ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt) и
[OpenVINO](/docs/export/openvino).
