---
title: "LiteRT vs TensorFlow Lite: что изменилось после переименования?"
description: "В сентябре 2024 года Google переименовала TensorFlow Lite в LiteRT. Среда выполнения и файлы .tflite остались прежними, ничего не сломалось. Разбираем причины переименования, реальные изменения и экспорт модели LibreYOLO в LiteRT."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "TensorFlow Lite устарел?"
    a: "Уходит прежнее название, а не технология. Среда выполнения продолжает развиваться под именем LiteRT с теми же API, а существующие модели TFLite и код по-прежнему работают. Новые функции, например API CompiledModel и ускорение на NPU, выпускаются под именем LiteRT."
  - q: "Нужно ли заново экспортировать модели для LiteRT?"
    a: "Нет. Файл .tflite, экспортированный для TensorFlow Lite, — это модель LiteRT. Файл не изменился, поэтому повторно экспортировать модель или выполнять миграцию не нужно."
---

**LiteRT — это TensorFlow Lite под новым именем. Это не новая среда выполнения и не новый формат файлов. Модели по-прежнему хранятся в файлах `.tflite`, и каждый `.tflite`-файл, который работал с TensorFlow Lite, без изменений работает и с LiteRT.**

Если вы искали «LiteRT vs TensorFlow Lite», «устарел ли TensorFlow Lite» или «что такое LiteRT», ответ — в этом абзаце. Ниже кратко объясняется, зачем понадобилось переименование и как экспортировать модель в этот формат с помощью LibreYOLO.

## Почему Google переименовала TensorFlow Lite

TensorFlow Lite появился в 2017 году как способ запускать модели TensorFlow на телефонах и встраиваемых платах. Со временем область его применения значительно расширилась: Google добавила пути конвертации моделей из PyTorch, JAX и Keras. К 2024 году многие модели, которые на нём запускались, вообще не использовали TensorFlow.

В этот момент название стало вводить в заблуждение. Пользователи PyTorch обходили инструмент стороной, потому что он звучал как средство только для TensorFlow. Поэтому в сентябре 2024 года Google [переименовала его в LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), сокращение от «Lite Runtime». Вот и вся история. Команда и код остались прежними, изменилось только название фреймворка, не привязанное к конкретному фреймворку машинного обучения.

## Что на самом деле изменилось

Изменений мало, и ни одно из них ничего не ломает:

* Код переехал в новый репозиторий: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* Документация переехала на [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* Теперь для запуска моделей используется пакет Python `ai-edge-litert`. Старый пакет `tflite-runtime` устарел; новый пакет содержит тот же класс `Interpreter`.
* После переименования новые функции выпускаются под именем LiteRT: более простой API `CompiledModel` и ускорение на GPU/NPU, которое Google объявила [готовым к промышленному использованию в январе 2026 года](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

Классический API Interpreter также продолжает работать без изменений. Появилось одно действительно новое расширение — `.litertlm`, но это формат упаковки только для LLM на устройствах. Для моделей компьютерного зрения оно не имеет значения. Поэтому когда в одной документации написано «TFLite», а в другой — «LiteRT», речь идёт об одном и том же файле.

## Экспорт модели LibreYOLO в LiteRT

В версии v1.3.0 в LibreYOLO появился путь экспорта TFLite/LiteRT. Нужны Python 3.12+ и один дополнительный пакет:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

Экспорт также можно выполнить из CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

Сейчас проверены детекция YOLO9, а также детекция, сегментация и оценка позы RF-DETR. Все эти варианты по-прежнему имеют статус экспериментальных. Полная таблица поддержки есть в [документации](/docs/v1.3.0).

Чтобы запустить экспортированный файл, используйте на целевом устройстве пакет Google `ai-edge-litert`:

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, your preprocessed image here
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

Обратите внимание: входной формат — NHWC (каналы в конце). При конвертации из ONNX порядок осей меняется, что нормально для этого формата.

## Что изменилось в LibreYOLO

Почти ничего, потому что менять почти ничего не пришлось. Теперь в документации формат называется «TFLite (LiteRT)», чтобы его можно было найти по обоим названиям, а API по-прежнему использует `format="tflite"` как имя формата.

## FAQ

**TensorFlow Lite устарел?**
Уходит прежнее название, а не технология. Среда выполнения продолжает развиваться под именем LiteRT с теми же API, а существующие модели TFLite и код по-прежнему работают. Новые функции, например API CompiledModel и ускорение на NPU, выпускаются под именем LiteRT.

**Нужно ли заново экспортировать модели для LiteRT?**
Нет. Файл `.tflite`, экспортированный для TensorFlow Lite, — это модель LiteRT. Файл не изменился, поэтому повторно экспортировать модель или выполнять миграцию не нужно.
