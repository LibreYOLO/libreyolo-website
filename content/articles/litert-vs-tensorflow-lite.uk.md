---
title: "LiteRT чи TensorFlow Lite: перейменування, а не новий формат"
description: "У вересні 2024 року Google перейменувала TensorFlow Lite на LiteRT. Те саме середовище виконання, ті самі файли .tflite, нічого не ламається. Пояснюємо причину перейменування, що саме змінилося та як експортувати модель LibreYOLO у цей формат."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "TensorFlow Lite застарів?"
    a: "Відмовляються від назви, а не від технології. Середовище виконання продовжує існувати під назвою LiteRT із тими самими API, а наявні моделі TFLite та код і далі працюють. Нові можливості, як-от API CompiledModel та прискорення на NPU, випускають під назвою LiteRT."
  - q: "Чи потрібно повторно експортувати моделі для LiteRT?"
    a: "Ні. Файл .tflite, експортований для TensorFlow Lite, є моделлю LiteRT. Формат файлу не змінився, тож повторно експортувати чи переносити модель не потрібно."
---

**LiteRT є новою назвою TensorFlow Lite. Це не нове середовище виконання і не новий формат файлів. Моделі й далі зберігаються у файлах `.tflite`, і кожен файл `.tflite`, який працював із TensorFlow Lite, так само працює з LiteRT без змін.**

Якщо ви шукали «LiteRT vs TensorFlow Lite», «TensorFlow Lite застарів?» або «що таке LiteRT», відповідь уже є в цьому абзаці. Далі коротко пояснюємо причину перейменування та те, як експортувати модель у цей формат за допомогою LibreYOLO.

## Чому Google перейменувала його

TensorFlow Lite з'явився у 2017 році як спосіб запускати моделі TensorFlow на телефонах і вбудованих платах. З роками його можливості значно розширилися: Google додала шляхи конвертації з PyTorch, JAX і Keras, тож до 2024 року значна частина моделей, які на ньому працювали, взагалі не взаємодіяла з TensorFlow.

На той момент назва вже вводила в оману. Користувачі PyTorch оминали цей інструмент, бо назва натякала, що він призначений лише для TensorFlow. Тож у вересні 2024 року Google [перейменувала його на LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), скорочено від «Lite Runtime». Ось і вся історія. Та сама команда, той самий код, назва, що не прив'язана до фреймворку.

## Що саме змінилося

Зміни мінімальні, і жодна з них нічого не ламає:

* Код переїхав на нову адресу: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* Документацію перенесли на [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* Python-пакет для запуску моделей тепер називається `ai-edge-litert`. Старий пакет `tflite-runtime` застарів; новий має той самий клас `Interpreter`.
* Після перейменування нові функції виходять під назвою LiteRT: простіший API `CompiledModel` і прискорення на GPU/NPU, які Google оголосила [готовими до промислового використання в січні 2026 року](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

Класичний API Interpreter також продовжує працювати без змін. Є одне справді нове розширення, `.litertlm`, але це пакувальний формат лише для локальних LLM; для моделей комп'ютерного зору він не має значення. Тож якщо в одній документації написано «TFLite», а в іншій «LiteRT», ідеться про той самий файл.

## Експорт моделі LibreYOLO у LiteRT

У v1.3.0 LibreYOLO додала шлях експорту TFLite/LiteRT. Потрібен Python 3.12+ і один додатковий пакет:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

Або з CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

Наразі перевірено шляхи для виявлення об'єктів YOLO9, а також виявлення об'єктів, сегментації та оцінювання пози RF-DETR. Усі вони досі мають статус експериментальних. Повна матриця підтримки наведена в [документації](/docs/v1.3.0).

Щоб запустити експортований файл, використовуйте на цільовому пристрої пакет Google `ai-edge-litert`:

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

Зверніть увагу: вхідні дані мають формат NHWC (канали останні). Під час конвертації з ONNX розташування осей змінюється, що є нормальним для цього формату.

## Що ми змінили в LibreYOLO

Майже нічого, бо й змінювати майже нічого не довелося. Тепер у документації формат позначено як «TFLite (LiteRT)», щоб його можна було знайти за обома назвами, а API й далі використовує `format="tflite"` як назву формату.

## Поширені запитання

**TensorFlow Lite застарів?**
Відмовляються від назви, а не від технології. Середовище виконання продовжує існувати під назвою LiteRT із тими самими API, а наявні моделі TFLite та код і далі працюють. Нові можливості, як-от API CompiledModel та прискорення на NPU, випускають під назвою LiteRT.

**Чи потрібно повторно експортувати моделі для LiteRT?**
Ні. Файл `.tflite`, експортований для TensorFlow Lite, є моделлю LiteRT. Формат файлу не змінився, тож повторно експортувати чи переносити модель не потрібно.
