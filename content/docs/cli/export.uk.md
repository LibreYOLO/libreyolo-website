---
title: libreyolo export
seo_title: довідник команди libreyolo export
description: >-
  Експорт контрольної точки у формат для розгортання: кожен аргумент із типовим
  значенням, куди потрапляє артефакт і які комбінації команда відхиляє.
lead: >-
  Перетворює одну контрольну точку на один формат для розгортання і записує
  артефакт у weights/. Саме формат вирішує, які з наведених нижче аргументів
  застосовуються.
keywords:
  - libreyolo export cli
  - експорт yolo в onnx
  - команда libreyolo export
  - експорт yolo в tensorrt
  - аргументи libreyolo export
last_verified: 1.5.0
meta:
  - label: Команда
    value: libreyolo export
    mono: true
  - label: Обов'язковий
    value: model
    mono: true
  - label: Вихідний файл
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: Базовий приклад
      language: bash
      code: |
        # Записує weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS усередині графа
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Запуск артефакта
      language: bash
      code: >
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640


        # Фабрика орієнтується на суфікс файлу, тому експорт завантажується як
        контрольна точка.

        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## Синтаксис

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Аргументи задаються парами `key=value`, і POSIX-форма теж працює, тому
`format=onnx` і `--format onnx` є тим самим аргументом.

## Аргументи

| Аргумент | Типове значення | Призначення |
|---|---|---|
| `model` | | Ваги моделі `.pt`. Обов'язковий |
| `format` | `onnx` | Формат експорту: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | Цільова платформа RKNN, наразі лише `rk3588`. Відхиляється з будь-яким іншим форматом |
| `imgsz` | | Розмір вхідного зображення: `640` або `480x640` (HxW). `480,640` теж приймається. Якщо не задано, береться власний розмір моделі |
| `batch` | `1` | Розмір батча для експорту |
| `half` | `false` | Точність FP16 |
| `int8` | `false` | Квантування INT8 |
| `dynamic` | `false` | Динамічні форми входу (ONNX) |
| `simplify` | `true` | Спрощення графа ONNX |
| `nms` | `false` | Вбудувати NMS у модель. Лише ONNX і CoreML |
| `conf` | `0.25` | Поріг впевненості для вбудованого NMS |
| `iou` | `0.45` | Поріг IoU для вбудованого NMS |
| `max_det` | `300` | Максимальна кількість виявлень для вбудованого NMS в ONNX |
| `opset` | | Версія opset для ONNX. Якщо не задано, вибирається автоматично |
| `data` | | Калібрувальні дані для INT8 |
| `fraction` | `1.0` | Частка калібрувальних даних, яку використовувати |
| `device` | `auto` | Пристрій для трасування |
| `allow_download_scripts` | `false` | Дозволити вбудований Python у блоках download у YAML датасету |
| `json` | `false` | Вивід JSON у stdout |
| `quiet` | `false` | Приглушити stderr |
| `verbose` | `false` | Докладне журналювання експорту |
| `verify` | `false` | Запустити симулятор RKNN Toolkit2 для ПК і порівняти з ONNX Runtime. Лише RKNN |
| `help_json` | `false` | Вивести схему команди як JSON і вийти |

`engine` є псевдонімом для `tensorrt`, а `litert` псевдонімом для `tflite`.
Обидва зводяться до канонічної назви ще до того, як щось буде записано, тому у
виводі JSON і в рядку журналу завжди зазначено `tensorrt` або `tflite`.

## Приклади

<code-tabs name="examples" />

## Примітки

### Куди потрапляє файл

Команда не приймає шлях виводу. Артефакт записується до `weights/` і отримує
назву з основи вихідної контрольної точки плюс суфікс формату, з доданим
`_fp16` або `_int8`, якщо було запитано одну з цих точностей. `LibreYOLO9s.pt`,
експортований у ONNX з FP16, стає `weights/LibreYOLO9s_fp16.onnx`. Результат
JSON містить визначений `output_path`, розмір файлу в МБ і форму входу як
`[batch, 3, height, width]`.

### Комбінації, які відхиляються

`nms=true` приймається для ONNX і CoreML, а для решти форматів відхиляється з
`nms_unsupported_format`. Для ONNX цей параметр примусово вимикає `dynamic`,
оскільки вбудований граф зафіксовано на батчі 1, і повідомляє про це в stderr.
Для CoreML він враховує `conf` та `iou`, але не `max_det`, тому нетипове
значення `max_det` разом із `format=coreml nms=true` завершує роботу з
`config_unsupported`.

`half=true` разом із `int8=true` не є помилкою. Перемагає INT8, `half`
відкидається, а попередження надходить у stderr.

`name` і `verify` наразі є опціями RKNN. Передавання будь-якої з них з іншим
форматом завершує роботу з `config_unsupported`, а не ігнорується.

### Які формати підтримує сімейство

Підтримка визначається окремо для кожного сімейства і кожного завдання, а не
глобально. `libreyolo formats family=<family> task=<task>` виводить рівень
підтримки кожного формату для цієї комбінації, разом із причиною та будь-яким
пов'язаним обмеженням. Аргументи описано на сторінці
[`libreyolo formats`](/docs/cli/utilities).

Деякі формати потребують додаткового встановлення, а деякі потребують набору
інструментів. Відсутня залежність Python завершує роботу з
`export_dep_missing`; точність, якої формат не може видати, завершує роботу з
`format_precision_unsupported`.

### Запуск експортованого

Експортовані артефакти завантажуються через ту саму фабрику моделей, що й
контрольні точки, за суфіксом файлу, тому
`libreyolo predict model=weights/LibreYOLO9s.onnx` працює без жодного
подальшого перетворення. Виняток становлять три опції передбачення, які
відхиляються на бекендах середовища виконання: `tiling`, `overlap_ratio` і
`output_file_format`.

Дві цілі розгортання мають власні сторінки:
[NVIDIA DeepStream](/docs/export/deepstream) і
[NVIDIA Jetson](/docs/export/jetson).

### Вивід і коди виходу

stdout несе результат; перебіг виконання надходить у stderr. Код виходу: `0` в
разі успіху, `2` при помилці використання або конфігурації, `4` коли модель не
вдається завантажити, `5` для невідомого формату, відсутньої залежності
експорту, непідтримуваної точності або відхиленого запиту на вбудований NMS, і
`1` для інших збоїв під час виконання.

Пов'язане: [`libreyolo quantize`](/docs/cli/quantize), яка залишається в
PyTorch і записує контрольну точку, а не артефакт для розгортання.
