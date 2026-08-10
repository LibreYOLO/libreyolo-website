---
title: libreyolo export
seo_title: "справочник команды libreyolo export"
description: "Экспорт чекпойнта в формат для развёртывания: каждый аргумент со значением по умолчанию, куда попадает артефакт и какие комбинации команда отклоняет."
lead: "Превращает один чекпойнт в один формат для развёртывания и пишет артефакт в weights/. Формат решает, какие из аргументов ниже применимы."
keywords: [libreyolo export cli, экспорт yolo в onnx, команда libreyolo export, экспорт yolo в tensorrt, аргументы libreyolo export]
last_verified: "1.5.0"
meta:
  - label: Команда
    value: libreyolo export
    mono: true
  - label: Обязательный
    value: model
    mono: true
  - label: Вывод
    value: "weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>"
    mono: true
snippets:
  examples:
    - label: Базовый пример
      language: bash
      code: |
        # Пишет weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS внутри графа
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Запуск артефакта
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640

        # Фабрика ориентируется на суффикс файла, поэтому экспорт загружается как чекпойнт.
        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
---

## Синопсис

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Аргументы задаются парами `key=value`, POSIX-форма тоже работает, поэтому
`format=onnx` и `--format onnx` — один и тот же аргумент.

## Аргументы

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `model` | | Веса модели `.pt`. Обязательный |
| `format` | `onnx` | Формат экспорта: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | Целевая платформа RKNN, сейчас только `rk3588`. С любым другим форматом отклоняется |
| `imgsz` | | Размер входного изображения: `640` или `480x640` (HxW). `480,640` тоже принимается. Если не задан — собственный размер модели |
| `batch` | `1` | Размер батча при экспорте |
| `half` | `false` | Точность FP16 |
| `int8` | `false` | Квантизация INT8 |
| `dynamic` | `false` | Динамические формы входа (ONNX) |
| `simplify` | `true` | Упрощение графа ONNX |
| `nms` | `false` | Встроить NMS в модель. Только ONNX и CoreML |
| `conf` | `0.25` | Порог уверенности для встроенного NMS |
| `iou` | `0.45` | Порог IoU для встроенного NMS |
| `max_det` | `300` | Максимум детекций для встроенного NMS в ONNX |
| `opset` | | Версия opset для ONNX. Если не задана, выбирается автоматически |
| `data` | | Калибровочные данные для INT8 |
| `fraction` | `1.0` | Доля калибровочных данных, которая используется |
| `device` | `auto` | Устройство для трассировки |
| `allow_download_scripts` | `false` | Разрешить встроенный Python в блоках download в YAML датасета |
| `json` | `false` | Вывод JSON в stdout |
| `quiet` | `false` | Подавить stderr |
| `verbose` | `false` | Подробный лог экспорта |
| `verify` | `false` | Запустить PC-симулятор RKNN Toolkit2 и сравнить с ONNX Runtime. Только RKNN |
| `help_json` | `false` | Вывести схему команды в JSON и выйти |

`engine` — псевдоним для `tensorrt`, а `litert` — псевдоним для `tflite`. Оба
приводятся к каноническому имени до того, как что-либо будет записано, поэтому в
JSON-выводе и в строке лога всегда стоит `tensorrt` или `tflite`.

## Примеры

<code-tabs name="examples" />

## Примечания

### Куда попадает файл

Команда не принимает путь для вывода. Артефакт записывается в `weights/`, имя
складывается из основы имени исходного чекпойнта и суффикса формата, а когда
запрошена одна из этих точностей, в него вставляется `_fp16` или `_int8`.
`LibreYOLO9s.pt`, экспортированный в ONNX с FP16, становится
`weights/LibreYOLO9s_fp16.onnx`. В JSON-результате есть итоговый `output_path`,
размер файла в МБ и форма входа в виде `[batch, 3, height, width]`.

### Комбинации, которые отклоняются

`nms=true` принимается для ONNX и CoreML и для любого другого формата
отклоняется с `nms_unsupported_format`. В ONNX он принудительно выключает
`dynamic`, потому что встроенный граф зафиксирован на батче 1, и сообщает об этом
в stderr. В CoreML он учитывает `conf` и `iou`, но не `max_det`, поэтому
`max_det`, отличный от значения по умолчанию, рядом с `format=coreml nms=true`
завершается с `config_unsupported`.

`half=true` вместе с `int8=true` — не ошибка. Побеждает INT8, `half`
отбрасывается, а в stderr уходит предупреждение.

`name` и `verify` сегодня относятся только к RKNN. Если передать любой из них с
другим форматом, он не игнорируется, а команда завершается с
`config_unsupported`.

### Какие форматы поддерживает семейство

Поддержка определяется для каждого семейства и каждой задачи, а не глобально.
`libreyolo formats family=<family> task=<task>` печатает уровень поддержки
каждого формата для этой комбинации, с причиной и приложенным ограничением.
Аргументы описаны в [`libreyolo formats`](/docs/cli/utilities).

Некоторым форматам нужна дополнительная установка, некоторым — тулчейн.
Отсутствующая Python-зависимость завершается с `export_dep_missing`; точность,
которую формат не может выдать, — с `format_precision_unsupported`.

### Запуск того, что вы экспортировали

Экспортированные артефакты загружаются через ту же фабрику моделей, что и
чекпойнты, по суффиксу файла, поэтому `libreyolo predict
model=weights/LibreYOLO9s.onnx` работает без какой-либо дальнейшей конвертации.
Исключение — три опции предсказания, которые на бэкендах сред выполнения
отклоняются: `tiling`, `overlap_ratio` и `output_file_format`.

У двух целей развёртывания есть собственные страницы:
[NVIDIA DeepStream](/docs/export/deepstream) и
[NVIDIA Jetson](/docs/export/jetson).

### Вывод и коды возврата

В stdout уходит результат, прогресс идёт в stderr. Код возврата — `0` при
успехе, `2` при ошибке использования или конфигурации, `4` когда модель не
удаётся загрузить, `5` при неизвестном формате, отсутствующей зависимости для
экспорта, неподдерживаемой точности или отклонённом запросе на встроенный NMS, и
`1` при прочих сбоях во время выполнения.

Смежное: [`libreyolo quantize`](/docs/cli/quantize) — остаётся в PyTorch и пишет
чекпойнт, а не артефакт для развёртывания.
