---
title: libreyolo predict
seo_title: "справочник по команде libreyolo predict"
description: "Запуск инференса из командной строки: каждый аргумент, его значение по умолчанию, прочитанное из определения CLI, и флаги, которые меняют то, что попадает в stdout."
lead: "Прогоняет загруженную модель по одному источнику и печатает предсказания. Источником может быть изображение, каталог, видео, URL или живой поток; моделью — чекпойнт или экспортированный артефакт."
keywords: [libreyolo predict cli, инференс yolo из командной строки, команда libreyolo predict, аргументы libreyolo predict, yolo json вывод в stdout]
last_verified: "1.5.0"
meta:
  - label: Команда
    value: libreyolo predict
    mono: true
  - label: Обязательный
    value: source
    mono: true
  - label: Вывод
    value: "Предсказания в stdout. При save=true — аннотированные файлы в runs/detect/predict"
snippets:
  examples:
    - label: Базовый вызов
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Сохранение аннотированных изображений
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Фильтр классов, JSON в stdout
      language: bash
      code: |
        # класс 0 — person в списке классов COCO, который идёт вместе с чекпойнтом.
        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50 \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
---

## Синопсис

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Аргументы — это пары `key=value`. Та же команда принимает и POSIX-форму, поэтому
`conf=0.4` и `--conf 0.4` взаимозаменяемы, а булев аргумент, записанный как
`save=true`, превращается в `--save`. Имена с подчёркиванием принимают оба
написания: `max_det=50` и `--max-det 50` попадают в одну и ту же опцию.

`libreyolo detect predict ...` тоже принимается и ведёт себя точно так же; слово
задачи отбрасывается перед разбором.

## Аргументы

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `source` | | Путь к изображению, каталог или URL. Обязателен |
| `model` | `yolox-s` | Имя или путь модели |
| `conf` | `0.25` | Порог уверенности |
| `iou` | `0.45` | Порог IoU для NMS |
| `imgsz` | | Размер входного изображения: `640` (квадрат) или `480x640` (HxW). Если не задан — собственный входной размер модели |
| `classes` | | Фильтр по ID классов, например `[0,2,5]`. Одиночное целое тоже принимается |
| `max_det` | `300` | Максимум детекций на изображение |
| `half` | `false` | Инференс в FP16 (только CUDA, нужна поддержка со стороны модели) |
| `save` | `false` | Сохранять аннотированные изображения |
| `batch` | `1` | Изображений за один прямой проход для источников-каталогов. Значение выше 1 запускает настоящий батчевый инференс на моделях, которые его поддерживают |
| `stream` | `false` | Выдавать результаты по мере готовности. Включается автоматически для веб-камер и живых потоков |
| `stream_buffer` | `false` | Буферизовать каждый живой кадр вместо хранения только самого свежего |
| `vid_stride` | `1` | Обрабатывать каждый N-й кадр видео или живого потока |
| `show` | `false` | Показывать результаты для видео и живых потоков; `q` останавливает |
| `tiling` | `false` | Потайловый инференс для больших изображений |
| `overlap_ratio` | `0.2` | Доля перекрытия тайлов |
| `output_path` | | Явный путь вывода. Иначе `project/name`, когда `save=true` |
| `color_format` | `auto` | Цвет входа: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Формат вывода: `jpg`, `png`, `webp` |
| `device` | `auto` | Устройство: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Модель детектора лиц (путь или имя из CLI). Обязательна для моделей оценки взгляда |
| `gallery` | | Галерея лиц `.npz` из `libreyolo enroll`, по которой опознаются лица. Только для моделей эмбеддингов лиц |
| `gallery_threshold` | `0.4` | Косинусный порог для совпадения личности по галерее |
| `project` | `runs/detect` | Корневой каталог вывода |
| `name` | `predict` | Имя эксперимента |
| `exist_ok` | `false` | Переиспользовать существующий каталог вывода |
| `json` | `false` | Вывод JSON в stdout |
| `quiet` | `false` | Подавить stderr |
| `verbose` | `false` | Подробный вывод в stderr |
| `help_json` | `false` | Вывести схему команды в JSON и выйти |

## Примеры

<code-tabs name="examples" />

## Примечания

Экспортированный артефакт загружается так же, как чекпойнт, поэтому
`model=weights/LibreYOLO9s.onnx` и `model=weights/LibreYOLO9s.engine` —
допустимые значения для `model`. Три опции на таких средах выполнения не
игнорируются, а отклоняются: `tiling`, `overlap_ratio` и `output_file_format`
завершаются с `config_unsupported`, если бэкенд среды выполнения не может их
выполнить.

С `half` всё наоборот. Экспортированные среды выполнения получают его и работают
в FP16; нативный инференс PyTorch пишет в лог, что параметр проигнорирован, и
продолжает в FP32.

Модели оценки взгляда двухэтапные и не имеют собственного детектора, поэтому для
них обязателен `face_detector`. `gallery` применим только к моделям с задачей
`embed`; передача его чему-то другому завершается с `config_unsupported`.

В stdout идут результаты и ничего больше; прогресс, предупреждения и ошибки
уходят в stderr. `json=true` печатает один JSON-объект на запуск или по одному
на кадр при стриминге, и в каждом есть `schema_version`. `quiet=true` заглушает
stderr. Вместе они дают машинному читателю чистый поток stdout.

Код возврата — `0` при успехе, `2` при ошибке использования или конфигурации,
`3` если источник не найден, `4` если модель не удалось загрузить, и `1` для
остальных сбоев во время выполнения.

`help_json=true` печатает параметры команды, их типы, значения по умолчанию и
флаги в виде JSON, ничего не запуская, — это надёжный способ получить эту
таблицу из установленной версии.

См. также: [`libreyolo val`](/docs/cli/val) — измеренные метрики на датасете,
[`libreyolo export`](/docs/cli/export) — создание артефактов сред выполнения,
упомянутых выше.
