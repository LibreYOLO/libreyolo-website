---
title: libreyolo predict
seo_title: довідник команди libreyolo predict
description: >-
  Запуск інференсу з командного рядка: кожен аргумент, його типове значення,
  прочитане з визначення CLI, і прапорці, які змінюють те, що потрапляє на
  stdout.
lead: >-
  Запускає завантажену модель на одному джерелі та друкує передбачення. Джерелом
  може бути зображення, каталог, відео, URL або живий потік; моделлю може бути
  контрольна точка або експортований артефакт.
keywords:
  - libreyolo predict cli
  - інференс yolo з командного рядка
  - детекція об'єктів yolo cli
  - аргументи команди libreyolo predict
  - yolo json вивід результатів
last_verified: 1.5.0
meta:
  - label: Команда
    value: libreyolo predict
    mono: true
  - label: Обов'язкове
    value: source
    mono: true
  - label: Вивід
    value: Передбачення на stdout. З save=true анотовані файли в runs/detect/predict
snippets:
  examples:
    - label: Базове використання
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Зберегти анотовані зображення
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Фільтр класів, JSON на stdout'
      language: bash
      code: >
        # клас 0 відповідає person у списку класів COCO, що постачається з
        контрольною точкою.

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## Синтаксис

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Аргументи задаються парами `key=value`. Та сама команда приймає й POSIX-форму,
тому `conf=0.4` і `--conf 0.4` взаємозамінні, а булеве значення, записане як
`save=true`, перетворюється на `--save`. Імена з підкресленням приймають обидва
написання: `max_det=50` і `--max-det 50` ведуть до того самого параметра.

`libreyolo detect predict ...` теж приймається і поводиться так само; слово
задачі відкидається перед розбором.

## Аргументи

| Аргумент | Типове значення | Опис |
|---|---|---|
| `source` | | Шлях до зображення, каталог або URL. Обов'язковий |
| `model` | `yolox-s` | Назва або шлях до моделі |
| `conf` | `0.25` | Поріг впевненості |
| `iou` | `0.45` | Поріг IoU для NMS |
| `imgsz` | | Розмір вхідного зображення: `640` (квадрат) або `480x640` (HxW). Якщо не задано, береться власний розмір входу моделі |
| `classes` | | Фільтр за ID класів, напр. `[0,2,5]`. Приймається й одне ціле число |
| `max_det` | `300` | Максимум виявлень на зображення |
| `half` | `false` | Інференс у FP16 (лише CUDA, потрібна підтримка з боку моделі) |
| `save` | `false` | Зберігати анотовані зображення |
| `batch` | `1` | Кількість зображень на один прямий прохід для джерел-каталогів. Значення понад 1 вмикає справжній батчевий інференс на моделях, які його підтримують |
| `stream` | `false` | Видавати результати поступово. Вмикається автоматично для вебкамер і живих потоків |
| `stream_buffer` | `false` | Буферизувати кожен живий кадр замість зберігання лише найновішого |
| `vid_stride` | `1` | Обробляти кожен N-й кадр відео чи живого потоку |
| `show` | `false` | Показувати відео та живі результати; `q` зупиняє |
| `tiling` | `false` | Тайловий інференс для великих зображень |
| `overlap_ratio` | `0.2` | Коефіцієнт перекриття тайлів |
| `output_path` | | Явний шлях виводу. Інакше `project/name`, коли `save=true` |
| `color_format` | `auto` | Колір входу: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Формат виводу: `jpg`, `png`, `webp` |
| `device` | `auto` | Пристрій: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Модель детектора облич (шлях або назва в CLI). Обов'язкова для моделей оцінювання погляду |
| `gallery` | | Галерея облич `.npz` з `libreyolo enroll`, за якою ідентифікуються обличчя. Лише для моделей ембедингів облич |
| `gallery_threshold` | `0.4` | Косинусний поріг для збігу особи з галереї |
| `project` | `runs/detect` | Корінь каталогу виводу |
| `name` | `predict` | Назва експерименту |
| `exist_ok` | `false` | Повторно використовувати наявний каталог виводу |
| `json` | `false` | Вивід JSON на stdout |
| `quiet` | `false` | Придушити stderr |
| `verbose` | `false` | Докладний вивід у stderr |
| `help_json` | `false` | Вивести схему команди як JSON і вийти |

## Приклади

<code-tabs name="examples" />

## Примітки

Експортований артефакт завантажується так само, як контрольна точка, тому
`model=weights/LibreYOLO9s.onnx` і `model=weights/LibreYOLO9s.engine` є
коректними значеннями для `model`. Три параметри в таких середовищах виконання
не ігноруються, а відхиляються: `tiling`, `overlap_ratio` і
`output_file_format` завершують роботу з `config_unsupported`, якщо бекенд
середовища виконання не може їх виконати.

`half` працює навпаки. Експортовані середовища виконання отримують його і
працюють у FP16; нативний інференс PyTorch записує в лог, що параметр
проігноровано, і продовжує у FP32.

Моделі оцінювання погляду двостадійні і не мають власного детектора, тому для
них потрібен `face_detector`. `gallery` стосується лише моделей із задачею
`embed`; передавання його будь-чому іншому завершується з
`config_unsupported`.

stdout несе лише результати й нічого більше; прогрес, попередження та помилки
йдуть у stderr. `json=true` друкує один об'єкт JSON на виклик або по одному на
кадр у режимі потоку, кожен із полем `schema_version`. `quiet=true` вимикає
вивід у stderr. Разом вони дають машинному читачеві чистий потік stdout.

Код виходу: `0` в разі успіху, `2` для помилки використання чи конфігурації,
`3` коли джерело не знайдено, `4` коли не вдалося завантажити модель, і `1` для
інших збоїв під час виконання.

`help_json=true` друкує параметри команди, їхні типи, типові значення та
прапорці у форматі JSON, нічого не запускаючи, і це надійний спосіб прочитати
цю таблицю зі встановленої версії.

Пов'язане: [`libreyolo val`](/docs/cli/val) для вимірювання метрик на датасеті,
[`libreyolo export`](/docs/cli/export) для створення згаданих вище артефактів
середовища виконання.
