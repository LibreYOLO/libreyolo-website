---
title: Triton Inference Server
seo_title: Обслуговування моделі LibreYOLO за допомогою NVIDIA Triton
description: >-
  Обслуговування експорту ONNX із LibreYOLO через NVIDIA Triton: структура
  репозиторію моделей, створений config.pbtxt та передбачення за HTTP-адресою
  моделі.
lead: >-
  Triton Inference Server розміщує репозиторій моделей і відповідає на запити
  інференсу через HTTP. LibreYOLO експортує граф ONNX, створює config.pbtxt із
  метаданими експорту в одному параметрі Triton і обробляє URL-адресу моделі як
  шлях до моделі, яку можна завантажити.
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt
  - tritonclient http
  - репозиторій моделей triton
  - віддалений інференс yolo
last_verified: 1.5.0
meta:
  - label: Виклик
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Допоміжна функція
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: Додатково
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Протокол
    value: >-
      Лише інференс V2 через HTTP та HTTPS. Без gRPC, автентифікації, спільної
      пам'яті, завантаження й вивантаження моделей.
  - label: Тайм-аути
    value: Типовий тайм-аут з'єднання та мережі становить 30 секунд
verification: >-
  Перевірено за файлами libreyolo/backends/triton.py,
  libreyolo/models/__init__.py, docs/triton.md та pyproject.toml у гілці dev.
  Команди контейнера взято із закріплених команд у docs/triton.md.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Експортувати у структуру репозиторію
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: Створити config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Отримана структура
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Запустити сервер
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Дочекатися готовності
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: Зупинити сервер
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Передбачення через обслуговувану модель
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Порівняти з локальною моделлю
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: Закріпити версію або змінити тайм-аут
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Другий сегмент шляху вибирає версію моделі. Без нього вибір
        # визначається налаштованою в Triton політикою версій.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Типовий тайм-аут з'єднання та мережі становить 30 секунд.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## Встановлення

<code-tabs name="install" />

Додатковий пакет `triton` встановлює `tritonclient[http]`. Додаткові пакети gRPC
і спільної пам'яті свідомо виключено: ця інтеграція підтримує лише інференс V2
через HTTP та HTTPS. Пакет `onnx` потрібен тому, що і обслуговуваний артефакт,
і генератор конфігурації працюють із графом ONNX.

## Побудова репозиторію моделей

Експортуйте з динамічною віссю батча у структуру каталогів, якої очікує Triton.

<code-tabs name="repo" />

Triton не зберігає користувацькі метадані ONNX у відповіді з конфігурацією моделі,
тому повні метадані експорту потрібно передати іншим способом.
`create_triton_config` кодує їх як один параметр із рядком JSON під назвою
`libreyolo_metadata` у `config.pbtxt`, створює оголошення входів і виходів у
порядку графа, обробляє екранування JSON та закріплює модель за `KIND_CPU`.

Допоміжна функція виконує валідацію перед записом. Вона вимагає рівно один вхід
графа ONNX, принаймні один вихід, визначувані форми тензорів і метадані, у яких
відображення `names` визначає кожен індекс класу від 0 до `nc - 1`. Модель, що
не проходить будь-яку з цих перевірок, відхиляється під час налаштування, а не
під час першого запиту.

`max_batch_size: 8` відповідає динамічному експорту й дає серверу змогу обробляти
до восьми зображень у батчі за один запит. Для фіксованого графа ONNX із батчем 1
використовуйте `max_batch_size=0`; тоді LibreYOLO надсилає зображення послідовно.

## Запуск сервера

<code-tabs name="serve" />

Команди закріплюють Triton Server 26.04 і свідомо не містять параметрів GPU для
Docker, оскільки `KIND_CPU` у створеній конфігурації в будь-якому разі запобігає
розміщенню на GPU.

## Запуск артефакту

URL-адреса моделі Triton є шляхом до моделі. `LibreYOLO()` перевіряє схему `http`
або `https` до будь-якої обробки локального шляху й повертає бекенд, що взаємодіє
із сервером. Тому місце виклику і повернений об'єкт `Results` ідентичні локальній
контрольній точці.

<code-tabs name="run" />

URL-адреса має форму `http(s)://host:port/model` із необов'язковим сегментом
версії. Порт потрібно зазначити явно. Вбудовані облікові дані, рядок запиту та
фрагмент відхиляються, як і шлях, що містить понад два сегменти.

Параметр `device` приймається та ігнорується із записом у журнал, оскільки рішення
про розміщення ухвалює сервер.

## Обмеження

Якщо контракт не виконано, бекенд завершується з прямою помилкою, а не повертає
погіршений результат. Причинами можуть бути відсутність метаданих LibreYOLO у
конфігурації моделі, кілька входів моделі, невідповідність між налаштованими
виходами й метаданими моделі, непідтримуваний тип вхідних даних або неготовий
сервер чи модель.

У цій версії поза контрактом залишаються gRPC, автентифікація, спільна пам'ять,
а також завантаження чи вивантаження моделей через API.

Можна обслуговувати будь-який формат, який підтримує сам Triton, але параметр
метаданих і створена конфігурація тут мають структуру ONNX. Тому шлях LibreYOLO
веде від [ONNX](/docs/export/onnx) до репозиторію. Для повного відеопайплайна,
а не сервера запитів і відповідей, дивіться сторінку
[DeepStream](/docs/export/deepstream).

