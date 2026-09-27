---
title: Triton Inference Server
seo_title: Развёртывание модели LibreYOLO на NVIDIA Triton
description: >-
  Развёртывание ONNX-экспорта LibreYOLO через NVIDIA Triton: структура
  репозитория моделей, генерируемый config.pbtxt и предсказание по HTTP-адресу
  модели.
lead: >-
  Triton Inference Server держит репозиторий моделей и отвечает на запросы
  инференса по HTTP. LibreYOLO экспортирует ONNX-граф, генерирует config.pbtxt,
  который несёт метаданные экспорта в одном параметре Triton, и трактует URL
  модели как путь к загружаемой модели.
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt
  - tritonclient http
  - репозиторий моделей triton
  - удалённый инференс yolo
last_verified: 1.5.0
meta:
  - label: Вызов
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Вспомогательная функция
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: Extra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Протокол
    value: >-
      Только инференс по HTTP и HTTPS V2. Без gRPC, аутентификации, разделяемой
      памяти, загрузки и выгрузки моделей.
  - label: Таймауты
    value: Таймауты подключения и сети по умолчанию — 30 секунд
verification: >-
  Прочитано из libreyolo/backends/triton.py, libreyolo/models/__init__.py,
  docs/triton.md и pyproject.toml в ветке dev. Команды для контейнера —
  зафиксированные из docs/triton.md.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Экспорт в структуру репозитория
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
    - label: Генерация config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Итоговая структура
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Запуск сервера
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Ожидание готовности
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: Остановка
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Предсказание через модель на сервере
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Сравнение с локальной моделью
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: Фиксация версии или изменение таймаута
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Второй сегмент пути выбирает версию модели. Без него
        # выбор делает настроенная в Triton политика версий.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Таймауты подключения и сети по умолчанию — 30 секунд.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## Установка

<code-tabs name="install" />

Extra `triton` ставит `tritonclient[http]`. Extra для gRPC и разделяемой памяти
исключены намеренно: эта интеграция — только инференс по HTTP и HTTPS V2.
`onnx` нужен, потому что и раздаваемый артефакт, и генератор конфига работают
с ONNX-графом.

## Сборка репозитория моделей

Экспорт с динамической осью батча, в структуру каталогов, которую ожидает Triton.

<code-tabs name="repo" />

Triton не сохраняет пользовательские метаданные ONNX в ответе с конфигурацией
модели, поэтому полные метаданные экспорта приходится передавать иначе.
`create_triton_config` кодирует их как один строковый JSON-параметр с именем
`libreyolo_metadata` в `config.pbtxt`, выписывает объявления входов и выходов в
порядке графа, берёт на себя экранирование JSON и фиксирует модель на `KIND_CPU`.

Функция проверяет всё до записи. Ей нужны ровно один вход ONNX-графа, хотя бы
один выход, разрешимые формы тензоров и метаданные, у которых отображение
`names` задаёт каждый индекс класса от 0 до `nc - 1`. Модель, не прошедшую любую
из этих проверок, отклоняют на этапе конфигурации, а не на первом запросе.

`max_batch_size: 8` соответствует динамическому экспорту и позволяет серверу
собирать до восьми изображений в один запрос. Для ONNX-графа с фиксированным
батчем 1 используйте `max_batch_size=0`; тогда LibreYOLO отправляет изображения
последовательно.

## Запуск сервера

<code-tabs name="serve" />

Команды фиксируют Triton Server 26.04 и намеренно опускают флаги GPU для Docker:
`KIND_CPU` в сгенерированном конфиге всё равно не даёт разместить модель на GPU.

## Запуск артефакта

URL модели в Triton — это путь к модели. `LibreYOLO()` проверяет схему `http`
или `https` до любой обработки локальных путей и возвращает бэкенд, который
общается с сервером, поэтому место вызова не отличается от локального чекпойнта,
как и возвращаемый объект `Results`.

<code-tabs name="run" />

Форма URL — `http(s)://host:port/model` с необязательным сегментом версии. Порт
нужно указывать явно. Встроенные учётные данные, строка запроса и фрагмент
отклоняются, как и путь длиннее двух сегментов.

`device` принимается и игнорируется с записью в лог, потому что размещение —
решение сервера.

## Ограничения

Если контракт не соблюдён, бэкенд падает с прямой ошибкой, а не выдаёт
ухудшенный результат: нет метаданных LibreYOLO в конфигурации модели, больше
одного входа модели, расхождение между настроенными выходами и метаданными
модели, неподдерживаемый тип данных на входе или неготовность сервера либо
модели.

Вне контракта в этой версии: gRPC, аутентификация, разделяемая память, а также
загрузка и выгрузка моделей через API.

Раздавать можно любой формат, который поддерживает сам Triton, но параметр с
метаданными и сгенерированный конфиг здесь заточены под ONNX, поэтому путь
LibreYOLO — [ONNX](/docs/export/onnx) в репозиторий. Если нужен полноценный
видеопайплайн, а не сервер «запрос — ответ», смотрите
[DeepStream](/docs/export/deepstream).
