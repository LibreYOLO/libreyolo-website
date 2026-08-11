---
title: NVIDIA DeepStream
seo_title: Запуск моделей YOLO на NVIDIA DeepStream
description: >-
  Экспорт модели LibreYOLO для NVIDIA DeepStream: ONNX-граф плюс сгенерированный
  конфиг nvinfer. Точные команды для сборки парсера и запуска пайплайна.
lead: >-
  NVIDIA DeepStream запускает инференс через свой элемент nvinfer, которому
  нужны ONNX-граф, подходящий файл конфигурации и парсер ограничивающих рамок.
  Флаг deepstream=True при экспорте в ONNX записывает первые два файла и
  связывает их с третьим.
keywords:
  - deepstream yolo
  - nvidia deepstream экспорт
  - nvinfer конфиг
  - кастомный парсер рамок deepstream
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - tensorrt движок
  - yolo на jetson
meta:
  - label: Флаг
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Записывает
    value: 'ONNX-граф, config_infer_primary_<stem>.txt и <stem>_labels.txt'
  - label: Покрытие
    value: 43 комбинации семейства и задачи по девяти задачам
  - label: Парсер
    value: >-
      NvDsInferParseYolo, из проекта DeepStream-Yolo под лицензией MIT, автор —
      Marcos Luciano. Собирается один раз на устройство.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Доступность
    value: Входит в v1.5.0. Влито в dev 2026-08-08 в пул-реквесте 728.
    links:
      - label: пул-реквест 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Проверено в среде выполнения
    value: 'DeepStream 8.0.0 на RTX 5070 Ti, только детекция, 2026-08-08'
verification: >-
  Написано по результатам проверки в среде выполнения от 2026-08-08. Списки
  семейств, ключи конфига и значения по умолчанию прочитаны из
  libreyolo/export/deepstream.py и libreyolo/export/exporter.py на коммите
  5f81e11e, который в тот же день влит в dev в пул-реквесте 728.
snippets:
  install:
    - label: Установка
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Записывает libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # и libreyolo9s_labels.txt в рабочую директорию.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Каждую модель детекции держите в своей директории: любой конфиг

        # детекции называет один и тот же файл кэша движка. См. «Известные
        ловушки».

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Аргументы
      language: python
      code: >
        model.export(
            format="onnx",     # deepstream=True отклоняется для всех остальных форматов
            deepstream=True,
            conf=0.25,         # задаёт pre-cluster-threshold (а также classifier-threshold,
                               # segmentation-threshold в этих задачах)
            iou=0.45,          # задаёт nms-iou-threshold, опускается при cluster-mode=4
            batch=1,           # задаёт batch-size и имя файла кэша движка
            half=False,        # True помечает конфиг как network-mode=2 (сборка fp16)
            int8=False,        # True помечает конфиг как network-mode=1
            dynamic=True,      # динамическая ось батча в ONNX-графе
            imgsz=640,         # задаёт infer-dims=3;H;W
        )


        # deepstream=True и nms=True взаимоисключающие: DeepStream выполняет

        # подавление на стадии кластеризации, поэтому в граф ничего не
        встраивается.
    - label: Сначала загрузка весов D-FINE
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Проверка проброса GPU в первую очередь
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, запуск внутри контейнера DeepStream'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 в этом образе — заглушка, и сборка на ней падает с

        # «fatal error: crt/host_defines.h: No such file or directory». Найдите

        # тулкит, в котором этот заголовок действительно есть; в образе 8.0 это
        cuda-12.5.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # В образе есть libcublas.so.12 и libcublas.so.12.8.4.1, но нет

        # libcublas.so без версии, которая нужна -lcublas, поэтому линковка
        падает с

        # «/usr/bin/ld: cannot find -lcublas». Дайте линковщику нужные ему
        имена.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: Для сегментации экземпляров нужен другой парсер
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: Запуск
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Оба шага в одном контейнере
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Доступность

Экспорт в DeepStream входит в v1.5.0. Он влит в `dev` 2026-08-08 в пул-реквесте
728, поэтому в свежей установке он уже есть и закреплять ветку не нужно.

<code-tabs name="install" />

Если вы клонировали ветку `deepstream-export` до 2026-08-08, замените её. Ветку
перебазировали и запушили с force, и в старой истории нет исправления, без
которого эти экспорты вообще не запускаются на машине с CUDA.

## Что записывает экспорт

`model.export(format="onnx", deepstream=True)` записывает рядом три файла.
Для `libreyolo9s.pt`:

- `libreyolo9s.onnx` — граф детекции, один выходной тензор формы
  `(batch, num_detections, 6)`, каждая строка — `[x1, y1, x2, y2, score, class_id]`
  в пиксельных координатах входа сети.
- `config_infer_primary_libreyolo9s.txt` — конфигурация `nvinfer` с константами
  предобработки семейства, количеством классов, порогами и привязкой к
  парсеру.
- `libreyolo9s_labels.txt` — по одному имени класса на строку.

Файл меток появляется тогда, когда в чекпойнте есть имена классов. У моделей
глубины их нет, поэтому им не достаётся ни файла, ни ключа `labelfile-path`.

LibreYOLO не создаёт `.so`. Тот `.so`, который загружает DeepStream, — это
парсер ограничивающих рамок из `marcoslucianops/DeepStream-Yolo`, собираемый
один раз на устройство, и это один и тот же бинарник, на какой бы детектор
LibreYOLO вы его ни направили. Модель — это ONNX. Классификации и семантической
сегментации парсер не нужен вовсе, потому что `nvinfer` постобрабатывает их сам.

## Экспорт модели

<code-tabs name="export" />

`LibreDFINE._load_weights` бросает `FileNotFoundError`, если файла ещё нет на
диске, и не пытается ничего скачать, поэтому `LibreDFINEs.pt` нужно сначала
загрузить самостоятельно. Этот пробел отслеживается в
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). Веса YOLO9
скачиваются при первом использовании.

Флаг доступен только из Python. У `libreyolo export` в этой ветке нет опции
`deepstream`, а CLI собирает аргументы экспорта из фиксированного списка и не
пробрасывает незнакомые ключи.

## Сборка парсера ограничивающих рамок

Детекции нужна библиотека парсера, сегментации экземпляров — другая, а
остальным задачам не нужна никакая. Две вещи в образе DeepStream 8.0 ломают
документированную команду сборки, и обе относятся к окружению, а не к проблемам
LibreYOLO.

В образе под `/usr/local` лежат `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` и
`cuda-12.9`. Полный тулкит есть только у `cuda-12.5`. Там же есть
`libcublas.so.12` и `libcublas.so.12.8.4.1`, но нет `libcublas.so` без версии, к
которому обращается `-lcublas`. Скрипт ниже обходит и то и другое.

<code-tabs name="parser" />

Затем направьте `custom-lib-path` в сгенерированном конфиге на собранный
`libnvdsinfer_custom_impl_Yolo.so`. Сгенерированное значение — относительный
путь `nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`; он
разрешается, когда `deepstream-app` запускается из клона `DeepStream-Yolo`, и в
остальных случаях его нужно править.

## Запуск пайплайна

Прежде чем тратить время на всё остальное, проверьте, что контейнер видит GPU.
Именно с этой проверки начался прогон валидации — на карте Blackwell под WSL2.

<code-tabs name="gpu" />

В прогоне валидации `deepstream-app` работал с одним файловым источником, без
display sink, с включённым экранным выводом и заданным `gie-kitti-output-dir`,
чтобы детекции каждого кадра попадали на диск в виде текста KITTI. Конфиг с
этими настройками:

<code-tabs name="run" />

`nvinfer` собирает TensorRT-движок из ONNX при первом запуске и кэширует его
рядом с моделью, поэтому первый запуск оплачивает сборку движка, а последующие
загружают кэш.

## Сгенерированный конфиг

Оба конфига ниже записал экспортёр для прогона валидации, вручную их потом не
правили.

| Ключ | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

Два конфига различаются в трёх местах: `maintain-aspect-ratio`, `cluster-mode` и
наличие `nms-iou-threshold` как такового. Конфиг D-FINE опускает этот ключ
полностью — этого и требует `cluster-mode=4`.

Головы, которые выдают не больше одного предсказания на объект, получают
`cluster-mode=4`, поэтому DeepStream не кластеризует их вывод; кластеризация
слила бы действительно разные детекции. Это `rfdetr`, `dfine`, `deim`,
`deimv2`, `ec`, `rtdetr`, `rtdetrv2`, `rtdetrv4` и `yolo9_e2e`. Сеточные и
якорные головы получают `cluster-mode=2` и `nms-iou-threshold`.

В конфигах детекции есть также
`engine-create-func-name=NvDsInferYoloCudaEngineGet`, который передаёт сборку
движка библиотеке парсера. Именно это фиксирует имя файла кэша движка и служит
источником коллизии, описанной в известных ловушках.

## Поддерживаемые задачи и семейства

Экспортируются сорок три комбинации семейства и задачи.
`deepstream_supported_tasks()` и `deepstream_supported_families(task)` в
`libreyolo/export/deepstream.py` возвращают те же списки во время выполнения.

| Задача | `network-type` | Библиотека парсера | Семейства |
|---|---|---|---|
| Детекция | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Классификация | 1 | Не нужна | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Семантическая сегментация | 2 | Не нужна | pidnet, eomt, dinov2, lingbotvision |
| Сегментация экземпляров | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Поза | 100 | Не нужна | yolo9, yolonas, rfdetr, ec |
| Глубина | 100 | Не нужна | depth_anything, zipdepth |
| Восстановление | 100 | Не нужна | nafnet, realesrgan, swinir |
| Маттинг | 100 | Не нужна | birefnet |
| Взгляд | 100 | Не нужна | l2cs |

`network-type=100` означает, что у DeepStream нет постобработчика для этой
задачи. В таких конфигах ставится `output-tensor-meta=1`, родные выходы графа
проходят насквозь без изменений, а приложение разбирает их из метаданных
тензоров. Графы с несколькими выходами здесь работают нормально: каждый
выходной слой доходит до метаданных с теми же именами выходов и динамическими
осями, что и при обычном экспорте в ONNX.

Строки сегментации экземпляров — это строка детекции, за которой идёт маска
этого экземпляра, развёрнутая в плоский вид при `(netH / 4, netW / 4)` — это
разрешение жёстко зашито в seg-парсере — как вероятности для
`segmentation-threshold`.

Классификация и взгляд работают как вторичный инференс. Чтобы поставить
классификатор за детектором, задайте в сгенерированном конфиге `process-mode=2`
и `operate-on-gie-id`. Взгляд — контракт только для головы, один вырез лица на
вход, поэтому перед ним нужен детектор лиц.

Трёх семейств здесь нет намеренно. `segformer` не подключён к общему контракту
семантического экспорта и не экспортируется в ONNX ни в каком виде. У RTMDet-Ins
и YOLO9 экспорт сегментации экземпляров заблокирован в самой LibreYOLO. У
`depth_anything3` нет реализации экспорта.

За двумя строками таблицы стоят пробелы в чекпойнтах. Опубликован только
семантический чекпойнт EoMT размера `l`, а для классификации DINOv2
опубликованного чекпойнта нет вовсе, поэтому для этой комбинации нужны свои
дообученные веса.

## Различия в предобработке

`nvinfer` вычисляет `net-scale-factor * (x - offsets)` по каналам со скалярным
масштабом, который не может выразить стандартное отклонение по каналам. У
семейств, которым оно нужно (`rfdetr`, `ec`, размеры `deimv2` с DINO-бэкбоном,
`rtmdet`, `picodet` и все семейства классификации), нормализация запечена в
экспортированный граф, а сгенерированный конфиг подаёт графу соответствующее
сырое входное пространство.

Расходятся собственные Python-пайплайны LibreYOLO и `nvinfer` по-прежнему в
геометрии:

- Семейства с letterbox (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`,
  `yolo3`, `yolo4`, `yolo7`) нативно дополняют поля серым. `nvinfer` дополняет
  чёрным.
- Детекция `yolonas` нативно масштабирует длинную сторону до 636 внутри своего
  холста 640. `maintain-aspect-ratio` в `nvinfer` использует все 640.
- Классификация нативно масштабирует короткую сторону, а затем делает
  центральный кроп. `nvinfer` растягивает кадр или ROI объекта до входа сети,
  поэтому плотно обрезанные объекты отличаются.
- EoMT нативно проходит семантическую сегментацию тайлами со скользящим окном.
  Экспортированный граф — один растянутый холст, что быстрее и менее точно.
- `pidnet` выдаёт карту классов в 1/8 входного разрешения, а `lingbotvision` —
  в 1/16. DeepStream увеличивает карту классов для отображения.

Проверка паритета ONNX подаёт уже предобработанные тензоры, поэтому она сверяет
выходы графа и не поймает неверный порядок каналов или политику дополнения полей
в конфиге. Прежде чем разворачивать нагрузку с точным паритетом, проверьте на
своих данных.

## Известные ловушки

### Две модели детекции в одной директории загружают движок друг друга

В каждом конфиге детекции есть одна и та же строка:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

Сборщик движка в парсере требует именно это базовое имя, и оно не зависит от
модели. Экспортируйте вторую модель детекции в ту же директорию — и второй
запуск загрузит кэшированный движок первой модели. Ничего не падает, просто
рамки неверные. Дайте каждой модели детекции свою директорию. В прогоне
валидации D-FINE пришлось вынести в отдельную, иначе его вообще не получалось
протестировать.

### Рамка может нести только один класс

Формат строки в `nvinfer` — `[x1, y1, x2, y2, score, class_id]`, один класс на
рамку, поэтому экспорт схлопывает оценки классов до их argmax. Рамка, которую
`predict` показывает под двумя классами, выживает под одним. Измеренный случай:
LibreYOLO выдаёт `vase 0.773` и `bottle 0.383` на одной и той же рамке, а граф
DeepStream оставляет `vase`. Это следует из формата строки у парсера и не
меняется без выхода из этого контракта, так что это ожидаемое поведение, а не
регрессия.

## Что проверено

`deepstream-app` дошёл до EOS с `App run successful` на обоих типах голов
детектора, на встроенном в NVIDIA `sample_1080p_h264.mp4` (1443 кадра), с
включёнными покадровыми дампами KITTI.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Тип головы | сеточная | один к одному |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Кадры с детекциями | 1443 | 1443 |
| Всего детекций | 18031 | 71105 |

Гистограммы классов по всем 1443 кадрам ставят машины на первое место, а людей
на второе для обеих моделей, и для уличной сцены это верно. Четырёхкратный
разрыв в числе детекций — это работа разницы в `cluster-mode`: D-FINE при
`cluster-mode=4` не кластеризует, поэтому выживает каждый запрос выше порога,
включая почти-дубликаты.

Две независимо обученные модели помещают доминирующий объект в одно и то же
место:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Этот прогон подтверждает пять вещей: TensorRT собирает движок из
экспортированного ONNX на sm_120, `nvinfer` принимает каждый ключ
сгенерированного конфига, `NvDsInferParseYolo` правильно читает раскладку
тензора, рамки оказываются в координатах исходного разрешения 1920x1080, а метки
разрешаются по сгенерированному файлу меток.

Окружение, в котором это работало:

| Компонент | Значение |
|---|---|
| ОС хоста | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Драйвер | 591.86 |
| Compute capability | 12.0 (Blackwell, sm_120) |
| Среда выполнения контейнеров | Docker Desktop 29.4.3, бэкенд WSL2 |
| Образ DeepStream | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| Версия DeepStream | 8.0.0 |
| CUDA в контейнере | 12.8.1 |
| Парсер | `marcoslucianops/DeepStream-Yolo` на HEAD |

Помимо прогона пайплайна, `tests/unit/test_deepstream_export.py` покрывает
адаптеры графа и ключи сгенерированного конфига, и его 35 тестов проходят на
этом коммите.

## Что не проверено

Указано, чтобы область выше не читалась шире, чем она есть.

- Jetson и aarch64. Контракт экспорта не зависит от архитектуры, но пайплайн
  запускали только на дискретной GPU с архитектурой x86.
- Сорок одна из 43 комбинаций. Через DeepStream прошли только детекция с
  `yolo9` и детекция с `dfine`. Классификация, семантическая сегментация,
  сегментация экземпляров и задачи с сырыми тензорами покрыты юнит-тестами и
  проверками паритета ONNX, а не прогоном пайплайна.
- FP16 и INT8. Проверялся только `network-mode=0`.
- Несколько потоков и батчинг. Один источник, `batch-size=1`.
- Точность относительно датасета с эталонной разметкой (ground truth). Детекции
  проверяли на смысловую правдоподобность и согласованность между моделями, а не
  оценивали как mAP через DeepStream.
