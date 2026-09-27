---
title: NVIDIA DeepStream
seo_title: Запуск моделей YOLO на NVIDIA DeepStream
description: >-
  Експорт моделі LibreYOLO для NVIDIA DeepStream: граф ONNX плюс згенерований
  конфіг nvinfer. Точні команди для збірки парсера та для пайплайна.
lead: >-
  NVIDIA DeepStream виконує інференс через елемент nvinfer, якому потрібні граф
  ONNX, відповідний файл конфігурації та парсер обмежувальних рамок. Параметр
  deepstream=True під час експорту в ONNX записує перші два і зв'язує їх із
  третім.
keywords:
  - NVIDIA DeepStream
  - DeepStream YOLO
  - nvinfer
  - парсер рамок deepstream
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - tensorrt engine
  - yolo на jetson
meta:
  - label: Прапорець
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Записує
    value: 'Граф ONNX, config_infer_primary_<stem>.txt і <stem>_labels.txt'
  - label: Покриття
    value: 43 комбінації сімейства і задачі в дев'яти задачах
  - label: Парсер
    value: >-
      NvDsInferParseYolo з проєкту DeepStream-Yolo під ліцензією MIT від Marcos
      Luciano. Збирається один раз на пристрій.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Доступність
    value: Входить до v1.5.0. Влито в dev 2026-08-08 у pull request 728.
    links:
      - label: pull request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Перевірене середовище виконання
    value: 'DeepStream 8.0.0 на RTX 5070 Ti, лише виявлення, 2026-08-08'
verification: >-
  Написано за результатами перевірки середовища виконання від 2026-08-08. Списки
  сімейств, ключі конфіга та типові значення прочитано з
  libreyolo/export/deepstream.py і libreyolo/export/exporter.py на коміті
  5f81e11e, який того самого дня влито в dev у pull request 728.
snippets:
  install:
    - label: Встановлення
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Записує libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # та libreyolo9s_labels.txt у робочий каталог.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Тримайте кожну модель виявлення в окремому каталозі: кожен конфіг

        # виявлення називає той самий файл кешу рушія. Див. «Відомі пастки».

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Аргументи
      language: python
      code: |
        model.export(
            format="onnx",     # deepstream=True відхиляється для будь-якого іншого формату
            deepstream=True,
            conf=0.25,         # задає pre-cluster-threshold (а на відповідних задачах ще
                               # classifier-threshold і segmentation-threshold)
            iou=0.45,          # задає nms-iou-threshold, пропускається при cluster-mode=4
            batch=1,           # задає batch-size і назву файлу кешу рушія
            half=False,        # True задає в конфігу network-mode=2 (збірка fp16)
            int8=False,        # True задає в конфігу network-mode=1
            dynamic=True,      # динамічна вісь батча в графі ONNX
            imgsz=640,         # задає infer-dims=3;H;W
        )

        # deepstream=True і nms=True взаємно виключні: DeepStream виконує
        # придушення на етапі кластеризації, тому в граф нічого не вбудовується.
    - label: Спершу завантажте ваги D-FINE
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Насамперед перевірте доступ до GPU з контейнера
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, запускати всередині контейнера DeepStream'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 у цьому образі є заглушкою, і збірка на ній падає з

        # "fatal error: crt/host_defines.h: No such file or directory". Знайдіть

        # інструментарій, який справді містить цей заголовок; в образі 8.0 це
        cuda-12.5.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # Образ містить libcublas.so.12 і libcublas.so.12.8.4.1, але не містить

        # libcublas.so без версії, потрібний для -lcublas, тому лінкування падає
        з

        # "/usr/bin/ld: cannot find -lcublas". Дайте лінкеру ті назви, яких він
        потребує.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: Сегментація екземплярів використовує інший парсер
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
    - label: Обидва кроки в одному контейнері
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Доступність

Експорт у DeepStream входить до v1.5.0. Його влито в `dev` 2026-08-08 у pull
request 728, тож у свіжому встановленні він уже є і закріплювати гілку не треба.

<code-tabs name="install" />

Якщо ви клонували гілку `deepstream-export` до 2026-08-08, замініть її. Ту гілку
перебазовано і надіслано з force-push, а в старішій історії немає виправлення,
без якого ці експорти взагалі не запускаються на машині з CUDA.

## Що записує експорт

`model.export(format="onnx", deepstream=True)` записує поруч три файли. Для
`libreyolo9s.pt`:

- `libreyolo9s.onnx`, граф виявлення: один вихідний тензор форми
  `(batch, num_detections, 6)`, де кожен рядок `[x1, y1, x2, y2, score, class_id]`
  подано в піксельних координатах входу мережі.
- `config_infer_primary_libreyolo9s.txt`, конфігурація `nvinfer` зі сталими
  попередньої обробки для цього сімейства, кількістю класів, порогами та
  прив'язкою до парсера.
- `libreyolo9s_labels.txt`, по одній назві класу на рядок.

Файл міток з'являється щоразу, коли контрольна точка містить назви класів. У
моделей глибини їх немає, тому вони не отримують ні файлу, ні ключа
`labelfile-path`.

LibreYOLO не створює `.so`. Той `.so`, який завантажує DeepStream, є парсером
обмежувальних рамок із `marcoslucianops/DeepStream-Yolo`: його компілюють один
раз на пристрій, і для будь-якого детектора LibreYOLO це той самий бінарний
файл. Моделлю є сам ONNX. Класифікація та семантична сегментація не потребують
парсера взагалі, бо `nvinfer` обробляє їхні виходи самостійно.

## Експорт моделі

<code-tabs name="export" />

`LibreDFINE._load_weights` викидає `FileNotFoundError`, якщо файлу ще немає на
диску, і навіть не намагається його завантажити, тож спершу завантажте
`LibreDFINEs.pt` самостійно. Цю прогалину відстежують у
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). Ваги YOLO9
завантажуються при першому використанні.

Прапорець доступний лише з Python. У `libreyolo export` на цій гілці немає опції
`deepstream`, а CLI збирає аргументи експорту з фіксованого списку і не пропускає
невідомі ключі далі.

## Складання парсера обмежувальних рамок

Виявленню потрібна бібліотека парсера, сегментації екземплярів потрібна інша, а
решті задач не потрібна жодна. Дві речі в образі DeepStream 8.0 ламають
задокументовану команду збірки, і обидві стосуються оточення, а не LibreYOLO.

В образі під `/usr/local` лежать `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` і
`cuda-12.9`. Повний інструментарій має лише `cuda-12.5`. Також в образі є
`libcublas.so.12` і `libcublas.so.12.8.4.1`, але немає `libcublas.so` без версії,
до якого звертається `-lcublas`. Наведений нижче скрипт обходить обидві проблеми.

<code-tabs name="parser" />

Далі вкажіть у `custom-lib-path` згенерованого конфіга шлях до зібраного
`libnvdsinfer_custom_impl_Yolo.so`. Згенероване значення є відносним шляхом
`nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`, який розв'язується,
коли `deepstream-app` запускають із каталогу `DeepStream-Yolo`, а в інших
випадках його треба відредагувати.

## Запуск пайплайна

Перш ніж витрачати час на щось інше, перевірте, чи бачить контейнер GPU. Саме з
цієї перевірки почався валідаційний запуск на карті Blackwell під WSL2.

<code-tabs name="gpu" />

У валідаційному запуску `deepstream-app` працював з одним файловим джерелом, без
sink для показу, з увімкненим екранним відображенням і встановленим
`gie-kitti-output-dir`, щоб виявлення з кожного кадру потрапляли на диск як текст
KITTI. Конфіг із такими налаштуваннями:

<code-tabs name="run" />

`nvinfer` збирає рушій TensorRT з ONNX під час першого запуску і кешує його поруч
із моделлю, тож перший запуск платить за збірку рушія, а наступні лише
завантажують кеш.

## Згенерований конфіг

Обидва наведені нижче конфіги записав експортер для валідаційного запуску, і
після цього їх не редагували.

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

Ці два конфіги відрізняються у трьох місцях: `maintain-aspect-ratio`,
`cluster-mode` і наявність `nms-iou-threshold` взагалі. Конфіг моделі D-FINE
повністю пропускає цей ключ, чого й вимагає `cluster-mode=4`.

Голови, що видають щонайбільше одне передбачення на об'єкт, отримують
`cluster-mode=4`, тож DeepStream не кластеризує їхні виходи; кластеризація злила
б справді різні виявлення. Сюди належать `rfdetr`, `dfine`, `deim`, `deimv2`,
`ec`, `rtdetr`, `rtdetrv2`, `rtdetrv4` і `yolo9_e2e`. Сіткові та якірні голови
отримують `cluster-mode=2` разом із `nms-iou-threshold`.

Конфіги виявлення також містять `engine-create-func-name=NvDsInferYoloCudaEngineGet`,
який передає збірку рушія бібліотеці парсера. Саме це закріплює назву файлу кешу
рушія, і саме звідси походить колізія, описана серед відомих пасток.

## Підтримувані задачі та сімейства

Експортуються сорок три комбінації сімейства і задачі. Функції
`deepstream_supported_tasks()` та `deepstream_supported_families(task)` у
`libreyolo/export/deepstream.py` повертають ті самі списки під час виконання.

| Задача | `network-type` | Бібліотека парсера | Сімейства |
|---|---|---|---|
| Виявлення | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Класифікація | 1 | Не потрібна | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Семантична сегментація | 2 | Не потрібна | pidnet, eomt, dinov2, lingbotvision |
| Сегментація екземплярів | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Поза | 100 | Не потрібна | yolo9, yolonas, rfdetr, ec |
| Глибина | 100 | Не потрібна | depth_anything, zipdepth |
| Відновлення | 100 | Не потрібна | nafnet, realesrgan, swinir |
| Matting | 100 | Не потрібна | birefnet |
| Погляд | 100 | Не потрібна | l2cs |

`network-type=100` означає, що для цієї задачі DeepStream не має постобробника.
Такі конфіги встановлюють `output-tensor-meta=1`, рідні виходи графа проходять
без змін, а застосунок декодує їх із метаданих тензорів. Графи з кількома
виходами тут не проблема: кожен вихідний шар потрапляє в метадані з тими самими
назвами виходів і динамічними осями, що й у звичайному експорті в ONNX.

Рядки сегментації екземплярів складаються з рядка виявлення, за яким іде маска
цього екземпляра, розгорнута у розмірі `(netH / 4, netW / 4)`, тобто в
роздільній здатності, жорстко зашитій у парсері сегментації, у вигляді
ймовірностей для `segmentation-threshold`.

Класифікація та погляд працюють як вторинний інференс. Щоб поставити класифікатор
за детектором, встановіть у згенерованому конфігу `process-mode=2` та
`operate-on-gie-id`. Погляд має контракт лише для голови, один вирізаний фрагмент
обличчя на вхід, тож перед ним потрібен детектор облич.

Трьох сімейств немає навмисно. `segformer` не підключено до спільного контракту
семантичного експорту, і воно не експортується в ONNX у жодному вигляді. У
моделей RTMDet-Ins і YOLO9 експорт сегментації екземплярів заблоковано в самому
LibreYOLO. `depth_anything3` не має реалізації експорту.

За двома рядками таблиці стоять прогалини в контрольних точках. З семантичних
контрольних точок EoMT опубліковано лише `l`, а для класифікації DINOv2 не
опубліковано жодної, тож ця комбінація потребує ваших власних донавчених ваг.

## Відмінності попередньої обробки

`nvinfer` обчислює `net-scale-factor * (x - offsets)` для кожного каналу зі
скалярним масштабом, який не може виразити стандартне відхилення на канал.
Сімейства, яким воно потрібне (`rfdetr`, `ec`, розміри `deimv2` з бекбоном DINO,
`rtmdet`, `picodet` і всі сімейства класифікації), мають нормалізацію, вбудовану
в експортований граф, а згенерований конфіг подає графу відповідний сирий вхідний
простір.

Геометрія лишається тим місцем, де власні Python-пайплайни LibreYOLO і `nvinfer`
все ще розходяться:

- Сімейства з letterbox (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`, `yolo3`,
  `yolo4`, `yolo7`) нативно доповнюють поля сірим. `nvinfer` доповнює чорним.
- Виявлення `yolonas` нативно масштабує найдовшу сторону до 636 усередині свого
  полотна 640. Натомість `maintain-aspect-ratio` у `nvinfer` використовує всі 640.
- Класифікація нативно масштабує найкоротшу сторону, а потім робить центральне
  обрізання. `nvinfer` розтягує кадр або ROI об'єкта до входу мережі, тому щільно
  обрізані об'єкти виглядають інакше.
- Для семантичної сегментації модель EoMT нативно проходить тайлами ковзного
  вікна. Експортований граф працює з одним розтягнутим полотном, що швидше і менш
  точно.
- `pidnet` видає карту класів у 1/8 вхідної роздільної здатності, а
  `lingbotvision` у 1/16. DeepStream збільшує карту класів для показу.

Перевірка паритету ONNX подає вже попередньо оброблені тензори, тому вона
перевіряє виходи графа і не може виявити неправильний порядок кольорів чи
політику доповнення в конфігу. Перед розгортанням навантаження, що потребує
точного паритету, перевірте результат на власних даних.

## Відомі пастки

### Дві моделі виявлення в одному каталозі завантажують рушій одна одної

Кожен конфіг виявлення містить той самий рядок:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

Складальник рушія в парсері вимагає саме такої базової назви, і вона не залежить
від моделі. Експортуйте другу модель виявлення в той самий каталог, і другий
запуск завантажить кешований рушій першої моделі. Нічого не падає, просто рамки
виходять неправильні. Виділіть кожній моделі виявлення окремий каталог. У
валідаційному запуску довелося ізолювати D-FINE в окремий каталог, перш ніж її
взагалі вдалося протестувати.

### Рамка може нести лише один клас

Формат рядка в `nvinfer` має вигляд `[x1, y1, x2, y2, score, class_id]`, один
клас на рамку, тому експорт згортає оцінки класів до їхнього argmax. Рамка, яку
`predict` показує під двома класами, лишається під одним. Виміряний випадок:
LibreYOLO повідомляє `vase 0.773` і `bottle 0.383` для однієї рамки, а граф
DeepStream залишає `vase`. Це випливає з формату рядка в парсері й не змінюється
без відмови від цього контракту, тож це очікувана поведінка, а не регресія.

## Перевірено

`deepstream-app` дійшов до EOS із `App run successful` на обох типах голів
детектора, на вбудованому в NVIDIA `sample_1080p_h264.mp4` (1443 кадри), з
увімкненими покадровими дампами KITTI.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Тип голови | сіткова | один-до-одного |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Кадрів із виявленнями | 1443 | 1443 |
| Усього виявлень | 18031 | 71105 |

Гістограми класів на всіх 1443 кадрах для обох моделей ставлять автомобілі на
перше місце, а людей на друге, що правильно для вуличної сцени. Чотириразовий
розрив у кількості виявлень є наслідком різниці в `cluster-mode`: модель D-FINE
при `cluster-mode=4` не кластеризує нічого, тож виживає кожен запит понад
порогом, разом із майже дублікатами.

Дві незалежно навчені моделі розміщують домінантний об'єкт в одному й тому самому
місці:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Цей запуск підтверджує п'ять речей: TensorRT будує рушій з експортованого ONNX на
sm_120, `nvinfer` приймає кожен ключ згенерованого конфіга, `NvDsInferParseYolo`
правильно читає розкладку тензора, рамки потрапляють у координати вихідної
роздільної здатності 1920x1080, а мітки розв'язуються за згенерованим файлом
міток.

Оточення, в якому це працювало:

| Компонент | Значення |
|---|---|
| ОС хоста | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 ГБ |
| Драйвер | 591.86 |
| Обчислювальна спроможність | 12.0 (Blackwell, sm_120) |
| Середовище виконання контейнерів | Docker Desktop 29.4.3, бекенд WSL2 |
| Образ DeepStream | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| Версія DeepStream | 8.0.0 |
| CUDA в контейнері | 12.8.1 |
| Парсер | `marcoslucianops/DeepStream-Yolo` на HEAD |

Поряд із запуском пайплайна `tests/unit/test_deepstream_export.py` покриває
адаптери графа та ключі згенерованого конфіга, і всі його 35 тестів проходять на
цьому коміті.

## Не перевірено

Перелічено, щоб описану вище область не читали ширше, ніж вона є.

- Jetson та aarch64. Контракт експорту не залежить від архітектури, але пайплайн
  запускали лише на дискретному GPU x86.
- Сорок одна з 43 комбінацій. Через DeepStream пройшли лише виявлення з `yolo9` і
  виявлення з `dfine`. Класифікацію, семантичну сегментацію, сегментацію
  екземплярів і задачі із сирими тензорами покривають модульні тести та перевірки
  паритету ONNX, а не запуск пайплайна.
- FP16 та INT8. Перевіряли лише `network-mode=0`.
- Кілька потоків і батчинг. Одне джерело, `batch-size=1`.
- Правильність відносно датасету з еталонною розміткою. Виявлення перевіряли на
  семантичну правдоподібність і узгодженість між моделями, а не оцінювали як mAP
  через DeepStream.
