---
title: Python API
seo_title: Справочник по Python API LibreYOLO
description: >-
  Имена, которые LibreYOLO экспортирует на уровне пакета: пять фабрик, классы
  семейств, полезная нагрузка Results, бэкенды, валидаторы, трекеры и
  вспомогательные функции для данных.
lead: >-
  Публичная поверхность Python в LibreYOLO — это список __all__ в
  libreyolo/__init__.py. Всё, что есть на этой странице, импортируется как from
  libreyolo import <name>; всё, чего в этом списке нет, — внутреннее.
keywords:
  - libreyolo python api
  - импорт libreyolo
  - фабрики libreyolo
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  Имена и сигнатуры прочитаны из libreyolo/__init__.py,
  libreyolo/models/__init__.py, libreyolo/models/base/model.py,
  libreyolo/models/base/inference.py, libreyolo/models/sam/model.py,
  libreyolo/models/vlm/__init__.py, libreyolo/models/openvocab/__init__.py и
  libreyolo/ensemble/model.py на версии 1.5.0.
snippets:
  usage:
    - label: Загрузка чего угодно через одну фабрику
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # Один источник-изображение возвращает один Results; список или каталог
        # возвращает их список.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: Прямой импорт класса семейства
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: Пять точек входа
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # Фабрика, определяющая модель по весам, — для семейств без промптов.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # Два детектора или больше за одной поверхностью предсказания.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # Остальным трём фабрикам нужен установленный extra:

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## Точки входа

Модель загружают пять вызываемых объектов. Они разделены по контракту вызова,
а не по архитектуре.

| Фабрика | Что загружает | Промпт при вызове | Нужен extra |
|---|---|---|---|
| `LibreYOLO` | Семейства без промптов, по содержимому чекпойнта или суффиксу файла | | |
| `LibreSAM` | Сегментаторы с промптами, по алиасу размера | Точки, рамки или текст концепта | `sam` |
| `LibreVLM` | Генеративные vision-language детекторы, по алиасу | Словарь классов или произвольный промпт | `vlm` |
| `LibreOpenVocab` | Детекторы, обусловленные текстом, по алиасу | Словарь классов | `openvocab` |
| `LibreEnsemble` | Два детектора или больше, объединённые в одну поверхность | | |

<code-tabs name="factories" />

Только `LibreYOLO` читает файл. Остальные три принимают строковый алиас и
разрешают его в репозиторий на Hugging Face, поэтому аргумент здесь — имя
модели, а не путь.

```python
LibreYOLO(
    model_path: str,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
    task: str | None = None,
    compute_units: str = "all",
)
```

`model_path` принимает чекпойнт `.pt`, ONNX-файл `.onnx`, ExecuTorch-файл
`.pte`, MNN-файл `.mnn`, TensorRT-файл `.engine`, каталог OpenVINO, Paddle или
ncnn, а также HTTP- или HTTPS-URL модели в Triton. Если `size` и `nb_classes` опущены,
они читаются из чекпойнта. `compute_units` учитывается только при загрузке
CoreML `.mlpackage` и принимает одно из значений `all`, `cpu_only`,
`cpu_and_gpu`, `cpu_and_ne`. `task` принимает любое каноническое имя задачи из
`libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Классы семейств

Каждое семейство, которое может вернуть фабрика, экспортируется и под
собственным именем, поэтому класс можно создать напрямую, когда чекпойнт
известен заранее. Конструкторы повторяют `BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

У класса семейства `size` не имеет значения по умолчанию — в этом и отличие
от фабрики. YOLO9 и его варианты вставляют `reg_max: int = 16`
после `size`.

Семейства для детекции и многозадачные семейства: `LibreYOLO9`,
`LibreYOLO9E2E`, `LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`,
`LibreYOLO4`, `LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`,
`LibreRTDETRv2`, `LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`,
`LibreDOMEDETR`, `LibreDEIM`, `LibreDEIMv2`, `LibreDETR`,
`LibreDeformableDETR`, `LibreDINODETR`, `LibreLWDETR`, `LibreMaskRCNN`,
`LibreFCOS`, `LibreFasterRCNN`, `LibreRetinaNet`, `LibreSSD`,
`LibreCenterNet`, `LibreEfficientDet`, `LibreEC`, `LibrePICODET`,
`LibreRTMDet`, `LibreFOMO`.

Семейства плотного предсказания: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Семейства классификации и эмбеддингов: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Другие задачи: `LibreHRNet` (pose), `LibreL2CS` (gaze), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

Родственные уровни тоже экспортируют свои классы семейств: `LibreSAM1`,
`LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (также пишется
`LibreModus`).

## Поверхность предсказания

Вызов модели запускает инференс. `predict` — алиас для `__call__`, так что они
взаимозаменяемы.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

Один источник-изображение возвращает один `Results`. Список, кортеж или
каталог возвращает их список, а `stream=True` возвращает генератор. Остальные
методы объекта модели описаны на
[странице API модели](/docs/reference/model-api).

## Полезная нагрузка Results

`Results` и его восемнадцать классов полезной нагрузки экспортируются на
уровне пакета: `Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`,
`OBB`, `Gaze`, `SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`,
`NormalMap`, `RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`,
`Identities`. Каждый из них описан в разделе
[Типы Results](/docs/reference/results-types).

## Бэкенды

Экспортированные артефакты загружаются через `LibreYOLO()` по суффиксу файла,
поэтому классы бэкендов редко создают вручную. Они экспортированы для случаев,
когда бэкенд нужно выбрать явно: `OnnxBackend`, `OpenVINOBackend`,
`PaddleBackend`, `TensorRTBackend`, `TritonBackend`, `NcnnBackend`,
`CoreMLBackend`, плюс `create_triton_config`. `BaseExporter` — реестр
экспортёров, который стоит за `model.export()`.

## Валидаторы

`model.val()` выбирает нужный валидатор по задаче, поэтому эти классы
экспортированы для прямого использования и для наследования:
`DetectionValidator`, `SegmentationValidator`, `PoseValidator`,
`SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`,
`EdgeValidator`, а также общий `ValidationConfig`.

## Трекинг

`model.track()` выбирает трекер по имени. Классы трекеров и
dataclass-конфигурации к ним тоже экспортированы: `ByteTracker` с
`TrackConfig`, `BoTSortTracker` с `BoTSortConfig` и `OCSortTracker` с
`OCSortConfig`.

## Вспомогательные функции для данных

`DATASETS_DIR` — вычисленный корень датасетов, `load_data_config` читает YAML
датасета, а `check_dataset` его проверяет. Загрузчики под конкретные задачи,
перечисленные в разделе
[Форматы датасетов](/docs/reference/dataset-formats), живут в `libreyolo.data`,
а не на уровне пакета.

## Галереи и дистилляция

`Gallery` и `FaceGallery` хранят зарегистрированные векторы личностей для
задачи `embed` и выдают полезную нагрузку `Identities`. `Distiller` и
`get_distill_config` управляют обучением по схеме «учитель — ученик».

## Ресурсы

`SAMPLE_IMAGE` — абсолютный путь к изображению, которое поставляется вместе с
пакетом, поэтому любой пример кода в этой документации запускается без
предварительного скачивания картинки.

## Ленивые импорты и переименованные классы

Большинство имён родственных уровней, бэкенды, валидаторы и вспомогательные
функции для данных разрешаются через `__getattr__` на уровне модуля, поэтому
импорт `libreyolo` не тянет за собой их зависимости. Если нужный extra не установлен,
импорт всё равно падает с понятным сообщением.

Два класса переименовали, и старое написание всё ещё разрешается — с
`DeprecationWarning`: `LibreYOLORTDETR` теперь `LibreRTDETR`, а
`LibreYOLORFDETR` теперь `LibreRFDETR`.
