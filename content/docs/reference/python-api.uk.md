---
title: Python API
seo_title: "Довідник Python API LibreYOLO"
description: "Назви, які LibreYOLO експортує на рівні пакета: п'ять фабрик, класи сімейств, корисні дані Results, бекенди, валідатори, трекери та допоміжні засоби для даних."
lead: "Публічна поверхня Python у LibreYOLO визначається списком __all__ у libreyolo/__init__.py. Усе на цій сторінці можна імпортувати як from libreyolo import <name>; усе, чого немає в списку, є внутрішнім."
keywords:
  - "libreyolo python api"
  - "libreyolo import"
  - "фабрика LibreYOLO"
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - "libreyolo __all__"
last_verified: "1.5.0"
verification: "Назви й сигнатури взято з libreyolo/__init__.py, libreyolo/models/__init__.py, libreyolo/models/base/model.py, libreyolo/models/base/inference.py, libreyolo/models/sam/model.py, libreyolo/models/vlm/__init__.py, libreyolo/models/openvocab/__init__.py та libreyolo/ensemble/model.py у v1.5.0."
snippets:
  usage:
    - label: Завантаження будь-чого через одну фабрику
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # Джерело з одним зображенням повертає один Results; список або каталог
        # повертає список таких об'єктів.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: Безпосередній імпорт класу сімейства
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: П'ять точок входу
      language: python
      code: |
        from libreyolo import LibreYOLO, LibreEnsemble

        # Фабрика з аналізом ваг для сімейств без підказок.
        detector = LibreYOLO("LibreYOLO9t.pt")

        # Два або більше детекторів за однією поверхнею передбачення.
        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # Для інших трьох фабрик потрібно встановити додатковий пакет:
        #   pip install 'libreyolo[sam]'        -> from libreyolo import LibreSAM
        #   pip install 'libreyolo[vlm]'        -> from libreyolo import LibreVLM
        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import LibreOpenVocab
        print(type(detector).__name__, ens.fusion)
---

## Точки входу

Модель завантажують п'ять викликаних об'єктів. Їх розділено за контрактом
виклику, а не за архітектурою.

| Фабрика | Завантажує | Підказка під час виклику | Потрібний додатковий пакет |
|---|---|---|---|
| `LibreYOLO` | Сімейства без підказок через аналіз контрольної точки або суфікса файла | | |
| `LibreSAM` | Сегментатори з підказками за псевдонімом розміру | Точки, рамки або текст концепції | `sam` |
| `LibreVLM` | Генеративні візуально-мовні детектори за псевдонімом | Словник класів або довільна підказка | `vlm` |
| `LibreOpenVocab` | Текстово-керовані детектори за псевдонімом | Словник класів | `openvocab` |
| `LibreEnsemble` | Два або більше детекторів, злитих в одну поверхню | | |

<code-tabs name="factories" />

Лише `LibreYOLO` читає файл. Інші три приймають рядковий псевдонім і визначають
за ним репозиторій Hugging Face, тому аргумент є назвою моделі, а не шляхом.

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

`model_path` приймає контрольну точку `.pt`, файл ONNX `.onnx`, ExecuTorch
`.pte`, MNN `.mnn`, TensorRT `.engine`, каталог OpenVINO, Paddle чи ncnn або
URL моделі Triton HTTP чи HTTPS. Якщо `size` і `nb_classes` не задано, вони
зчитуються з контрольної точки. `compute_units` зчитується лише для
завантаження CoreML `.mlpackage` і має одне зі значень `all`, `cpu_only`,
`cpu_and_gpu`, `cpu_and_ne`. `task` приймає будь-яку канонічну назву завдання
з `libreyolo.tasks.TASKS`.

<code-tabs name="usage" />

## Класи сімейств

Кожне сімейство, яке може повернути фабрика, також експортується за назвою,
тому клас можна створити безпосередньо, якщо контрольна точка відома заздалегідь.
Конструктори відповідають `BaseModel.__init__`:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

У класі сімейства `size` не має типового значення, що відрізняє його від
фабрики. YOLO9 та його варіанти вставляють `reg_max: int = 16` після `size`.

Сімейства виявлення та багатозадачні сімейства: `LibreYOLO9`, `LibreYOLO9E2E`,
`LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`,
`LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`,
`LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`,
`LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`,
`LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`,
`LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`,
`LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

Сімейства щільного передбачення: `LibreMiDaS`, `LibreDepthAnythingV2`,
`LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`,
`LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`,
`LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`,
`LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

Сімейства класифікації та ембедінгів: `LibreViT`, `LibreMobileNetV4`,
`LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`,
`LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

Інші завдання: `LibreHRNet` (поза), `LibreL2CS` (погляд), `LibrePPOCR` (ocr),
`LibreFaceEmbedder` (embed).

Суміжні рівні також експортують свої класи сімейств: `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`;
`LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`,
`LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`,
`LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (також пишеться
`LibreModus`).

## Поверхня передбачення

Виклик моделі запускає інференс. `predict` є псевдонімом `__call__`, тому вони
взаємозамінні.

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

Джерело з одним зображенням повертає один `Results`. Список, кортеж або каталог
повертає список таких об'єктів, а `stream=True` повертає генератор. Інші методи
об'єкта моделі документовано на сторінці [API моделі](/docs/reference/model-api).

## Корисні дані Results

`Results` та його вісімнадцять класів корисних даних експортуються на рівні
пакета: `Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`,
`Gaze`, `SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`,
`NormalMap`, `RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`,
`Identities`. Кожен описано в розділі
[Типи Results](/docs/reference/results-types).

## Бекенди

Експортовані артефакти завантажуються через `LibreYOLO()` за суфіксом файла,
тому класи бекендів рідко створюються вручну. Їх експортовано для випадків, коли
бекенд потрібно вибрати явно: `OnnxBackend`, `OpenVINOBackend`, `PaddleBackend`,
`TensorRTBackend`, `TritonBackend`, `NcnnBackend`, `CoreMLBackend`, а також
`create_triton_config`. `BaseExporter` є реєстром експортерів, що стоїть за
`model.export()`.

## Валідатори

`model.val()` спрямовує виклик до правильного валідатора за завданням, тому
наведені класи експортуються для безпосереднього використання й створення
підкласів: `DetectionValidator`, `SegmentationValidator`, `PoseValidator`,
`SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`,
`EdgeValidator` і спільний `ValidationConfig`.

## Відстеження

`model.track()` вибирає трекер за назвою. Класи трекерів і їхні dataclass
конфігурації також експортуються: `ByteTracker` із `TrackConfig`,
`BoTSortTracker` із `BoTSortConfig` та `OCSortTracker` із `OCSortConfig`.

## Допоміжні засоби для даних

`DATASETS_DIR` є визначеним коренем датасетів, `load_data_config` зчитує YAML
датасету, а `check_dataset` перевіряє його. Завантажувачі окремих завдань,
названі в [Форматах датасетів](/docs/reference/dataset-formats), розташовані в
`libreyolo.data`, а не на рівні пакета.

## Галереї та дистиляція

`Gallery` і `FaceGallery` зберігають зареєстровані вектори ідентичностей для
завдання `embed` і створюють корисні дані `Identities`. `Distiller` і
`get_distill_config` керують навчанням учитель-учень.

## Ресурси

`SAMPLE_IMAGE` є абсолютним шляхом до зображення, включеного до пакета, тому
кожен фрагмент цієї документації запускається без попереднього завантаження зображення.

## Ліниві імпорти та перейменовані класи

Більшість назв суміжних рівнів, бекенди, валідатори та допоміжні засоби для
даних визначаються через `__getattr__` на рівні модуля, тому імпорт `libreyolo`
не імпортує їхні залежності. Якщо потрібного додаткового пакета немає, імпорт
усе одно завершується зрозумілим повідомленням.

Два класи перейменовано, а старий варіант написання досі визначається з
`DeprecationWarning`: `LibreYOLORTDETR` тепер має назву `LibreRTDETR`, а
`LibreYOLORFDETR` тепер має назву `LibreRFDETR`.
