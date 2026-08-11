---
title: Python API
seo_title: LibreYOLO Python API 참조
description: >-
  패키지 수준에서 LibreYOLO가 내보내는 이름: 다섯 개의 팩토리, 패밀리 클래스, Results 페이로드, 백엔드, 검증기, 트래커 및
  데이터 헬퍼.
lead: >-
  LibreYOLO의 공개 Python 인터페이스는 libreyolo/__init__.py.의 __all__ 목록입니다. 이 페이지의 모든
  항목은 'from libreyolo import <name>'로 가져올 수 있으며, 해당 목록에 없는 항목은 내부용입니다.
keywords:
  - libreyolo python api
  - libreyolo import
  - LibreYOLO 팩토리
  - LibreSAM
  - LibreVLM
  - LibreOpenVocab
  - LibreEnsemble
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  v1.5.0에서 libreyolo/__init__.py, libreyolo/models/__init__.py,
  libreyolo/models/base/model.py, libreyolo/models/base/inference.py,
  libreyolo/models/sam/model.py, libreyolo/models/vlm/__init__.py,
  libreyolo/models/openvocab/__init__.py 및 libreyolo/ensemble/model.py에서 읽어온 이름과
  시그니처.
snippets:
  usage:
    - label: 하나의 팩토리를 통해 모든 항목을 로드
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # 단일 이미지 소스는 하나의 결과를 반환합니다; 목록 또는 디렉토리
        # 그것들의 목록을 반환합니다.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: 패밀리 클래스를 직접 가져오기
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        model = LibreYOLO9("LibreYOLO9t.pt", size="t")
        result = model(SAMPLE_IMAGE)

        print(len(result))
  factories:
    - label: 다섯 개의 진입점
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreEnsemble


        # 프롬프트 없는 패밀리 위에서 가중치 탐지 팩토리.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # 하나의 예측 표면 뒤에 두 개 이상의 탐지기.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # 나머지 세 개의 팩토리은 추가 설치가 필요합니다:

        #   pip install 'libreyolo[sam]'        -> from libreyolo import
        LibreSAM

        #   pip install 'libreyolo[vlm]'        -> from libreyolo import
        LibreVLM

        #   pip install 'libreyolo[openvocab]'  -> from libreyolo import
        LibreOpenVocab

        print(type(detector).__name__, ens.fusion)
source_hash: 66e34e78b2e0fb2d
---

## 진입점

다섯 개의 호출 가능 항목이 모델을 로드합니다. 이들은 아키텍처가 아니라 호출 계약에 따라 구분됩니다.

| 팩토리 | 로드 | 호출 시 프롬프트 | 추가 요구 사항 |
|---|---|---|---|
| `LibreYOLO` | 체크포인트나 파일 접미사를 탐지하여 프롬프트 없는 계열 | | |
| `LibreSAM` | 크기 별칭으로 프롬프트 가능한 분할기 | 포인트, 박스 또는 개념 텍스트 | `sam` |
| `LibreVLM` | 별칭으로 생성형 비전-언어 탐지기 | 클래스 어휘 또는 자유 형식 프롬프트 | `vlm` |
| `LibreOpenVocab` | 별칭으로 텍스트 조건 탐지기 | 클래스 어휘 | `openvocab` |
| `LibreEnsemble` | 두 개 이상의 탐지기를 하나의 표면으로 결합 | | |

<code-tabs name="factories" />

`LibreYOLO`는 파일을 읽는 유일한 것. 나머지 세 가지는 문자열 별칭을 받아 Hugging Face 저장소로 해석하므로, 인자는 경로가 아닌 모델 이름임.

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

`model_path`는 `.pt` 체크포인트, ONNX `.onnx` 파일, ExecuTorch `.pte`, MNN `.mnn`, TensorRT `.engine`, OpenVINO, Paddle 또는 ncnn 디렉터리, 또는 Triton HTTP 또는 HTTPS 모델 URL을 허용합니다. `size`와 `nb_classes`는 생략될 경우 체크포인트에서 읽힙니다. `compute_units`는 CoreML `.mlpackage` 로드에서만 읽히며, `all`, `cpu_only`, `cpu_and_gpu`, `cpu_and_ne` 중 하나입니다. `task`는 `libreyolo.tasks.TASKS`에서 임의의 표준 작업 이름을 가질 수 있습니다.

<code-tabs name="usage" />

## 패밀리 클래스

팩토리이 반환할 수 있는 모든 패밀리는 이름으로도 내보내져, 체크포인트가 미리 알려져 있는 경우 클래스를 직접 구성할 수 있습니다. 생성자는 `BaseModel.__init__`을 따릅니다:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`size`는 패밀리 클래스에서는 기본값이 없으며, 이는 팩토리 클래스와의 차이입니다. YOLO9 및 그 변형은 `size` 후에 `reg_max: int = 16`를 삽입합니다.

감지 및 다중 작업 계열: `LibreYOLO9`, `LibreYOLO9E2E`, `LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`, `LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`, `LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`, `LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`, `LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`, `LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`, `LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

밀집 예측 계열: `LibreMiDaS`, `LibreDepthAnythingV2`, `LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`, `LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`, `LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`, `LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

분류 및 임베딩 계열: `LibreViT`, `LibreMobileNetV4`, `LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`, `LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

기타 작업: `LibreHRNet` (포즈), `LibreL2CS` (시선), `LibrePPOCR` (OCR), `LibreFaceEmbedder` (임베드).

형제 계층도 그들의 계열 클래스를 내보냅니다: `LibreSAM1`, `LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`; `LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (`LibreModus`라고도 표기).

## 예측 표면

모델을 호출하면 추론이 실행됩니다. `predict`은 `__call__`의 별칭이므로 두 가지는 서로 교환 가능합니다.

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

단일 이미지 소스는 하나의 `Results`를 반환합니다. 리스트, 튜플 또는 디렉토리는 그들의 리스트를 반환하며, `stream=True`는 제너레이터를 반환합니다. 모델 객체의 다른 메서드는 [모델 API 페이지](/docs/reference/model-api)에 문서화되어 있습니다.

## 결과 페이로드

`Results`와 그 18개의 페이로드 클래스는 패키지 수준에서 다음과 같이 내보내집니다: `Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`, `Gaze`, `SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`, `NormalMap`, `RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`, `Identities`. 각 항목은 [결과 유형](/docs/reference/results-types)에서 설명되어 있습니다.

## 백엔드

내보낸 아티팩트는 파일 접미사를 통해 `LibreYOLO()`로 로드되므로, 백엔드 클래스는 거의 수작업으로 생성되지 않습니다. 이러한 클래스는 백엔드를 명시적으로 선택해야 하는 경우에 내보내집니다: `OnnxBackend`, `OpenVINOBackend`, `PaddleBackend`, `TensorRTBackend`, `TritonBackend`, `NcnnBackend`, `CoreMLBackend`, 그리고 `create_triton_config` 또한 포함됩니다. `BaseExporter`는 `model.export()` 뒤에 있는 내보내기 레지스트리입니다.

## 검증기

`model.val()`는 작업별로 올바른 검증기로 전달하므로, 다음과 같이 직접 사용하거나 서브클래싱을 위해 내보내집니다: `DetectionValidator`, `SegmentationValidator`, `PoseValidator`, `SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`, `EdgeValidator`, 및 공유된 `ValidationConfig`.

## 추적

`model.track()`은 이름으로 트래커를 선택합니다. 트래커 클래스와 그 구성 데이터 클래스도 내보내집니다: `ByteTracker`와 `TrackConfig`, `BoTSortTracker`와 `BoTSortConfig`, `OCSortTracker`와 `OCSortConfig`.

## 데이터 헬퍼

`DATASETS_DIR`는 해결된 데이터셋 루트이며, `load_data_config`는 데이터셋 YAML을 읽고, `check_dataset`는 이를 검증합니다. [데이터셋 형식](/docs/reference/dataset-formats)에서 언급된 작업별 로더는 패키지 수준이 아니라 `libreyolo.data`에 존재합니다.

## 갤러리와 증류

`Gallery`와 `FaceGallery`는 `embed` 작업을 위한 등록된 정체성 벡터를 보유하며, `Identities` 페이로드를 생성합니다. `Distiller`와 `get_distill_config`는 교사-학생 학습을 수행합니다.

## 자산

`SAMPLE_IMAGE`는 패키지에 포함된 이미지에 대한 절대 경로이므로, 이 문서의 모든 코드 조각은 먼저 이미지를 다운로드하지 않고 실행됩니다.

## 지연 임포트 및 이름 변경된 클래스

대부분의 동급 이름, 백엔드, 검증기 및 데이터 헬퍼는 모듈 수준의 `__getattr__`를 통해 해결되므로 `libreyolo`를 임포트해도 그들의 의존성이 임포트되지 않습니다. 필요한 추가 항목이 없으면 임포트는 여전히 명확한 메시지와 함께 실패합니다.

두 개의 클래스 이름이 변경되었으며, 이전 철자도 여전히 `DeprecationWarning`를 통해 해결됩니다: `LibreYOLORTDETR`는 이제 `LibreRTDETR`이고, `LibreYOLORFDETR`는 이제 `LibreRFDETR`입니다.
