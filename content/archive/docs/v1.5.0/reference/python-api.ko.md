---
title: 파이썬 API
seo_title: LibreYOLO Python API 참조
description: >-
  LibreYOLO가 패키지 수준에서 내보내는 이름: 다섯 개의 팩토리, 계열 클래스, Results 페이로드, 백엔드, 검증기, 트래커 및
  데이터 도우미.
lead: >-
  LibreYOLO의 공용 Python 표면은 libreyolo/__init__.py.에 있는 __all__ 목록입니다. 이 페이지의 모든
  항목은 from libreyolo import <name>; 와 같이 가져올 수 있으며, 그 목록에 없는 것은 내부용입니다.
keywords:
  - libreyolo 파이썬 API
  - libreyolo 가져오기
  - 리브레YOLO 팩토리
  - 리브레샘
  - 리브레VLM
  - 리브레오픈보캡
  - 리브르앙상블
  - libreyolo __all__
last_verified: 1.5.0
verification: >-
  v1.5.0에서 libreyolo/__init__.py, libreyolo/models/__init__.py,
  libreyolo/models/base/model.py, libreyolo/models/base/inference.py,
  libreyolo/models/sam/model.py, libreyolo/models/vlm/__init__.py,
  libreyolo/models/openvocab/__init__.py 및 libreyolo/ensemble/model.py에서 이름과 서명
  읽기.
snippets:
  usage:
    - label: 하나의 팩토리를 통해 아무 것이나 적재하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # 단일 이미지 소스는 하나의 결과를 반환합니다; 목록 또는 디렉토리
        # 그들의 목록을 반환합니다.
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
        print(result.names)
    - label: 계열 클래스를 직접 가져오기
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


        # 무게를 맡는 팩토리가 무자극 계열들 위에 있습니다.

        detector = LibreYOLO("LibreYOLO9t.pt")


        # 하나의 예측 표면 뒤에 두 개 이상의 탐지기.

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])


        # 나머지 세 팩토리에는 추가 설치가 필요합니다:

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

다섯 개의 호출 가능 객체가 모델을 로드합니다. 이들은 아키텍처가 아니라 호출 계약에 따라 구분됩니다.

| 팩토리 | 많음 | 호출 시 프롬프트 | 추가 필요 |
|---|---|---|---|
| `LibreYOLO` | 체크포인트나 파일 확장자를 냄새 맡으며 프롬프트 없는 계열들 | | |
| `LibreSAM` | 프롬프트 가능한 세그멘터, 크기 별칭으로 | 포인트, 박스, 또는 개념 텍스트 | `sam` |
| `LibreVLM` | 별칭에 의한 생성적 시각-언어 탐지기 | 수업 어휘 또는 자유 형식 프롬프트 | `vlm` |
| `LibreOpenVocab` | 텍스트 조건 탐지기, 별칭으로 | 수업 어휘 | `openvocab` |
| `LibreEnsemble` | 두 개 이상의 탐지기가 하나의 표면으로 융합됨 | | |

<code-tabs name="factories" />

`LibreYOLO`는 파일을 읽는 유일한 것입니다. 나머지 세 가지는 문자열 별칭을 받아 그것을 Hugging Face 저장소로 해석하므로, 인수는 경로가 아니라 모델 이름입니다.

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

`model_path`는 `.pt` 체크포인트, ONNX `.onnx` 파일, ExecuTorch `.pte`, MNN `.mnn`, TensorRT `.engine`, OpenVINO, Paddle 또는 ncnn 디렉터리, 또는 Triton HTTP 혹은 HTTPS 모델 URL을 허용합니다. `size`와 `nb_classes`는 생략될 경우 체크포인트에서 읽어옵니다. `compute_units`는 CoreML `.mlpackage` 로드에서만 읽어오며 `all`, `cpu_only`, `cpu_and_gpu`, `cpu_and_ne` 중 하나입니다. `task`는 `libreyolo.tasks.TASKS`의 모든 정식 작업 이름을 사용할 수 있습니다.

<code-tabs name="usage" />

## 계열 수업

팩토리가 반환할 수 있는 모든 클래스는 이름으로도 내보내지므로, 체크포인트를 미리 알고 있다면 클래스를 직접 구성할 수 있습니다. 생성자는 `BaseModel.__init__`를 따릅니다:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`size`는 계열 클래스에 기본값이 없으며, 이것이 팩토리과의 차이점입니다. YOLO9와 그 변형들은 `size` 뒤에 `reg_max: int = 16`를 삽입합니다.

검출 및 다중 작업 계열: `LibreYOLO9`, `LibreYOLO9E2E`, `LibreYOLO9P2`, `LibreYOLONAS`, `LibreYOLOX`, `LibreYOLO7`, `LibreYOLO4`, `LibreYOLO3`, `LibreYOLO2`, `LibreYOLO1`, `LibreRTDETR`, `LibreRTDETRv2`, `LibreRTDETRv4`, `LibreRFDETR`, `LibreDFINE`, `LibreDOMEDETR`, `LibreDEIM`, `LibreDEIMv2`, `LibreDETR`, `LibreDeformableDETR`, `LibreDINODETR`, `LibreLWDETR`, `LibreMaskRCNN`, `LibreFCOS`, `LibreFasterRCNN`, `LibreRetinaNet`, `LibreSSD`, `LibreCenterNet`, `LibreEfficientDet`, `LibreEC`, `LibrePICODET`, `LibreRTMDet`, `LibreFOMO`.

밀집 예측 계열: `LibreMiDaS`, `LibreDepthAnythingV2`, `LibreDepthAnything3`, `LibreZipDepth`, `LibreMoGe2`, `LibreTEED`, `LibreDexiNed`, `LibreNAFNet`, `LibreRealESRGAN`, `LibreSwinIR`, `LibreBiRefNet`, `LibreFeyNobg`, `LibreFCN`, `LibreEoMT`, `LibreDeepLabv3`, `LibrePIDNet`, `LibreSegformer`, `LibreLingBotVision`.

분류 및 임베딩 계열: `LibreViT`, `LibreMobileNetV4`, `LibreConvNeXt`, `LibreDeiT`, `LibreSwin`, `LibreEfficientNetV2`, `LibreVGG`, `LibreResNet`, `LibreAlexNet`, `LibreCLIP`, `LibreSigLIP2`, `LibreDINOv2`.

다른 작업: `LibreHRNet` (포즈), `LibreL2CS` (시선), `LibrePPOCR` (OCR), `LibreFaceEmbedder` (임베드).

형제 계층은 그들의 계열 클래스도 내보냅니다: `LibreSAM1`, `LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM`, `LibrePicoSAM3`; `LibreGroundingDINO`, `LibreOWLv2`, `LibreOMDetTurbo`; `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`, `LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything`, `LibreMODUS` (`LibreModus`로도 표기됨).

## 예측 표면

모델을 호출하면 추론이 실행됩니다. `predict`는 `__call__`의 별칭이므로 두 개는 서로 교환할 수 있습니다.

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

단일 이미지 소스는 하나의 `Results`를 반환합니다. 리스트, 튜플 또는 디렉토리는 그것들의 리스트를 반환하며, `stream=True`는 제너레이터를 반환합니다. 모델 객체의 다른 메서드는 [모델 API 페이지](/docs/reference/model-api)에서 문서화되어 있습니다.

## 결과 페이로드

`Results`과 그 18개의 페이로드 클래스는 패키지 수준에서 내보내집니다: `Results`, `Boxes`, `Masks`, `Keypoints`, `Points`, `Probs`, `OBB`, `Gaze`, `SemanticMask`, `PanopticSegmentation`, `DepthMap`, `EdgeMap`, `NormalMap`, `RestoredImage`, `Matte`, `Meshes`, `OCRRegions`, `Embeddings`, `Identities`. 각각은 [결과 유형](/docs/reference/results-types)에서 설명됩니다.

## 백엔드

내보낸 아티팩트는 파일 확장자를 통해 `LibreYOLO()`로 로드되므로, 백엔드 클래스는 거의 수동으로 생성되지 않습니다. 이러한 클래스들은 백엔드를 명시적으로 선택해야 하는 경우를 위해 내보내집니다: `OnnxBackend`, `OpenVINOBackend`, `PaddleBackend`, `TensorRTBackend`, `TritonBackend`, `NcnnBackend`, `CoreMLBackend`, 그리고 `create_triton_config`. `BaseExporter`는 `model.export()` 뒤에 있는 내보내기 레지스트리입니다.

## 검증자

`model.val()`는 작업별로 올바른 검증자에게 전송되므로, 다음은 직접 사용 및 서브클래싱을 위해 내보내집니다: `DetectionValidator`, `SegmentationValidator`, `PoseValidator`, `SemanticValidator`, `PanopticValidator`, `DepthValidator`, `NormalValidator`, `EdgeValidator`, 그리고 공유된 `ValidationConfig`.

## 추적

`model.track()`는 이름으로 트래커를 선택합니다. 트래커 클래스와 그 구성 데이터 클래스도 내보내집니다: `ByteTracker`와 `TrackConfig`, `BoTSortTracker`와 `BoTSortConfig`, 그리고 `OCSortTracker`와 `OCSortConfig`.

## 데이터 도우미

`DATASETS_DIR`는 해결된 데이터셋 루트이며, `load_data_config`는 데이터셋 YAML을 읽고, `check_dataset`는 이를 검증합니다. [데이터셋 형식](/docs/reference/dataset-formats)에 명시된 작업별 로더들은 패키지 수준이 아닌 `libreyolo.data`에 위치합니다.

## 갤러리와 증류

`Gallery`와 `FaceGallery`는 `embed` 작업을 위해 등록된 정체 벡터를 보유하고 `Identities` 페이로드를 생성합니다. `Distiller`와 `get_distill_config`는 교사-학생 학습을 진행합니다.

## 자산

`SAMPLE_IMAGE`는 패키지와 함께 번들된 이미지에 대한 절대 경로이므로, 이 문서의 모든 코드 조각은 먼저 그림을 다운로드하지 않고도 실행됩니다.

## 지연 가져오기와 이름이 바뀐 클래스

대부분의 형제 계층 이름, 백엔드, 검증기 및 데이터 도우미는 모듈 수준 `__getattr__`를 통해 해결되므로, `libreyolo`를 가져와도 그들의 종속성은 가져오지 않습니다. 필요한 추가 항목이 없으면 가져오기는 여전히 명확한 메시지와 함께 실패합니다.

두 개의 클래스 이름이 변경되었으며, 이전 철자는 여전히 해결됩니다. `DeprecationWarning`: `LibreYOLORTDETR`는 이제 `LibreRTDETR`이고, `LibreYOLORFDETR`는 이제 `LibreRFDETR`입니다.
