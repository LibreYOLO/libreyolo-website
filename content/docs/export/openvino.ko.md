---
title: OpenVINO
seo_title: LibreYOLO에서 OpenVINO IR로 내보내기
description: >-
  LibreYOLO 모델을 OpenVINO IR로 변환합니다. model.xml 및 model.bin 파일 쌍, FP16 가중치 압축,
  NNCF INT8, CPU, GPU 또는 NPU 추론을 설명합니다.
lead: >-
  OpenVINO IR은 Intel의 런타임 형식으로, model.xml 그래프와 model.bin 가중치 블롭이 나란히 있습니다.
  LibreYOLO는 ONNX 중간 형식을 내보내고 ov.convert_model로 변환한 다음 같은 디렉터리에 metadata.yaml을
  작성합니다.
keywords:
  - yolo openvino 변환
  - openvino ir 모델
  - model.xml model.bin
  - ov.convert_model 사용법
  - nncf int8 양자화
  - openvino npu 추론
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="openvino")
    mono: true
  - label: 출력
    value: 'model.xml, model.bin, metadata.yaml이 있는 디렉터리'
  - label: 추가 설치
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: 다시 불러오기
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: 형상
    value: ONNX 중간 형식을 따릅니다. dynamic=True이면 동적 배치를 사용합니다.
  - label: 정밀도
    value: 'FP32, FP16 가중치 압축(half=True), NNCF를 통한 INT8(int8=True와 data=)을 지원합니다.'
verification: >-
  dev 브랜치의 libreyolo/export/openvino.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/openvino.py, pyproject.toml을
  확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        # IR은 ONNX 중간 형식에서 변환되므로 두 extra가 모두 필요합니다.
        pip install "libreyolo[onnx,openvino]"
    - label: INT8에 필요한 NNCF 추가 설치
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t_openvino 디렉터리를 작성합니다
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: 인수
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True이면 IR 전체에서 동적 배치 축을 유지합니다
            half=False,       # True이면 FP16 가중치를 저장합니다
            int8=False,       # True이면 NNCF 학습 후 양자화를 실행합니다
            data=None,        # int8=True일 때 필요합니다
            output_path=None, # None이면 weights/<stem>_openvino을 작성합니다
        )
  int8:
    - label: 보정 데이터를 사용하는 INT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # 필수이며 이 형식에는 기본값이 없습니다
            fraction=1.0,
        )
  run:
    - label: LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 디바이스 선택
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto"와 "cpu"는 CPU로, "gpu"와 "cuda"는 GPU로 매핑되며,
        # 그 외 값은 대문자로 변환해 전달됩니다. 예: "npu" -> NPU.
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: OpenVINO 직접 사용
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # 클래스 이름, 작업, 입력 크기는 IR 옆의 metadata.yaml에 있습니다.

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # 이 경로에서는 전처리와 후처리를 직접 담당합니다.
  support:
    - label: 내보내기 전 계열 및 작업 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## 설치

<code-tabs name="install" />

변환은 ONNX 중간 형식을 거치므로 `onnx` extra는 선택적 보조 요소가 아니라 요구
사항의 일부입니다. NNCF는 별도로 설치하며 `int8=True`에만 필요합니다.

## 내보내기

<code-tabs name="export" />

아티팩트는 파일이 아니라 디렉터리입니다. `weights/LibreYOLO9t_openvino`에는
`model.xml`, `model.bin`, `metadata.yaml`이 있으며, `half=True`이면 접미사 앞에
`_fp16`이 삽입됩니다. 전체 디렉터리를 이동하거나 복사해야 합니다. 세 파일이 하나의
아티팩트입니다.

`half=True`는 저장할 때 `compress_to_fp16`을 설정합니다. 이는 IR의 가중치
압축이며 런타임에 디바이스가 선택하는 추론 정밀도를 바꾸지는 않습니다.

### INT8

<code-tabs name="int8" />

`int8=True`는 혼합 프리셋을 사용하는 LibreYOLO 보정 로더에서 NNCF 학습 후
양자화를 실행하며 `data`가 필수입니다. 이 형식에는 8개 이미지 대체값이 없습니다.
NNCF가 없으면 설치 명령을 명시한 `ImportError`가 발생합니다.

## 아티팩트 실행

<code-tabs name="run" />

`LibreYOLO()`는 `model.xml`이 포함된 모든 디렉터리를 인식하고 체크포인트와 동일한
`Results` 객체를 반환하며, 클래스 이름, 작업, 입력 크기, 자세 스키마를
`metadata.yaml`에서 읽습니다.

디바이스 문자열은 그대로 전달되지 않고 매핑됩니다. `auto`와 `cpu`는 모두 CPU용으로
컴파일하고, `gpu`와 `cuda`는 모두 GPU용으로 컴파일하며, 그 외 값은 대문자로
변환하여 OpenVINO에 전달합니다. 이 방식으로 NPU 대상을 지정할 수 있습니다.

세 번째 스니펫은 LibreYOLO가 설치되지 않은 사용자를 위한 것입니다. 이 경로에서는
전처리, 디코딩, NMS, 좌표 크기 조정을 직접 담당하며 클래스 이름은
`metadata.yaml`에만 있습니다.

## 제약 조건

`metadata.yaml`이 없는 IR도 불러오지만 백엔드는 클래스 80개와 탐지 작업으로
대체합니다. 다른 모든 경우에는 잘못된 값이므로 디렉터리를 온전히 유지합니다.

추적 전에 차단되는 조합은 YOLO9 분할, RTMDet-Ins 분할, SSD, Faster R-CNN,
RetinaNet 탐지, 그리고 OpenVINO 2026.2가 공유 매트 디코더의 표준 ONNX
`DeformConv-19` 연산을 낮출 수 없는 BiRefNet 또는 FeyNobg 매팅입니다.

검증됨 또는 차단됨으로 표시되지 않은 조합에서는 변환기 경로를 사용할 수 있지만
프로젝트에 OpenVINO 런타임 동등성 기록이 없습니다. 여러 조합에는 명시적인 검증
맥락이 첨부됩니다. 예를 들어 OpenVINO 2026.2의 CPU 기본 추론 정밀도에서 고정
520 x 520 입력을 사용하는 DeepLabV3 시맨틱 분할과 고정 448 x 448 얼굴 크롭을
사용하는 L2CS 시선 추정이 있습니다. `libreyolo formats`는 조합별로 이 맥락을
출력합니다.

전체 계열 및 작업 표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참조합니다. 조합 하나를 확인하려면 다음을 실행합니다.

<code-tabs name="support" />
