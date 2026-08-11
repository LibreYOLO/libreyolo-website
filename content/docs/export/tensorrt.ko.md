---
title: TensorRT
seo_title: LibreYOLO에서 TensorRT로 내보내기
description: >-
  LibreYOLO 모델에서 TensorRT 엔진을 빌드합니다. ONNX 중간 형식, FP16 및 INT8 빌드, 동적 배치 프로필, 엔진
  이식성 제한을 설명합니다.
lead: >-
  TensorRT는 그래프를 특정 GPU에 맞게 튜닝된 엔진으로 컴파일합니다. LibreYOLO는 먼저 ONNX 중간 형식을 내보내고
  TensorRT의 ONNX 파서로 파싱하여 엔진을 빌드한 다음 모델 메타데이터를 JSON 사이드카로 옆에 작성합니다.
keywords:
  - yolo tensorrt 변환
  - tensorrt 엔진 빌드
  - trt fp16
  - tensorrt int8 보정
  - 최적화 프로필
  - tensorrt 동적 배치
  - 하드웨어 호환성 수준
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="tensorrt")
    mono: true
  - label: 출력
    value: .engine 파일 하나와 .engine.json 메타데이터 사이드카
  - label: 추가 설치
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: 다시 불러오기
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: 형상
    value: 기본값은 정적이며 dynamic=True이면 배치 축 최적화 프로필을 추가합니다.
  - label: 정밀도
    value: 'FP32, FP16(half=True), INT8(int8=True와 data=)을 지원합니다.'
  - label: 요구 사항
    value: 빌드 및 실행 시 NVIDIA GPU가 필요합니다. 엔진은 서로 다른 GPU 아키텍처 간에 이동하지 않습니다.
verification: >-
  dev 브랜치의 libreyolo/export/tensorrt.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tensorrt.py, pyproject.toml을
  확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        # 엔진은 ONNX 중간 형식에서 빌드되므로 두 extra가 모두 필요합니다.
        pip install "libreyolo[onnx,tensorrt]"
    - label: 빌드 전 도구 체인 확인
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # weights/LibreYOLO9t_fp16.engine과 weights/LibreYOLO9t_fp16.engine.json을
        작성합니다

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: 인수
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # int8=True일 때 필요합니다
            dynamic=False,
            workspace=4.0,                  # 빌드 시 스크래치 메모리(GiB)
            min_batch=1,                    # 동적 프로필 범위
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # 또는 "ampere_plus"
            gpu_device=0,                   # 다중 GPU 호스트의 빌드 디바이스
            verbose=False,
        )
  dynamic:
    - label: 동적 배치 엔진
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 프로필이 바인딩할 수 있도록 ONNX 중간 형식에 동적 배치 축이 필요합니다.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: 보정 데이터를 사용하는 INT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # 필수이며 이 형식에는 기본값이 없습니다
            fraction=1.0,
        )
  run:
    - label: LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: TensorRT 직접 사용
      language: python
      code: |
        import json

        import tensorrt as trt

        path = "weights/LibreYOLO9t_fp16.engine"
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # 클래스 이름, 작업, 입력 크기는 엔진이 아니라 사이드카에 있습니다.
        # 이 경로에서는 버퍼 할당, 전처리, 후처리를 직접 담당합니다.
        print(json.load(open(path + ".json"))["names"])
  support:
    - label: 빌드 전 계열 및 작업 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## 설치

빌드와 실행 모두 정상 작동하는 CUDA 스택이 있는 NVIDIA GPU가 필요합니다. 이
형식에는 CPU 대체 경로가 없습니다.

<code-tabs name="install" />

`tensorrt` extra는 `tensorrt-cu12`와 `pycuda`를 고정하며 마커는 macOS에서 둘을
모두 제외합니다. Jetson에서는 이 extra를 사용하지 마십시오. CUDA 13 플랫폼에
CUDA 12 빌드를 고정합니다. 대신 [NVIDIA Jetson](/docs/export/jetson)에 설명된 대로
JetPack이 설치하는 TensorRT를 사용합니다.

## 내보내기

<code-tabs name="export" />

내보내기는 두 단계로 실행됩니다. 첫 번째 단계는 임시 경로에 ONNX 중간 형식을 쓰고,
두 번째 단계는 이를 파싱하여 엔진을 빌드하며 이후 중간 형식을 제거합니다.
`workspace`는 GiB 단위의 빌드 시 스크래치 메모리입니다. 값이 클수록 빌더가 더 많은
커널을 시도할 수 있으며 추론 메모리에는 영향을 주지 않습니다.

메타데이터 사이드카는 엔진 옆에 `<engine>.json`으로 작성되며 빌드가 실제로 구현한
정밀도를 기록합니다. GPU에 빠른 FP16 또는 INT8이 없으면 빌더가 경고하고 대체
정밀도를 사용하며, 사이드카는 요청한 정밀도가 아니라 생성된 정밀도를 보고합니다.

FP16에서는 그래프의 ViT 백본을 탐지하고 해당 float 레이어를 FP32로 고정합니다.
DINOv2 방식 백본은 FP16에서 오버플로되어 NaN을 생성하므로 빌드는
`OBEY_PRECISION_CONSTRAINTS`를 설정하고 `FP16 (FP32 ViT backbone)`을
보고합니다. CNN 백본에서는 이 패스가 아무 작업도 하지 않습니다.

### 동적 배치

<code-tabs name="dynamic" />

`dynamic=True`는 `min_batch`부터 `max_batch`까지 범위에서 `opt_batch`에 최적화된
하나의 최적화 프로필을 추가하고 이 3개 값을 사이드카에 기록합니다. 프로필은 ONNX
중간 형식에 실제 동적 배치 차원이 있을 때만 추가됩니다. 없으면 빌드가 정적 최적화를
사용한다고 기록하고 계속합니다.

### INT8

<code-tabs name="int8" />

INT8은 LibreYOLO 보정 로더에서 TensorRT 엔트로피 보정기를 사용하며 `data`가
필수입니다. 이 형식에는 8개 이미지 대체값이 없습니다. 보정에는 디바이스 버퍼용
`cuda-python` 또는 `pycuda`가 필요합니다. 보정 캐시는 ONNX 바이트 해시를 키로
사용하므로 같은 출력 경로에 작성된 다른 모델에 한 모델의 스케일을 재사용하지
않습니다.

`half=True`와 `int8=True`를 함께 사용하면 경고한 뒤 INT8을 빌드하며, TensorRT가
양자화할 수 없는 레이어에는 FP16 대체 정밀도를 유지합니다.

## 아티팩트 실행

<code-tabs name="run" />

`LibreYOLO()`는 `.engine` 접미사에 따라 디스패치하고 사이드카에서 클래스 이름,
작업, 자세 스키마를 읽은 뒤 체크포인트와 동일한 `Results` 객체를 반환합니다. CUDA
디바이스가 없으면 즉시 오류가 발생합니다.

두 번째 스니펫은 런타임을 직접 사용하는 경로입니다. 이 경로에서는 호스트 및
디바이스 버퍼 할당, 전처리, 디코딩, NMS, 좌표 크기 조정을 모두 직접 담당합니다.
엔진 자체에는 클래스 이름이 없으므로 사이드카를 함께 이동해야 합니다.

## 제약 조건

직렬화된 엔진은 빌드에 사용한 GPU 아키텍처, 드라이버 스택, TensorRT 버전에
종속됩니다. 워크스테이션에서 빌드한 엔진은 다른 아키텍처에서 불러오지 못하므로
배포 머신에서 빌드 단계를 실행합니다. `hardware_compatibility="ampere_plus"`는
일부 성능을 절충하여 Ampere 및 이후 세대 간 이식성을 제공합니다.
`"same_compute_capability"` 값은 `NONE`으로 매핑되고 경고합니다. 엔진은 현재
GPU에만 최적화되며, 내보내기는 적용하지 않은 이식성을 주장하지 않고 이를
명시합니다.

프로필은 배치 축에만 적용됩니다. 동적 공간 차원 빌드는 이 계약에 포함되지 않으므로
FCOS가 차단됩니다. FCOS에는 800 x 1333 종횡비 변환을 유지하기 위한 동적 패딩 높이와
너비가 필요합니다.

추적 전에 차단되는 조합은 YOLO9 분할, RTMDet-Ins 분할, SSD, Faster R-CNN,
RetinaNet 탐지, 그리고 TensorRT 10.16이 공유 ONNX `DeformConv` 노드에 도달하지만
플러그인 레지스트리에 `ModulatedDeformConv2d`가 없어 파싱할 수 없는 BiRefNet 또는
FeyNobg 매팅입니다.

검증됨 또는 차단됨으로 표시되지 않은 조합에서는 변환기 경로를 사용할 수 있지만
프로젝트에 TensorRT 런타임 동등성 기록이 없습니다. 이는 빌드 성공 여부가 아니라
근거에 관한 설명입니다.

전체 계열 및 작업 표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참조합니다. 조합 하나를 확인하려면 다음을 실행합니다.

<code-tabs name="support" />
