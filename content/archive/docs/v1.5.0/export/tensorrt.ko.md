---
title: TensorRT
seo_title: LibreYOLO에서 TensorRT로 내보내기
description: >-
  LibreYOLO 모델에서 TensorRT 엔진 구축: ONNX 중간 파일, FP16 및 INT8 빌드, 동적 배치 프로필, 엔진 이식성
  한계.
lead: >-
  TensorRT는 그래프를 하나의 GPU에 맞게 조정된 엔진으로 컴파일합니다. LibreYOLO는 먼저 ONNX 중간 파일을 내보내고,
  TensorRT의 ONNX 파서를 사용해 이를 파싱한 뒤, 엔진을 구축하고 모델 메타데이터를 JSON 사이드카로 그 옆에 기록합니다.
keywords:
  - yolo 텐서RT 내보내기
  - 텐서RT 엔진
  - trt fp16
  - 텐서RT INT8 보정
  - 최적화 프로필
  - 동적 배치 텐서RT
  - 하드웨어 호환성 수준
last_verified: 1.5.0
meta:
  - label: 깃발
    value: export(format="tensorrt")
    mono: true
  - label: 쓴다
    value: 하나의 .engine 파일과 .engine.json 메타데이터 사이드카
  - label: 추가
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: 다시 로드
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: 모양
    value: Static by default; dynamic=True adds a batch-axis optimization profile
  - label: 정확성
    value: 'FP32, FP16 (half=True), INT8 (int8=True with data=)'
  - label: 필요하다
    value: 빌드 시점과 실행 시점에 NVIDIA GPU. 엔진은 GPU 아키텍처 간에 이동하지 않습니다.
verification: >-
  개발 브랜치에서 libreyolo/export/tensorrt.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tensorrt.py 및 pyproject.toml를
  읽으십시오.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        # 엔진은 ONNX 중간 단계에서 생성되므로 두 가지 추가 항목이 모두 필요합니다.
        pip install "libreyolo[onnx,tensorrt]"
    - label: 빌드하기 전에 툴체인을 확인하십시오
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


        # weights/LibreYOLO9t_fp16.engine와 weights/LibreYOLO9t_fp16.engine.json를
        씁니다

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: 논쟁
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # int8=True일 때 필요
            dynamic=False,
            workspace=4.0,                  # 빌드 시간 스크래치의 GiB
            min_batch=1,                    # 동적 프로필 경계
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # 또는 "ampere_plus"
            gpu_device=0,                   # 멀티 GPU 호스트에서 장치 빌드
            verbose=False,
        )
  dynamic:
    - label: 동적 배치 엔진
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ONNX 중간 단계는 프로파일을 위해 동적 배치 축이 필요합니다
        # 묶을 무언가를 가지기 위하여.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: 보정 데이터가 있는 INT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # 필수: 이 형식에는 기본값이 없습니다
            fraction=1.0,
        )
  run:
    - label: LibreYOLO를 통해
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 베어 텐서RT
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

        # 클래스 이름, 작업 및 입력 크기는 엔진이 아니라 사이드카에 있습니다.
        # 버퍼 할당, 전처리 및 후처리는 여기서 직접 처리해야 함.
        print(json.load(open(path + ".json"))["names"])
  support:
    - label: 빌드하기 전에 하나의 계열와 작업을 확인하십시오
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## 설치

빌드와 실행 모두 작동하는 CUDA 스택이 있는 NVIDIA GPU가 필요합니다. 이 형식에는 CPU 대체 기능이 없습니다.

<code-tabs name="install" />

`tensorrt` 추가 핀 `tensorrt-cu12` 및 `pycuda`와 마커 드롭은 macOS에서 모두 작동합니다. Jetson에서는 해당 추가 핀을 사용하지 마십시오: 이는 CUDA 13 플랫폼에 대해 CUDA 12 빌드를 고정시키기 때문입니다. 대신 [NVIDIA Jetson](/docs/export/jetson)에 설명된 대로 JetPack이 설치하는 TensorRT를 사용하십시오.

## 내보내기

<code-tabs name="export" />

내보내기는 두 단계로 실행됩니다. 첫 번째 단계에서는 ONNX 중간 파일을 임시 경로에 작성하고, 두 번째 단계에서는 이를 파싱하고 엔진을 구축하며, 이후 중간 파일은 제거됩니다. `workspace`는 GiB 단위의 빌드 시 임시 메모리이며, 값이 크면 빌더가 더 많은 커널을 시도할 수 있지만 추론 메모리에는 영향을 주지 않습니다.

메타데이터 사이드카는 엔진 옆에 `<engine>.json`로 작성되며 빌드가 실제로 달성한 정밀도를 기록합니다. GPU에 빠른 FP16이나 빠른 INT8이 없으면 빌더는 경고를 표시하고 대체하며, 사이드카는 요청된 정밀도 대신 실제로 생성된 정밀도를 보고합니다.

FP16에서는 그래프에서 ViT 백본이 탐지되고 그 float 레이어가 FP32로 고정됩니다. DINOv2 스타일 백본은 FP16에서 오버플로우를 일으켜 NaN을 생성하므로, 빌드는 `OBEY_PRECISION_CONSTRAINTS`를 설정하고 `FP16 (FP32 ViT backbone)`를 보고합니다. 이 패스는 CNN 백본에서는 아무 작업도 수행하지 않습니다.

### 동적 배치

<code-tabs name="dynamic" />

`dynamic=True`는 `min_batch`에서 `max_batch`까지 걸친 하나의 최적화 프로파일을 추가하며, `opt_batch`에서 최적화하고 해당 세 값을 사이드카에 기록합니다. 프로파일은 ONNX 중간 결과가 실제로 동적 배치 차원을 포함할 때만 추가되며, 그렇지 않으면 빌드 로그에 정적 최적화를 사용하고 있음을 기록하고 계속 진행합니다.

### INT8

<code-tabs name="int8" />

INT8은 LibreYOLO 보정 로더 대신 TensorRT의 엔트로피 보정기를 사용하며, `data`는 필수입니다: 이 형식에는 8장 이미지 대체가 없습니다. 보정에는 장치 버퍼를 위해 `cuda-python` 또는 `pycuda`가 필요합니다. 보정 캐시는 ONNX 바이트의 해시를 키로 사용하므로, 한 모델의 스케일은 동일한 출력 경로에 쓰는 다른 모델에서 재사용되지 않습니다.

`half=True`와 `int8=True`는 함께 INT8을 경고하고 구축하며, TensorRT가 양자화할 수 없는 레이어에 대해 FP16 폴백을 유지합니다.

## 그 유물을 실행

<code-tabs name="run" />

`LibreYOLO()`는 `.engine` 접미사에서 디스패치를 수행하고, 클래스 이름, 태스크 및 포즈 스키마를 위해 사이드카를 읽으며, 체크포인트와 동일한 `Results` 객체를 반환합니다. CUDA 장치가 없으면 즉시 오류를 발생시킵니다.

두 번째 스니펫은 베어 런타임 경로입니다. 호스트 및 장치 버퍼 할당, 전처리, 디코딩, NMS 및 좌표 재조정 모두 사용자의 책임이 되며, 엔진 자체에는 클래스 이름이 없으므로 사이드카가 함께 이동해야 합니다.

## 제약

직렬화된 엔진은 GPU 아키텍처, 드라이버 스택 및 엔진을 빌드한 TensorRT 버전에 연결되어 있습니다. 워크스테이션에서 빌드된 엔진은 다른 아키텍처에서는 로드되지 않기 때문에 빌드 단계가 배포 머신에서 실행됩니다. `hardware_compatibility="ampere_plus"`는 성능의 일부를 포기하는 대신 Ampere 이후 모델 간의 이식성을 제공합니다. `"same_compute_capability"` 값은 `NONE`에 매핑되며, 엔진이 현재 GPU에만 최적화되어 있음을 경고하고, 내보내기 시 적용되지 않은 이식성을 주장하지 않습니다.

배치 축만 프로파일링됩니다. 동적 공간 차원을 가진 빌드는 이 계약에 포함되지 않기 때문에 FCOS가 차단됩니다: FCOS는 800x1333 비율 변환을 유지하기 위해 동적 패딩된 높이와 너비가 필요합니다.

추적 전에 차단됨: YOLO9 분할, RTMDet-Ins 분할, SSD, Faster R-CNN 및 RetinaNet 검출, 그리고 BiRefNet 또는 FeyNobg 매팅, 여기서 TensorRT 10.16이 공유된 ONNX `DeformConv` 노드에 도달하지만 `ModulatedDeformConv2d`가 플러그인 레지스트리에 없어 이를 파싱할 수 없습니다.

조합이 검증되지도 차단되지도 않은 경우, 변환기 경로는 사용 가능하며 프로젝트에서 해당 조합에 대한 TensorRT 런타임 일치를 기록하지 않았습니다. 이는 빌드가 성공하는지 여부에 대한 것이 아니라 증거에 대한 진술입니다.

전체 계열 및 작업 그리드는 [내보내기 매트릭스](/docs/reference/export-matrix)를 참조하십시오. 한 가지 조합의 경우:

<code-tabs name="support" />
