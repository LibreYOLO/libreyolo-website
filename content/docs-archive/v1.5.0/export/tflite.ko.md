---
title: TFLite
seo_title: LibreYOLO에서 TFLite(LiteRT)로 내보내기
description: >-
  onnx2tf를 거쳐 LibreYOLO 모델을 .tflite FlatBuffer로 내보냅니다. 정적 형상, FP32 전용, NHWC 입력,
  정상적으로 변환되는 계열을 설명합니다.
lead: >-
  TFLite는 LiteRT가 모바일 및 임베디드 대상에서 실행하는 FlatBuffer 형식입니다. LibreYOLO는 정적 ONNX 그래프를
  내보내고 flatbuffer-direct 모드의 onnx2tf로 변환하며, 모델 메타데이터를 JSON 사이드카로 아티팩트 옆에 작성합니다.
keywords:
  - yolo tflite 변환
  - litert 사용법
  - onnx2tf 변환
  - ai-edge-litert
  - tflite flatbuffer
  - tflite nhwc 입력
  - 엣지 추론
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="tflite")
    mono: true
  - label: 출력
    value: .tflite 파일 하나와 .tflite.json 메타데이터 사이드카
  - label: 추가 설치
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: 다시 불러오기
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: 형상
    value: 정적 형상만 지원합니다. dynamic=True는 거부됩니다.
  - label: 정밀도
    value: FP32만 지원합니다. half=True와 int8=True는 거부됩니다.
  - label: 요구 사항
    value: onnx2tf 2.4.x가 이전 버전용 휠을 제공하지 않으므로 Python 3.12 이상이 필요합니다.
verification: >-
  dev 브랜치의 libreyolo/export/tflite.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tflite.py, pyproject.toml을
  확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        # LiteRT는 TensorFlow Lite의 현재 Google 명칭입니다. 두 extra 모두
        # 동일한 도구 체인을 설치하고 같은 .tflite 출력을 생성합니다.
        pip install "libreyolo[tflite]"
    - label: Python 버전 먼저 확인
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.tflite와 weights/LibreYOLO9t.tflite.json을 작성합니다
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert"는 별칭으로 허용되며 동일한 내보내기로 해석됩니다.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: 인수
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int 또는 (height, width)
            batch=1,
            simplify=True,    # ONNX 중간 형식에 onnxsim을 적용합니다
            output_path=None, # None이면 weights/<stem>.tflite를 작성합니다
            verbose=False,    # True이면 onnx2tf 로그를 스트리밍합니다
        )

        # dynamic=True는 ValueError를 일으킵니다. 변환기에는 정적 형상이 필요합니다.
        # half=True와 int8=True는 추적 전에 거부됩니다.
  run:
    - label: LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: LiteRT 직접 사용
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NCHW가 아니라 NHWC입니다


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # 클래스 이름, 작업, 입력 크기는 사이드카에 있습니다.

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # 전처리, NCHW-NHWC 전치, 후처리를 직접 담당합니다.
  support:
    - label: 내보내기 전 계열 및 작업 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## 설치

<code-tabs name="install" />

extra는 변환용 `onnx2tf`와 결과 실행용 `ai-edge-litert`를 가져오며 둘 다 Python
3.12 마커가 적용됩니다. 이전 인터프리터에서 내보내기를 시도하면 변환기 내부에서
실패하는 대신 버전 요구 사항을 명시한 `ImportError`가 발생합니다.

`libreyolo[litert]`도 정확히 같은 항목을 설치합니다. 형식 문자열 `litert`는
`tflite`의 별칭이며 어느 쪽이든 출력 파일은 `.tflite`입니다.

## 내보내기

<code-tabs name="export" />

다른 작업보다 먼저 계열과 작업을 확인하므로 지원되지 않는 조합은 일반적인
메시지가 아니라 제외 원인이 된 구체적인 변환기 또는 런타임 오류와 함께 즉시
실패합니다. 변환 자체는 정적 ONNX 중간 형식을 `flatbuffer_direct` 모드의
`onnx2tf`로 처리하는 서브프로세스 호출입니다.

메타데이터는 사이드카입니다. `weights/LibreYOLO9t.tflite.json`에는 계열, 작업,
클래스 이름, 입력 크기, 자세 스키마가 들어 있습니다. FlatBuffer 자체에는
LibreYOLO 메타데이터 필드가 없으므로 두 파일을 함께 이동해야 합니다.

## 아티팩트 실행

<code-tabs name="run" />

`LibreYOLO()`는 `.tflite` 접미사에 따라 디스패치하고 체크포인트와 동일한
`Results` 객체를 반환합니다. 백엔드는 사이드카를 읽고, 인터프리터가 채널 마지막
입력을 요구하면 NCHW 블롭을 NHWC로 전치하고, 있는 경우 인터프리터의 양자화
스케일과 영점을 적용한 다음 출력을 LibreYOLO 후처리가 예상하는 레이아웃으로 다시
전치합니다.

두 번째 스니펫은 런타임을 직접 사용하는 경로입니다. 이 경로에서는 전처리,
레이아웃 전치, 디코딩, NMS, 좌표 크기 조정을 모두 직접 담당합니다. 가장 놓치기
쉬운 부분은 레이아웃입니다. onnx2tf는 채널 마지막 입력을 생성하므로
`(1, 3, 640, 640)` 형상의 블롭은 바인딩되지 않습니다.

## 제약 조건

정적 형상만 지원합니다. `dynamic=True`는 추적 전에 `ValueError`를 일으키며,
내보내기 캔버스는 `imgsz`가 해석된 값으로 고정됩니다.

FP32만 지원합니다. `half=True`와 `int8=True`는 모두 검증 중 거부되므로 현재 이
내보내기에서는 양자화 배포를 사용할 수 없습니다.

여기서 적용 범위는 그래프 형식보다 좁으며 계열이 아니라 측정으로 결정됩니다.
검증된 조합에는 YOLO9, YOLOX, YOLO-NAS 탐지, PIDNet 시맨틱 분할, CNN 분류기
4개 계열, DINOv2 및 SigLIP2 임베딩, SigLIP2 분류, TEED 및 DexiNed 엣지 검출,
Real-ESRGAN 및 SwinIR 복원이 포함됩니다. SwinIR에는 추가 주의 사항이 있습니다.
소스 크기가 내보내기 캔버스와 정확히 일치할 때 동등성이 유지되며, 더 작은 소스는
트랜스포머 실행 전에 캔버스 크기로 패딩되어 네이티브 가변 크기 추론과 달라질 수
있습니다.

차단된 항목에는 정확한 실패 원인이 표시되므로 해결 방법을 시도하기 전에 읽어볼
가치가 있습니다. 몇 가지 예를 들면 다음과 같습니다. RF-DETR 탐지는 네이티브
384 캔버스에서 변환되지만 `STRIDED_SLICE`가 지원되는 5-D 차원을 초과한 입력을
받아 LiteRT가 할당할 수 없습니다. PicoDet은 `RESHAPE`가 19,200개 입력 요소를
9,600개 출력 요소로 매핑하므로 거부됩니다. D-FINE은 `GatherElements` 형상 처리
중 변환기가 중단됩니다. RTMDet은 원시 동등성이 온전한 상태로 내보내고 다시 불러올
수 있지만 공개 바운딩 박스는 29.9 px 좌표 편차와 함께 IoU 0.911로 떨어집니다.

전체 계열 및 작업 표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참조합니다. 차단 사유 문자열을 포함해 조합 하나를 확인하려면 다음을 실행합니다.

<code-tabs name="support" />
