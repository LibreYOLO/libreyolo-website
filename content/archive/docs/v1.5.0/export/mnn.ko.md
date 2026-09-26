---
title: MNN
seo_title: LibreYOLO에서 MNN으로 내보내기
description: >-
  ONNX와 mnnconvert를 거쳐 LibreYOLO 탐지기를 MNN으로 내보냅니다. 고정 NCHW 형상, CPU의 FP32, 런타임
  계약에 필요한 메타데이터 사이드카를 설명합니다.
lead: >-
  MNN은 Alibaba의 경량 추론 엔진입니다. LibreYOLO는 정적 ONNX 그래프를 내보내고, MNN 패키지에 포함된
  mnnconvert 도구로 변환한 다음, 입출력 이름, 고정 입력 형상, 클래스 이름을 기록하는 JSON 사이드카를 작성합니다.
keywords:
  - yolo mnn 변환
  - mnnconvert 사용법
  - mnn 추론
  - 모바일 객체 탐지 추론
  - 고정 nchw 형상
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="mnn")
    mono: true
  - label: 출력
    value: .mnn 파일 하나와 .mnn.json 메타데이터 사이드카
  - label: 추가 설치
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: 다시 불러오기
    value: LibreYOLO("weights/LibreYOLO9t.mnn")
    mono: true
  - label: 형상
    value: 고정 NCHW입니다. dynamic=True는 거부됩니다.
  - label: 정밀도
    value: FP32 및 CPU만 지원합니다.
  - label: 작업
    value: 이 버전에서는 탐지만 지원합니다.
verification: >-
  dev 브랜치의 libreyolo/export/mnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/mnn.py, pyproject.toml을
  확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        # extra에는 libreyolo[onnx]가 포함됩니다. MNN은 ONNX 중간 형식에서 변환합니다.
        pip install "libreyolo[mnn]"
    - label: 변환기가 경로에 있는지 확인
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.mnn과 weights/LibreYOLO9t.mnn.json을 작성합니다
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: 인수
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int 또는 (height, width)
            batch=1,          # 아티팩트에 고정됩니다
            simplify=True,    # ONNX 중간 형식에 onnxsim을 적용합니다
            output_path=None, # None이면 weights/<stem>.mnn을 작성합니다
            verbose=False,    # True이면 mnnconvert 로그를 스트리밍합니다
        )

        # dynamic=True는 ValueError를 일으킵니다. half=True와 int8=True는 거부됩니다.
  run:
    - label: LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: MNN 직접 사용
      language: python
      code: >
        import json


        import MNN

        import numpy as np


        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))

        print(meta["mnn_input_names"], meta["mnn_output_names"],
        meta["mnn_input_shape"])


        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )

        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )


        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)

        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )

        outputs = module.forward([input_var])

        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # 이 경로에서는 전처리와 후처리를 직접 담당합니다.
  support:
    - label: 내보내기 전 계열 및 작업 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 68fad34d07aea149
---

## 설치

<code-tabs name="install" />

변환이 ONNX 중간 형식을 거치므로 extra에는 `libreyolo[onnx]`가 포함됩니다.
내보내기는 먼저 활성 Python 인터프리터 옆에서, 그다음 `PATH`에서 함께 설치되는
`mnnconvert` 실행 파일을 찾습니다. 변환기가 없으면 변환 도중 실패하는 대신 설치
명령을 명시한 `ImportError`가 발생합니다.

## 내보내기

<code-tabs name="export" />

내보내기는 그래프를 넘기기 전에 ONNX 입력 계약을 읽고 표현할 수 없는 입력을
거부합니다. 이미지 입력이 둘 이상이거나 입력 형상에 기호 차원이 있으면
거부됩니다. 이 버전의 MNN에는 완전히 고정된 NCHW 형상이 필요하며, `batch`는
불러올 때 협상되지 않고 아티팩트에 고정됩니다.

사이드카는 선택적인 기록이 아닙니다. `weights/LibreYOLO9t.mnn.json`은 입출력
이름, 고정 입력 형상, 배치, 클래스 이름, 사용한 MNN 버전, 아티팩트가 빌드된
백엔드를 기록하며, 런타임은 불러올 때 이러한 필드를 모두 검증합니다.

Windows에서 MNN 3.6.1은 때때로 변환을 완료한 뒤 프로세스 종료 중 액세스 위반
또는 빠른 실패 상태로 끝납니다. 내보내기는 이러한 특정 종료 코드를 인식하며,
출력 파일이 있으면 변환에 성공한 것으로 처리합니다.

## 아티팩트 실행

<code-tabs name="run" />

`LibreYOLO()`는 `.mnn` 접미사에 따라 디스패치하고 체크포인트와 동일한 `Results`
객체를 반환합니다. 불러오기는 의도적으로 엄격합니다. 사이드카는 `format=mnn`,
`mnn_backend=cpu`, `dynamic=false`, `precision=fp32`, 크기, 탐지 작업, 기록된
이미지 크기와 일치하는 고정 양의 NCHW 형상, `0`부터 `nc - 1`까지 모든 인덱스를
포괄하는 클래스 이름을 선언해야 합니다. 일치하지 않으면 추측하지 않고 오류를
발생시킵니다.

아티팩트를 빌드할 때와 다른 `imgsz`로 예측해도 오류가 발생하며, 여기서 MNN
내보내기는 CPU에서 실행되므로 `device`는 경고와 함께 무시됩니다.

두 번째 스니펫은 런타임을 직접 사용하는 경로입니다. 이 경로에서는 전처리,
디코딩, NMS, 좌표 크기 조정을 직접 담당하며, MNN 모듈 로더가 명시적인 이름을
요구하므로 입출력 이름은 사이드카에서 가져옵니다.

## 제약 조건

탐지만 지원합니다. 백엔드는 불러올 때 다른 모든 작업을 거부하며 내보내기 측도
동일합니다. 기록된 조합을 벗어나면 사전 검사가 "MNN v1 has no implemented
runtime contract for this family and task."라는 메시지와 함께 오류를 발생시킵니다.

FP32, CPU, 고정 형상만 지원합니다. `dynamic=True`는 `ValueError`를 일으키며,
`half=True`와 `int8=True`는 검증 중 거부됩니다.

검증된 탐지 계열은 YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC, RT-DETR,
RT-DETRv2, RT-DETRv4, D-FINE, DEIM, YOLO-NAS입니다. 각 계열은 변환, 새 아티팩트
다시 불러오기, MNN CPU 실행, 메타데이터 검사, PyTorch 모델 대비 NMS 이후 탐지
동등성 일치를 거쳤습니다. DEIMv2는 변환, 다시 불러오기, 실행이 가능하고 NMS
이후 탐지를 보존하지만, 중간 ONNX 경로의 쿼리 수준 점수 동등성이 불완전하므로
검증됨이 아니라 사용 가능으로 기록됩니다.

전체 계열 및 작업 표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참조합니다. 조합 하나를 확인하려면 다음을 실행합니다.

<code-tabs name="support" />
