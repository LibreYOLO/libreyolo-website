---
title: ONNX
seo_title: LibreYOLO에서 ONNX로 내보내기
description: >-
  LibreYOLO 모델을 ONNX로 내보냅니다. LibreYOLO가 계열별로 선택하는 opset, 동적 축, 내장 NMS, INT8,
  그래프를 다시 불러오는 방법을 설명합니다.
lead: >-
  ONNX는 이식 가능한 그래프 형식입니다. LibreYOLO는 torch.onnx.export로 모델을 추적하고 선택적으로 그래프를
  단순화하며, 모든 LibreYOLO 백엔드가 후처리를 다시 빌드할 수 있도록 계열, 작업, 클래스 이름, 입력 크기를 파일 자체 메타데이터에
  작성합니다.
keywords:
  - yolo onnx 변환
  - onnxruntime 사용법
  - torch.onnx.export
  - onnx opset
  - 동적 축
  - onnx nms 내장
  - onnx int8 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="onnx")
    mono: true
  - label: 출력
    value: 메타데이터가 그래프에 내장된 .onnx 파일 하나
  - label: 추가 설치
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: 다시 불러오기
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: 형상
    value: Python에서는 동적 배치가 기본값이며 작업별 예외는 아래에 설명합니다.
  - label: 정밀도
    value: 'FP32, FP16(half=True), INT8(int8=True, YOLO9 탐지)을 지원합니다.'
verification: >-
  dev 브랜치의 libreyolo/export/onnx.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/onnx.py,
  libreyolo/cli/commands/export.py를 확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.onnx를 작성합니다
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: 인수
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int 또는 (height, width)
            batch=1,
            dynamic=True,     # Python 기본값이며 CLI 기본값은 False입니다
            simplify=True,    # 그래프에 onnxsim을 실행합니다
            opset=None,       # None은 13, DETR 계열에서는 17을 선택합니다
            half=False,       # FP16 가중치와 활성화
            int8=False,       # QDQ INT8, YOLO9 탐지 전용
            data=None,        # 보정 data.yaml, INT8 전용
            device=None,      # 추적 디바이스이며 None은 모델 디바이스를 사용합니다
            output_path=None, # None이면 weights/<stem>.onnx를 작성합니다
        )
  nms:
    - label: 그래프에 NMS 내장
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLO9 탐지 전용, 배치 1입니다. dynamic은 False로 강제됩니다.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: 보정 데이터를 사용하는 INT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # 대표 이미지 수백 장
            fraction=1.0,
        )
  run:
    - label: LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ONNX Runtime 직접 사용
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # 이 경로에서는 전처리와 후처리를 직접 담당합니다.

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # 그래프에는 계열, 작업, 클래스 이름, 입력 크기가 들어 있습니다.

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: 내보내기 전 계열 및 작업 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## 설치

<code-tabs name="install" />

extra는 `onnx`, `onnxsim`, `onnxruntime`을 가져옵니다. 파일을 쓰는 데는 `onnx`만
필요하며 `onnxsim`은 단순화 패스를 실행하고 `onnxruntime`은 아티팩트 실행과 INT8
보정을 수행합니다.

## 내보내기

<code-tabs name="export" />

`output_path`를 지정하지 않으면 체크포인트 이름으로 `weights/`에 파일을 작성하며,
해당 정밀도를 요청한 경우 `_fp16` 또는 `_int8`을 붙입니다.

`dynamic`의 기본값은 Python에서 `True`, CLI에서 `False`입니다. 활성화하면 배치 축이
기호로 바뀌고 일부 작업은 더 넓게 동적화됩니다. 시맨틱 분할은 마스크 높이와 너비도
열고, Real-ESRGAN 복원은 공간 축을 열며, 2단계 탐지기는 그래프 내부에서 크기를
조정하므로 소스 높이와 너비를 동적으로 유지합니다.

`opset`을 생략하면 계열별로 선택합니다. DETR 계열(`detr`, `deformable_detr`,
`dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `rfdetr`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`)과 `deit`, `midas`, `moge2`에는
`aten::scaled_dot_product`가 낮춰지는 opset 17을 사용합니다. 그 외 모든 계열에는
13을 사용합니다. BiRefNet 디코더에는 ONNX가 opset 19부터 정의하는 `DeformConv`
연산자가 필요하므로 매팅은 항상 19로 올립니다.

`simplify=True`는 `onnxsim`을 실행하며 패스가 실패하면 원본 그래프를 유지하므로
단순화 오류는 내보내기 실패가 아니라 경고입니다. macOS arm64에서 `onnx` 1.22 이상과
`onnxsim` 0.6.5 이하 조합은 Python 프로세스를 중단할 수 있으므로 패스를 완전히
건너뜁니다.

### 내장 NMS

<code-tabs name="nms" />

`nms=True`는 YOLO9 탐지와 배치 1만 지원합니다. `dynamic=True`와 함께 요청하면
경고를 기록하고 동적 형상을 끕니다. 그래프는 `(batch, max_det, 6)` 형상의
`output`과 LibreYOLO 자체 백엔드가 사용하는 디코딩되지 않은 탐지기 텐서 `raw`라는
출력 2개를 갖습니다. 따라서 후처리는 PyTorch 경로와 동일하게 유지됩니다.

### DeepStream

`deepstream=True`는 ONNX 전용 옵션입니다. NVIDIA DeepStream 파서가 예상하는
레이아웃으로 그래프를 내보내고 옆에 `config_infer_primary_<stem>.txt`와
`<stem>_labels.txt`라는 사이드카 2개를 작성하므로 직접 구성을 작성하지 않고
파이프라인에 넣을 수 있습니다.

`nms=True`와 함께 사용할 수 없으며 둘 다 요청하면 `ValueError`가 발생합니다.
DeepStream은 자체 클러스터링 단계에서 억제를 실행합니다. ONNX 외의 형식에 전달해도
오류가 발생합니다. 지원되는 계열 및 작업 표와 파서 빌드는
[DeepStream](/docs/export/deepstream)을 참조합니다.

### INT8

<code-tabs name="int8" />

`int8=True`는 ONNX Runtime 정적 양자화를 실행하고 float32 입출력을 사용하는 QDQ
그래프를 작성합니다. `Conv` 및 `Gemm` 노드만 양자화합니다. 탐지 헤드 디코딩을
float32로 유지하는 것은 의도적입니다. 이 연결은 픽셀 스케일 바운딩 박스 좌표와
0부터 1까지 클래스 점수를 혼합하므로 바운딩 박스 크기가 지배하는 단일 텐서별 활성화
스케일은 모든 점수를 0으로 만들 수 있습니다.

현재 이 플래그는 YOLO9 탐지에만 적용되며 그 외에는 사전 검사에서
`NotImplementedError`가 발생합니다. `data`를 생략하면 경고와 함께 `coco8.yaml`로
대체하지만 이미지 8장은 대표적인 보정 집합이 아닙니다. PyTorch에서 이미 양자화된
모델은 [양자화](/docs/export/quantization)에 설명된 다른 경로를 사용합니다.

## 아티팩트 실행

<code-tabs name="run" />

`LibreYOLO()`는 `.onnx` 접미사에 따라 디스패치하고 `.pt` 체크포인트와 동일한
`Results` 객체를 반환합니다. 내보낼 때 클래스 이름, 작업, 입력 크기, 자세 스키마를
그래프의 `metadata_props`에 작성했기 때문입니다. `device="auto"`를 사용하면 ONNX
Runtime이 보고하는 경우 `CUDAExecutionProvider`를 사용하고, 그 외에는 CPU로
대체합니다.

두 번째 스니펫은 LibreYOLO가 설치되지 않은 사용자를 위한 것입니다. 이 경로에서는
전처리, 디코딩, NMS, 좌표 크기 조정을 모두 직접 담당하며 메타데이터 블록은 여전히
읽을 수 있습니다.

## 제약 조건

출력 텐서 이름은 작업별로 고정되며 메타데이터가 없는 소비자가 맞춰야 하는
이름입니다.

| 작업 | 출력 이름 |
|---|---|
| 탐지, 그리드 및 앵커 헤드 | `output` |
| 탐지, DETR 계열 | `pred_logits`, `pred_boxes` |
| 탐지, RF-DETR | `dets`, `labels` |
| 분류 | `output` |
| 시맨틱 분할 | `semantic_logits` |
| 깊이 | `depth` |
| 표면 법선 | `normal` |
| 엣지 | `edges` |
| 복원 | `restored` |
| 매팅 | `matte` |
| 시선 | `yaw_logits`, `pitch_logits` |

RF-DETR은 입력 텐서 이름이 `images`가 아니라 `input`인 유일한 계열이기도 합니다.

이 버전에서 여러 작업은 고정 해상도 런타임 계약을 사용합니다. 깊이, 표면 법선,
엣지는 `batch != 1`을 거부하고 `dynamic=False`를 강제합니다. BiRefNet의 Swin 상대
위치 테이블은 해상도에 종속되므로 매팅은 네이티브 1024 정사각형을 강제합니다.
복원은 완전 컨볼루션 생성기인 Real-ESRGAN을 제외한 모든 계열에서 고정 캔버스를
강제합니다.

직사각형 `imgsz`는 YOLO9 계열, HRNet, NAFNet, Real-ESRGAN에서 작동합니다. 고정
정사각형 계약이 있는 계열(`clip`, `deformable_detr`, `detr`, `dinodetr`, `dfine`,
`deim`, `deimv2`, `ec`, `lwdetr`, `moge2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`,
`rfdetr`, `siglip2`, `ssd`)은 이를 즉시 거부합니다.

두 조합은 추적 전에 거부됩니다. LibreYOLO의 YOLO9은 탐지만 지원하므로 YOLO9 분할이
거부되며, 동적 커널 마스크 디코딩에 내보낸 런타임 계약이 없는 RTMDet-Ins 분할도
거부됩니다.

전체 계열 및 작업 표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참조합니다. 조합 하나는 라이브러리에 직접 확인합니다.

<code-tabs name="support" />
