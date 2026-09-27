---
title: ONNX
seo_title: LibreYOLO에서 ONNX로 내보내기
description: >-
  LibreYOLO 모델을 ONNX로 내보내기: 각 계열별 LibreYOLO가 선택하는 opset, 동적 축, 내장 NMS, INT8, 그리고
  그래프가 다시 로드되는 방법.
lead: >-
  ONNX는 휴대 가능한 그래프 형식입니다. LibreYOLO는 torch.onnx.export로 모델을 추적하고, 선택적으로 그래프를
  단순화하며, 파일 자체 메타데이터에 계열, 작업, 클래스 이름 및 입력 크기를 기록하여 모든 LibreYOLO 백엔드가 후처리를 재구성할 수
  있도록 합니다.
keywords:
  - yolo onnx 내보내기
  - onnxruntime
  - torch.onnx.내보내기
  - onnx 연산 세트
  - 동적 축
  - 임베디드 NMS ONNX
  - onnx int8 qdq
  - onnx 메타데이터_속성
last_verified: 1.5.0
meta:
  - label: 깃발
    value: export(format="onnx")
    mono: true
  - label: 쓴다
    value: '하나의 .onnx 파일, 그래프에 메타데이터 포함'
  - label: 추가
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: 다시 로드
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: 모양
    value: 기본적으로 Python에서 동적 배치; 아래는 작업별 예외
  - label: 정확성
    value: 'FP32, FP16 (half=True), INT8 (int8=True, YOLO9 detection)'
verification: >-
  개발 브랜치에서 libreyolo/export/onnx.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/onnx.py 및
  libreyolo/cli/commands/export.py를 읽으십시오.
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

        # weights/LibreYOLO9t.onnx를 씁니다
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: 논쟁
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # 정수, 또는 (높이, 너비)
            batch=1,
            dynamic=True,     # 파이썬 기본값; CLI의 기본값은 False입니다
            simplify=True,    # 그래프에서 onnxsim 실행
            opset=None,       # DETR 스타일 계열에는 13이나 17을 선택하지 않습니다
            half=False,       # FP16 가중치 및 활성화
            int8=False,       # QDQ INT8, YOLO9 탐지만
            data=None,        # 보정 data.yaml, INT8 전용
            device=None,      # 추적 장치; 아무도 모델의 장치를 사용하지 않습니다
            output_path=None, # 아무도 weights/<stem>.onnx를 쓰지 않습니다
        )
  nms:
    - label: 그래프에 NMS를 삽입
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLO9 탐지만, 배치 1. dynamic은 False로 강제 설정됨.
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
    - label: 보정 데이터가 있는 INT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # 수백 장의 대표 이미지
            fraction=1.0,
        )
  run:
    - label: LibreYOLO를 통해
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 기본 ONNX 런타임
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # 이 경로에서는 전처리와 후처리를 직접 처리해야 함.

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # 그래프에는 계열, 작업, 클래스 이름과 입력 크기가 포함되어 있습니다.

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: 내보내기 전에 하나의 계열와 작업을 확인하십시오
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## 설치

<code-tabs name="install" />

추가로 `onnx`, `onnxsim` 및 `onnxruntime`가 가져옵니다. `onnx`만으로 파일을 작성하기에 충분하며; `onnxsim`는 단순화 단계를 실행하고 `onnxruntime`는 아티팩트를 실행하고 INT8 보정을 수행합니다.

## 내보내기

<code-tabs name="export" />

`output_path` 없이 파일은 체크포인트의 스템 아래 `weights/`에 저장되며, 해당 정밀도가 요청되었을 때 `_fp16` 또는 `_int8`가 추가됩니다.

`dynamic`는 Python에서는 `True`로, CLI에서는 `False`로 기본 설정됩니다. 이것이 켜져 있으면 배치 축이 기호화되고 몇 가지 작업이 더 확장됩니다: 의미론적 세그멘테이션은 마스크 높이와 너비를 열고, Real-ESRGAN 복원은 공간 축을 열며, 2단계 탐지기는 그래프 내에서 리사이즈가 발생하기 때문에 소스 높이와 너비를 동적으로 유지합니다.

`opset`는 생략될 때 각 계열마다 선택됩니다. DETR 스타일 계열(`detr`, `deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`)와 `deit`, `midas`, `moge2`는 opset 17을 사용하며, `aten::scaled_dot_product`가 낮아지는 곳입니다. 나머지는 모두
13. Matting은 어쨌든 19로 올려집니다. 왜냐하면 BiRefNet의 디코더가 `DeformConv` 연산자를 필요로 하고, ONNX가 이를 opset 19에서 정의하기 때문입니다.

`simplify=True`는 `onnxsim`를 실행하고 패스가 실패하면 원래 그래프를 유지하므로 단순화 오류는 내보내기 실패가 아니라 경고입니다. macOS arm64에서 `onnx` 1.22 이상과 `onnxsim` 0.6.5 이하를 사용할 경우 이 패스는 완전히 건너뛰며, 이는 해당 조합이 Python 프로세스를 중단시킬 수 있기 때문입니다.

### 임베디드 NMS

<code-tabs name="nms" />

`nms=True`는 YOLO9 탐지만 가능하며 배치 1이 필요합니다; `dynamic=True`로 요청하면 경고가 기록되고 동적 기능이 꺼집니다. 그러면 그래프에는 두 개의 출력이 있습니다: `output`, 형태는 `(batch, max_det, 6)`, 그리고 `raw`, LibreYOLO 자체 백엔드가 사용하는 디코딩되지 않은 탐지기 텐서로 후처리가 PyTorch 경로와 동일하게 유지됩니다.

### 딥스트림

`deepstream=True`는 ONNX 전용 옵션입니다. 이는 그래프를 NVIDIA DeepStream의 파서가 기대하는 레이아웃으로 내보내고, 그 옆에 두 개의 사이드카 파일인 `config_infer_primary_<stem>.txt`와 `<stem>_labels.txt`를 작성하여 수작업 구성 없이 아티팩트가 파이프라인에 바로 들어가도록 합니다.

이는 `nms=True`와 상호 배타적이며, 두 가지를 모두 요청하면 `ValueError`가 발생합니다: DeepStream은 자체 클러스터링 단계에서 억제를 실행합니다. 이를 ONNX가 아닌 다른 형식으로 전달하면 역시 오류가 발생합니다. 지원되는 계열와 작업 그리드 및 파서 빌드에 대해서는 [DeepStream](/docs/export/deepstream)를 참조하십시오.

### INT8

<code-tabs name="int8" />

`int8=True`는 ONNX Runtime 정적 양자화를 실행하고 float32 입력 및 출력을 가진 QDQ 그래프를 작성합니다. `Conv`와 `Gemm` 노드만 양자화됩니다. 탐지 헤드 디코드를 float32로 그대로 두는 것은 의도적인데, 그 연결(concatenation)은 픽셀 단위 상자 좌표와 0에서 1 범위의 클래스 점수를 혼합하고, 상자 크기에 지배되는 단일 텐서별 활성화 스케일이 모든 점수를 0으로 몰아갈 수 있기 때문입니다.

이 플래그는 현재 YOLO9 검출에만 적용되며, 다른 경우에는 사전 점검에서 `NotImplementedError`를 발생시킵니다. `data`를 생략하면 경고와 함께 `coco8.yaml`로 대체됩니다; 8개의 이미지는 대표적인 보정 세트가 아닙니다. 이미 PyTorch에서 양자화된 모델은 [양자화](/docs/export/quantization)에 설명된 다른 경로를 따릅니다.

## 아티팩트를 실행

<code-tabs name="run" />

`LibreYOLO()`는 `.onnx` 접미사에서 디스패치되며, 클래스 이름, 작업, 입력 크기 및 포즈 스키마가 내보내기 시 그래프의 `metadata_props`에 기록되었기 때문에 `.pt` 체크포인트로 동일한 `Results` 객체를 반환합니다. `device="auto"`을 사용하면 세션이 ONNX Runtime에서 이를 보고할 때 `CUDAExecutionProvider`을 가져오고, 그렇지 않으면 CPU로 대체됩니다.

두 번째 스니펫은 LibreYOLO가 설치되지 않은 읽는 사람들을 위한 것입니다. 전처리, 디코딩, NMS 및 좌표 재스케일링은 모두 해당 경로에서 사용자가 처리하게 되며; 메타데이터 블록은 여전히 읽을 수 있습니다.

## 제약

출력 텐서 이름은 작업마다 고정되어 있으며, 메타데이터가 없는 사용자가 일치시켜야 하는 것입니다:

| 작업 | 출력 이름 |
|---|---|
| 탐지, 그리드 및 앵커 헤드 | `output` |
| 탐지, DETR 스타일 | `pred_logits`, `pred_boxes` |
| 탐지, RF-DETR | `dets`, `labels` |
| 분류 | `output` |
| 시맨틱 분할 | `semantic_logits` |
| 깊이 | `depth` |
| 표면 법선 | `normal` |
| 모서리 | `edges` |
| 복원 | `restored` |
| 매팅 | `matte` |
| 응시 | `yaw_logits`, `pitch_logits` |

RF-DETR은 또한 입력 텐서가 `images`가 아니라 `input`로 명명된 유일한 계열이기도 합니다.

이 버전에서는 여러 작업이 고정 해상도 런타임 계약을 갖습니다. 깊이, 표면 법선 및 엣지 거부는 `batch != 1`를 사용하고, 강제는 `dynamic=False`를 사용합니다. 매팅은 BiRefNet의 Swin 상대 위치 테이블이 해상도에 맞춰져 있기 때문에 기본 1024 제곱을 강제합니다. 복원은 Real-ESRGAN을 제외한 모든 계열에서 고정 캔버스를 강제하며, Real-ESRGAN의 생성기는 완전 합성곱 방식입니다.

직사각형 `imgsz`는 YOLO9 계열, HRNet, NAFNet 및 Real-ESRGAN에 작동합니다. 고정된 정사각형 계약이 있는 계열(`clip`, `deformable_detr`, `detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `moge2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`)은 이를 완전히 거부합니다.

두 가지 조합은 추적 전에 거부됩니다: YOLO9 세분화는 LibreYOLO에서 YOLO9가 검출 전용이기 때문에, 그리고 RTMDet-Ins 세분화는 동적 커널 마스크 디코드에 내보낸 런타임 계약이 없기 때문입니다.

전체 계열 및 작업 그리드는 [내보내기 매트릭스](/docs/reference/export-matrix)를 참조하십시오. 하나의 조합에 대해서는 라이브러리에 직접 문의하십시오:

<code-tabs name="support" />
