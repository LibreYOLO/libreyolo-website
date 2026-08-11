---
title: ncnn
seo_title: LibreYOLO에서 ncnn으로 내보내기
description: >-
  PNNX를 거쳐 LibreYOLO 모델을 ncnn으로 내보냅니다. param 및 bin 파일 쌍, 고정 내보내기 캔버스, YOLOX
  Focus 재작성, 변환 가능한 계열을 설명합니다.
lead: >-
  ncnn은 모바일 대상을 위한 Tencent의 CPU 추론 라이브러리입니다. LibreYOLO는 PNNX를 거쳐 변환하며,
  model.ncnn.param 그래프와 model.ncnn.bin 가중치 파일을 나란히 작성하고 계열, 작업, 클래스 이름을 담은
  metadata.yaml도 생성합니다.
keywords:
  - yolo ncnn 변환
  - pnnx 사용법
  - model.ncnn.param
  - 모바일 cpu 추론
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="ncnn")
    mono: true
  - label: 출력
    value: 'model.ncnn.param, model.ncnn.bin, metadata.yaml이 있는 디렉터리'
  - label: 추가 설치
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: 다시 불러오기
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: 형상
    value: 고정입니다. 메타데이터는 플래그와 관계없이 dynamic=False를 기록합니다.
  - label: 정밀도
    value: FP32만 지원합니다. half=True와 int8=True는 거부됩니다.
verification: >-
  dev 브랜치의 libreyolo/export/ncnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/ncnn.py, pyproject.toml을
  확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        # pnnx가 변환하고 ncnn이 결과를 실행합니다.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t_ncnn 디렉터리를 작성합니다
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: 인수
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int 또는 (height, width)
            batch=1,
            simplify=True,    # ONNX 대체 경로에만 적용됩니다
            opset=None,       # 자동이며 ONNX 대체 경로에만 적용됩니다
            output_path=None, # None이면 weights/<stem>_ncnn을 작성합니다
        )

        # half=True와 int8=True는 검증 중 거부됩니다.
  run:
    - label: LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ncnn 직접 사용
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnn은 배치가 아니라 단일 CHW 이미지를 받습니다.
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # 이 경로에서는 전처리와 후처리를 직접 담당합니다.
  support:
    - label: 내보내기 전 계열 및 작업 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## 설치

<code-tabs name="install" />

extra는 도구 체인의 양쪽을 모두 가져옵니다. `pnnx`가 변환을 수행하고 `ncnn`이
결과를 실행합니다. 기본 경로에서는 어느 쪽도 ONNX를 거치지 않습니다.

## 내보내기

<code-tabs name="export" />

아티팩트는 디렉터리입니다. `weights/LibreYOLO9t_ncnn`에는
`model.ncnn.param`, `model.ncnn.bin`, `metadata.yaml`이 있으며, 세 파일은
하나의 아티팩트이므로 함께 이동해야 합니다.

변환은 먼저 PyTorch에서 PNNX로 직접 처리하려고 시도합니다. 실패하면 임시
디렉터리에 정적 ONNX 그래프를 내보내고 `pnnx` 명령줄 도구로 처리합니다. 두 경로가
모두 실패한 경우에만 내보내기가 두 오류를 모두 보고하며 실패합니다. 따라서
`opset`과 `simplify`는 대체 경로에만 영향을 줍니다.

YOLOX를 변환하려면 한 가지 재작성이 필요합니다. Focus 레이어는 PNNX가 낮출 수
없는 스트라이드 슬라이싱을 사용하므로, 내보내기가 이를 `pixel_unshuffle`로 바꾸고
서로 다른 채널 순서를 보정하도록 다음 컨볼루션의 입력 채널을 순열합니다. 출력은
수치적으로 동일하며 내보내기 후 원래 가중치가 복원됩니다.

## 아티팩트 실행

<code-tabs name="run" />

`LibreYOLO()`는 `model.ncnn.param`과 `model.ncnn.bin`이 있는 모든 디렉터리를
인식하고 `metadata.yaml`을 읽은 뒤 체크포인트와 동일한 `Results` 객체를
반환합니다.

두 번째 스니펫은 런타임을 직접 사용하는 경로이며, 두 가지 세부 사항이 여기의
다른 모든 형식과 다릅니다. ncnn은 배치가 아닌 단일 CHW 이미지로 작업하므로 앞에
배치 축이 없습니다. 블롭 이름은 `.param` 파일에서 가져옵니다. PNNX는 관례상
`in0`과 `out0`을 작성하며 백엔드는 이를 가정하지 않고 파일을 파싱합니다. 이
경로에서는 전처리, 디코딩, NMS, 좌표 크기 조정을 직접 담당합니다.

## 제약 조건

고정 캔버스의 FP32를 지원합니다. `half=True`와 `int8=True`는 모두 검증 중
거부되며, 내보낸 메타데이터는 플래그 값과 관계없이 `dynamic=False`를 기록하므로
백엔드가 그래프에 없는 축을 가정하지 않습니다.

모든 DETR 계열은 사전 검사에서 거부됩니다. `detr`, `deformable_detr`,
`dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`, `rtdetrv2`,
`rtdetrv4`, `rfdetr`, `ec`가 해당합니다. 모두 디코더 또는 샘플링 연산이 필요하지만
ncnn에서는 사용할 수 없다는 동일한 메시지를 표시하며, 대안으로 ONNX, OpenVINO,
TorchScript, TensorRT를 안내합니다.

반면 컨볼루션 계열은 광범위하게 변환됩니다. YOLO9, YOLO9-E2E, YOLOX, PicoDet,
YOLO-NAS 탐지 및 자세 추정, 이전 YOLO1, YOLO3, YOLO4, YOLO7 탐지기, CNN 분류기
4개 계열, PIDNet 시맨틱 분할, 고정 96 x 96 크기의 FOMO 포인트 탐지, ZipDepth,
NAFNet, Real-ESRGAN이 해당합니다.

차단된 항목은 구체적인 실패 원인을 명시합니다. 트랜스포머 그래프에는 일반적으로
지원되지 않는 `pnnx.Expression` 노드가 남아 실행 가능한 입력 블롭이 없는 네트워크가
생성되며, 이 문제로 DINOv2, CLIP, SigLIP2, SegFormer가 중단됩니다. BiRefNet에는
PNNX가 낮출 수 없는 torchvision 변형 가능 컨볼루션이 필요합니다. YOLO2의 변환된
그래프는 Windows에서 출력 추출 중 네이티브 정수 0 나누기로 ncnn 런타임을
종료합니다.

전체 계열 및 작업 표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참조합니다. 조합 하나를 확인하려면 다음을 실행합니다.

<code-tabs name="support" />
