---
title: Paddle
seo_title: LibreYOLO에서 PaddlePaddle로 내보내기
description: >-
  X2Paddle을 거쳐 LibreYOLO 탐지기를 PaddlePaddle 추론 모델로 변환합니다. 고정된 도구 체인, 정적 배치 1 FP32
  그래프, CPU 추론을 설명합니다.
lead: >-
  PaddlePaddle 추론 모델은 model.pdmodel 그래프와 model.pdiparams 가중치 파일로 구성됩니다.
  LibreYOLO는 정적 opset-15 ONNX 그래프를 내보내고 X2Paddle로 변환하며, 다른 모든 런타임과 같은 팩토리로 불러올 수
  있도록 metadata.yaml과 함께 패키징합니다.
keywords:
  - yolo paddle 변환
  - paddlepaddle 추론
  - x2paddle 사용법
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="paddle")
    mono: true
  - label: 출력
    value: 'model.pdmodel, model.pdiparams, metadata.yaml이 있는 디렉터리'
  - label: 추가 설치
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: 다시 불러오기
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: 백엔드
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: 형상
    value: '정적, 배치 1, opset 15입니다. 세 조건이 모두 강제됩니다.'
  - label: 정밀도
    value: FP32 및 CPU만 지원합니다.
  - label: 도구 체인
    value: 'PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 이하를 정확히 검사합니다.'
verification: >-
  dev 브랜치의 libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md,
  pyproject.toml을 확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        # Python 3.10부터 3.12까지 지원합니다. Ubuntu 22.04의 WSL2가 검증된 Windows 경로입니다.
        pip install "libreyolo[paddle]"
    - label: 고정 버전 확인
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t_paddle 디렉터리를 작성합니다
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: 인수
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int이며 이 계열의 정사각형 캔버스입니다
            batch=1,          # 다른 값은 ValueError를 일으킵니다
            dynamic=False,    # True는 ValueError를 일으킵니다
            simplify=True,    # False는 ValueError를 일으킵니다
            opset=15,         # 다른 값은 ValueError를 일으킵니다
            output_path=None, # None이면 weights/<stem>_paddle을 작성합니다
        )
  run:
    - label: LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: 백엔드 직접 사용
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # LibreYOLO()가 Paddle 디렉터리용으로 생성하는 항목입니다. 팩토리 라우팅
        # 없이 동일한 Results 객체를 반환합니다.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Paddle 직접 사용
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # 이 경로에서는 전처리와 후처리를 직접 담당합니다.
  support:
    - label: 내보내기 전 계열 및 작업 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## 설치

<code-tabs name="install" />

extra는 동등성 작업에서 측정한 정확한 스택인 PaddlePaddle 2.6.2, X2Paddle 1.6.0,
ONNX 1.17 이하로 고정합니다. 이러한 고정 버전은 설치할 때뿐 아니라 내보낼 때도
검사하며, 다른 버전이면 예상 버전을 명시한 `ImportError`가 발생합니다. 최신 Paddle
릴리스는 X2Paddle 1.6.0이 생성하는 일부 정적 코드를 거부하므로 아무도 검증하지 않은
아티팩트를 만드는 것보다 일찍 실패하는 편이 낫습니다.

## 내보내기

<code-tabs name="export" />

4개 인수는 기본값이 아니라 고정값입니다. 완전한 정적 변환 그래프를 위해 `dynamic`은
`False`, `batch`는 1, `simplify`는 `True`여야 하며, `opset`은 X2Paddle 1.6.0이
허용하는 상한인 15여야 합니다. 다른 값을 전달하면 추적 전에 오류가 발생합니다.

중간 그래프에 한 가지 정규화를 실행합니다. ONNX는 생략된 MaxPool dilation을 1로
정의하고 PyTorch는 모든 값이 1인 속성을 명시적으로 작성하지만 X2Paddle 1.6.0은 이를
거부합니다. 따라서 내보내기는 중복된 기본값을 제거하고 지정된 연산을 바꾸지
않습니다.

아티팩트는 `model.pdmodel`, `model.pdiparams`, `metadata.yaml`이 있는
디렉터리입니다. X2Paddle이 변환 중 생성하는 Python은 아티팩트에 포함되지 않습니다.

## 아티팩트 실행

<code-tabs name="run" />

`LibreYOLO()`는 `model.pdmodel`과 `model.pdiparams`가 모두 있는 모든 디렉터리를
인식하고 `metadata.yaml`을 읽은 뒤 체크포인트와 동일한 `Results` 객체를
반환합니다. 이 백엔드는 CPU 전용이므로 `auto` 또는 `cpu`가 아닌 디바이스는 오류를
일으킵니다.

팩토리가 생성하는 항목은 `libreyolo`에서 내보내고
`libreyolo.backends.paddle.PaddleBackend`로 가져올 수 있는 `PaddleBackend`입니다.
팩토리의 접미사 라우팅 없이 백엔드를 사용하려면 직접 생성합니다. 예를 들어 직접
작성하지 않은 `metadata.yaml`이 있는 디렉터리에 `task=`를 명시적으로 전달할 수
있습니다. `predict()`는 같은 소스를 받고 같은 결과를 반환합니다.

런타임 직접 사용 스니펫은 백엔드 구성을 그대로 반영하며 3개 옵션을 의도적으로
비활성화했습니다. Paddle 2.6 CPU 융합 파이프라인은 변형 가능 어텐션용으로 생성된 큰
gather 및 scatter 그래프를 최적화하는 동안 중단될 수 있으므로, 이식 가능한 비융합
정적 그래프를 기준으로 동등성을 측정했습니다. 이 경로에서는 전처리, 디코딩, NMS,
좌표 크기 조정을 직접 담당합니다.

## 제약 조건

동적 형상, FP16, INT8, 내장 NMS, GPU 런타임은 지원하지 않습니다.

검증된 조합은 YOLO9 탐지, YOLO9-E2E 및 YOLO9-P2 탐지, EC 탐지, 자세 및 분할,
RT-DETRv4, D-FINE, DEIM, DEIMv2 탐지, YOLO-NAS 탐지 및 자세입니다. 각 조합은
변환, CPU 런타임 다시 불러오기, 원시 출력 동등성, 일치하는 공개 결과로 검증됩니다.

차단된 조합과 기록된 이유는 다음과 같습니다.

| 조합 | 이유 |
|---|---|
| RF-DETR, 모든 작업 | ONNX opset 17 및 GridSample이 필요하지만 X2Paddle 1.6.0은 opset 15 이하만 허용하고 GridSample 매퍼가 없습니다. |
| RT-DETR 및 RT-DETRv2 탐지 | 학습된 그래프에 opset 16 이상의 GridSample이 필요합니다. |
| D-FINE 분할 | 변환하고 다시 불러오지만 마스크 로짓 상대 RMS 오차가 3.52%이며 일치한 최소 마스크 IoU가 0.582입니다. |
| YOLO9 분할 | LibreYOLO의 YOLO9은 탐지만 지원합니다. |
| RTMDet-Ins 분할 | 동적 커널 마스크 디코딩에 내보낸 런타임 계약이 없습니다. |

검증됨 또는 차단됨으로 나열되지 않은 항목은 ONNX에서 Paddle로의 변환 경로에서
검증되지 않았다는 알림과 함께 거부됩니다.

전체 계열 및 작업 표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참조합니다. 조합 하나를 확인하려면 다음을 실행합니다.

<code-tabs name="support" />
