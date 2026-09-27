---
title: Paddle
seo_title: LibreYOLO에서 PaddlePaddle로 내보내기
description: >-
  X2Paddle를 통해 LibreYOLO 탐지기를 PaddlePaddle 추론 모델로 변환: 고정 툴체인, 정적 배치-1 FP32 그래프,
  CPU 추론.
lead: >-
  PaddlePaddle 추론 모델은 model.pdiparams 가중치 파일 옆의 model.pdmodel 그래프입니다. LibreYOLO는
  정적 opset-15 ONNX 그래프를 내보내고, 이를 X2Paddle로 변환한 다음 metadata.yaml로 결과를 패키지하여 다른 모든
  런타임과 동일한 팩토리를 통해 불러올 수 있도록 합니다.
keywords:
  - yolo 패들 내보내기
  - 패들패들 추론
  - x2패들
  - model.pdmodel
  - model.pdiparams
  - onnx 연산 집합 15
last_verified: 1.5.0
meta:
  - label: 깃발
    value: export(format="paddle")
    mono: true
  - label: 쓴다
    value: 'model.pdmodel, model.pdiparams 및 metadata.yaml가 있는 디렉터리'
  - label: 추가
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: 다시 로드
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: 백엔드
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: 모양
    value: '정적, 배치 1, opset 15. 세 가지 모두 적용됩니다.'
  - label: 정확성
    value: 'FP32만, CPU만.'
  - label: 툴체인
    value: 'PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 또는 이전, 정확히 확인됨'
verification: >-
  개발 브랜치에서 libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md 및
  pyproject.toml를 읽으십시오.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        # Python 3.10에서 3.12까지. WSL2와 Ubuntu 22.04가 검증된 Windows 경로입니다.
        pip install "libreyolo[paddle]"
    - label: 고정된 버전을 확인하십시오
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

        # 디렉터리 weights/LibreYOLO9t_paddle를 작성합니다
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: 논쟁
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; 이 계열의 사각 캔버스
            batch=1,          # 다른 어떤 값도 ValueError를 발생시킵니다
            dynamic=False,    # True가 ValueError를 발생시킵니다
            simplify=True,    # False는 ValueError를 발생시킵니다
            opset=15,         # 다른 어떤 값도 ValueError를 발생시킵니다
            output_path=None, # 아무도 weights/<stem>_paddle를 쓰지 않습니다
        )
  run:
    - label: LibreYOLO를 통해
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
    - label: 백엔드가 직접
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # Paddle 디렉토리에 대해 LibreYOLO()가 생성하는 것. 동일한 결과
        # 객체, 그 사이에 팩토리 라우팅 없음.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: 맨 노
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

        # 이 경로에서는 전처리와 후처리를 직접 처리해야 함.
  support:
    - label: 내보내기 전에 하나의 계열와 작업을 확인하십시오
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## 설치

<code-tabs name="install" />

정확한 스택에서 여분의 핀은 패리티 작업 측정을 위해 다음과 같습니다: PaddlePaddle 2.6.2, X2Paddle 1.6.0 및 ONNX 1.17 이하. 이러한 핀은 설치 시뿐만 아니라 내보낼 때에도 확인되며, 다른 버전은 예상된 핀의 이름을 `ImportError`로 변경합니다. 최신 Paddle 릴리스는 X2Paddle 1.6.0이 생성한 정적 코드의 일부를 거부하므로, 누구도 검증하지 않은 아티팩트를 생성하는 것보다 미리 실패하는 것이 더 낫습니다.

## 내보내기

<code-tabs name="export" />

네 개의 인수는 기본값이 아니라 고정되어 있습니다. `dynamic`는 `False`여야 하고, `batch`는 1이어야 하며, `simplify`는 완전히 정적인 변환 그래프를 위해 `True`여야 하고, `opset`는 15여야 하는데, 이는 X2Paddle 1.6.0이 허용하는 최대값입니다. 다른 값을 전달하면 추적 전에 오류가 발생합니다.

중간 그래프에서 한 번의 정규화가 실행됩니다. ONNX는 생략된 MaxPool 확장을 1로 정의하고, PyTorch는 명시적으로 모두 1인 속성을 작성하며, X2Paddle 1.6.0은 이를 거부하므로, 익스포터는 해당 불필요한 기본값을 제거하고 지정된 연산을 변경하지 않고 그대로 둡니다.

해당 아티팩트는 디렉토리입니다: `model.pdmodel`, `model.pdiparams` 및 `metadata.yaml`. X2Paddle이 변환 중 생성하는 파이썬은 그 일부가 아닙니다.

## 아티팩트를 실행

<code-tabs name="run" />

`LibreYOLO()`는 `model.pdmodel`와 `model.pdiparams`를 모두 포함하는 디렉토리를 인식하고, `metadata.yaml`를 읽으며, 체크포인트로 동일한 `Results` 객체를 반환합니다. `auto`나 `cpu`가 아닌 장치는 다음 오류를 발생시킵니다: 이 백엔드는 CPU 전용입니다.

팩토리가 구성하는 것은 `PaddleBackend`이며, `libreyolo`에서 내보내기되고 `libreyolo.backends.paddle.PaddleBackend`로 가져올 수 있습니다. 팩토리의 접미사 라우팅 없이 백엔드를 원할 때 직접 구성하십시오. 예를 들어, 작성하지 않은 `metadata.yaml`의 디렉터리에 대해 `task=`를 명시적으로 전달하고자 할 때 그렇습니다. 그 `predict()`는 동일한 소스를 사용하고 동일한 결과를 반환합니다.

베어 런타임 스니펫은 백엔드에서 구성한 것을 그대로 반영하며, 세 가지 비활성화된 옵션은 의도된 것입니다. Paddle 2.6 CPU 퓨전 파이프라인은 변형된 어텐션을 위해 생성된 큰 gather 및 scatter 그래프를 최적화하는 동안 충돌할 수 있으므로, 포터블 비퓨전 정적 그래프가 패리티가 측정된 대상입니다. 전처리, 디코딩, NMS 및 좌표 재스케일링은 그 경로에서 직접 처리해야 합니다.

## 제약

동적 형태 없음, FP16 없음, INT8 없음, 내장 NMS 없음, GPU 런타임 없음.

검증된 조합은 YOLO9 검출, YOLO9-E2E 및 YOLO9-P2 검출, EC 검출, 포즈 및 세그먼테이션, RT-DETRv4, D-FINE, DEIM 및 DEIMv2 검출, 그리고 YOLO-NAS 검출 및 포즈입니다. 각각은 변환, CPU 런타임 재로드, 원시 출력 일치 및 일치된 공개 결과로 다루어집니다.

차단되었으며, 조합별로 사유가 기록됨:

| 조합 | 왜 |
|---|---|
| RF-DETR, 모든 작업 | ONNX opset 17과 GridSample이 필요합니다; X2Paddle 1.6.0은 opset 15 이하만 지원하며 GridSample 매퍼가 없습니다 |
| RT-DETR 및 RT-DETRv2 탐지 | 학습된 그래프는 opset 16 이상에서 GridSample이 필요합니다 |
| D-FINE 세분화 | 변환 및 재로드하지만, 마스크-로짓 상대 RMS 오차는 3.52%이고 최소 일치 마스크 IoU는 0.582입니다 |
| YOLO9 세분화 | YOLO9는 LibreYOLO에서 탐지만 가능합니다 |
| RTMDet-Ins 분할 | 동적 커널 마스크 디코드는 내보낸 런타임 계약이 없습니다 |

검증되었거나 차단된 것으로 나열되지 않은 모든 항목은 ONNX-투-패들 변환 경로를 통해 검증되지 않았다는 메모와 함께 거부됩니다.

전체 계열 및 작업 그리드는 [내보내기 매트릭스](/docs/reference/export-matrix)를 참조하십시오. 한 가지 조합의 경우:

<code-tabs name="support" />
