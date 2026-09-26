---
title: ExecuTorch
seo_title: LibreYOLO에서 ExecuTorch로 내보내기
description: >-
  LibreYOLO 모델을 XNNPACK 델리게이트를 적용한 ExecuTorch .pte 프로그램으로 내보내기: 고정 형상, 배치 1,
  FP32, 그리고 함께 필요한 메타데이터 사이드카.
lead: >-
  ExecuTorch는 PyTorch 프로그램을 엣지 타깃에서 실행합니다. LibreYOLO는 torch.export의 strict 모드로
  모델을 캡처하고, XNNPACK으로 로어링(lowering)한 뒤, .pte 프로그램과 JSON 메타데이터 사이드카를 하나의 단위로
  커밋합니다.
keywords:
  - yolo executorch 변환
  - pte 파일 실행
  - xnnpack 파티셔너
  - torch.export strict 모드
  - executorch 런타임
  - 엣지 pytorch 추론
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="executorch")
    mono: true
  - label: 생성 파일
    value: .pte 프로그램 하나와 .pte.json 메타데이터 사이드카 하나
  - label: 추가 설치
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: 다시 불러오기
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: 형상
    value: 고정입니다. dynamic=True와 batch != 1은 거부됩니다.
  - label: 정밀도
    value: FP32만 지원합니다. half=True와 int8=True는 거부됩니다.
  - label: 델리게이트
    value: 'XNNPACK, CPU. delegate=''xnnpack''만 허용되는 값입니다.'
verification: >-
  dev 브랜치의 libreyolo/export/executorch.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/executorch.py와 pyproject.toml을
  읽고 확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        # 의도적으로 libreyolo[all]에서 제외했습니다: ExecuTorch는 함께 사용할 수 있는
        # Torch 버전을 제한합니다.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.pte와 weights/LibreYOLO9t.pte.json을 생성합니다
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: 인자
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int, 또는 (높이, 너비)
            batch=1,               # 다른 값은 ValueError를 발생시킵니다
            dynamic=False,         # True는 ValueError를 발생시킵니다
            delegate="xnnpack",    # 유일하게 허용되는 값
            device="cpu",          # 다른 장치는 ValueError를 발생시킵니다
            output_path=None,      # None이면 weights/<stem>.pte에 씁니다
        )
  run:
    - label: LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ExecuTorch 런타임만 사용
      language: python
      code: >
        import json

        from pathlib import Path


        import torch

        from executorch.runtime import Runtime


        runtime = Runtime.get()

        print(runtime.backend_registry.is_available("XnnpackBackend"))


        program =
        runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())

        method = program.load_method("forward")


        # 이 경로에서는 전처리와 후처리를 직접 해야 합니다.

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: 내보내기 전에 계열과 작업 하나 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## 설치

<code-tabs name="install" />

이 extra 패키지는 ExecuTorch가 함께 동작하는 Torch 버전을 고정하고 이를 설치하면
환경 전체가 그 조합에 묶이기 때문에, 의도적으로 `libreyolo[all]` 밖에 두었습니다.
제약을 감수할 수 있는 환경에 설치하십시오.

Windows에서는 로어링 단계가 ExecuTorch에 함께 포함된 `flatc` 실행 파일을
호출합니다. `PATH`에 없으면 내보내기가 그 사실을 알리는 `RuntimeError`를
발생시키며, Visual Studio 2022 Developer PowerShell에서 실행하면 해결됩니다.

## 내보내기

<code-tabs name="export" />

캡처는 `torch.export.export(..., strict=True)`이며, 이는 기록된 트레이스가 아니라
가드를 갖춘 실제 그래프 캡처입니다. 호스트 스칼라 읽기와 데이터 의존적 제어 흐름은
조용히 상수로 굳어지는 대신 거부되므로, 다른 곳에서는 트레이스가 성공하는 몇몇
계열이 여기서는 실패합니다; 그 이유는 지원 매트릭스에 조합별로 기록되어 있습니다.

로어링은 XNNPACK 파티셔너와 함께 `to_edge_transform_and_lower`를 실행합니다.
결과에 델리게이트 파티션이 하나도 없으면, 내보내기는 포터블 커널만 사용하는
프로그램을 XNNPACK으로 표시하는 대신 예외를 발생시킵니다.

프로그램과 사이드카는 함께 커밋됩니다. 둘 다 스테이징되고 둘 다 교체되며, 실패하면
이전에 있던 상태로 되돌아가므로, 불완전한 한 쌍이 디스크에 남는 일은 없습니다.

## 산출물 실행

<code-tabs name="run" />

`LibreYOLO()`는 `.pte` 접미사를 보고 디스패치하며, 체크포인트와 동일한 `Results`
객체를 반환합니다. 사이드카는 불러올 때 반드시 필요합니다: `<program>.pte.json`이
없으면 백엔드가 `FileNotFoundError`를 발생시키는데, 프로그램 자체에는 클래스 이름,
작업, 입력 크기가 담겨 있지 않기 때문입니다. 백엔드는 불러오기 전에 설치된 런타임이
`XnnpackBackend`를 제공하는지도 확인하고, 파일을 매핑하는 대신 바이트로 프로그램을
읽어 들여, 백엔드가 살아 있는 동안 Windows 파일 잠금을 유지하지 않도록 합니다.

두 번째 스니펫은 런타임을 직접 사용하는 경로입니다. 거기서는 전처리, 디코딩, NMS,
좌표 재조정을 직접 처리해야 합니다.

## 제약

배치 1, 고정 형상, FP32, CPU. `batch != 1`과 `dynamic=True`는 내보내기가 무언가를
바꾸기 전에 모두 `ValueError`를 발생시키고, `half=True`와 `int8=True`는 검증
단계에서 거부되며, CPU가 아닌 장치는 허용되지 않습니다.

`delegate`는 이 버전에서 `"xnnpack"` 외의 값을 받지 않습니다.

분류 내보내기에는 `crop_pct`와 `interpolation`이라는 메타데이터 키 두 개가 추가로
들어가므로, 런타임이 해당 계열의 리사이즈와 센터 크롭 정책을 재현할 수 있습니다.

차단된 항목은 범주가 아니라 구체적인 실패를 명시합니다. D-FINE의 탐지와 분할은
strict 캡처에서 디포머블 어텐션의 지원되지 않는 `ContextVar` 읽기에 도달하고, 수동
grid-sample 경로를 강제하면 직렬화까지는 되지만 실행 시점에 잘못된 델리게이트 텐서
차원 순서로 실패합니다. DEIM과 DEIMv2는 캡처, 로어링, 직렬화까지 통과한 뒤 실행
중에 실패합니다. EoMT 시맨틱 분할은 마스크 경로의 데이터 의존적 심볼릭 표현식에서
실패합니다. BiRefNet 매팅은 1024 x 1024에서 캡처되지만
`torchvision::deform_conv2d`의 out 변형이 없습니다. SwinIR 복원은 다시
불러오기까지는 되지만 차원 순서가 맞지 않아 `aten::alias_copy.out`에서 실패합니다.

전체 계열과 작업 그리드는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참고하십시오. 조합 하나만 확인하려면:

<code-tabs name="support" />
