---
title: RKNN
seo_title: Rockchip NPU용 RKNN으로 내보내기
description: >-
  LibreYOLO 탐지기를 Rockchip .rknn 아티팩트로 컴파일합니다. 직접 설치하는 공급업체 SDK, 검증된 RK3588 변형
  4개, 시뮬레이터 동등성을 설명합니다.
lead: >-
  RKNN은 Rockchip의 컴파일된 NPU 형식입니다. LibreYOLO는 opset-19 ONNX 중간 형식을 내보내고 RKNN
  Toolkit2 SDK로 컴파일하며, 보드 없이 Toolkit2의 호스트 시뮬레이터에서 컴파일된 그래프를 ONNX Runtime과 비교할 수
  있습니다.
keywords:
  - yolo rknn 변환
  - rockchip npu 모델
  - rk3588 yolo
  - rknn-toolkit2 사용법
  - rknn 시뮬레이터 동등성
  - orange pi rockchip 추론
last_verified: 1.5.0
meta:
  - label: 플래그
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: 출력
    value: >-
      .rknn 파일 하나, .rknn.metadata.json 사이드카, verify=True일 때 .rknn.parity.json
      보고서
  - label: 추가 설치
    value: PyPI에는 없습니다. rknn-toolkit2는 직접 설치하는 공급업체 SDK입니다.
  - label: 다시 불러오기
    value: LibreYOLO로는 불러오지 않습니다. 아티팩트는 Rockchip 런타임을 사용해 보드에서 실행합니다.
  - label: 형상
    value: '고정 정사각형, 배치 1, opset 19입니다. 세 조건이 모두 강제됩니다.'
  - label: 정밀도
    value: 공급업체 부동소수점 빌드입니다. half=True와 int8=True는 거부됩니다.
  - label: 범위
    value: 'RK3588의 탐지 변형 4개: YOLO9-t, YOLO9-E2E-t, PicoDet-s, YOLO-NAS-s'
verification: >-
  dev 브랜치의 libreyolo/export/rknn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, docs/rknn.md를 확인했습니다. 측정된 동등성 수치는 docs/rknn.md에
  있는 2026-08-04 검증 기록에서 가져왔습니다.
snippets:
  install:
    - label: LibreYOLO 측
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 직접 설치하는 공급업체 SDK
      language: bash
      code: |
        # rknn-toolkit2는 별도 라이선스가 적용되는 Rockchip SDK입니다. LibreYOLO는
        # 이를 번들하거나 설치하지 않습니다. x86_64 Linux 전용이며 Windows에서는
        # WSL2 또는 Linux 컨테이너를 사용합니다.
        #
        # Toolkit2 2.3.2에는 setuptools<81이 필요하며 ONNX 1.19 이상에서는
        # 컴파일러가 여전히 가져오는 onnx.mapping이 제거되어 실패합니다.
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # 다음으로 Rockchip 자체 휠 저장소에서 일치하는 rknn-toolkit2 휠을
        # 설치하고 가져오기가 되는지 확인합니다.
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # weights/LibreYOLO9t.rknn과 weights/LibreYOLO9t.rknn.metadata.json을
        작성합니다

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: 인수
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # 대상 플랫폼이며 target= 및 target_platform=도 작동합니다
            imgsz=640,         # 기록된 변형의 캔버스와 일치해야 합니다
            batch=1,           # 다른 값은 NotImplementedError를 일으킵니다
            dynamic=False,     # True는 ValueError를 일으킵니다
            opset=19,          # 다른 값은 NotImplementedError를 일으킵니다
            verify=False,      # True이면 PC 시뮬레이터를 실행하고 동등성으로 판정합니다
        )
  parity:
    - label: 기존 ONNX 아티팩트와 보드 없이 동등성 비교
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: 컴파일 전 계열 및 작업 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## 설치

컴파일에는 Rockchip 자체 라이선스로 공급되는 공급업체 SDK인 RKNN Toolkit2가
필요하며 LibreYOLO 의존성이 아닙니다. `libreyolo[rknn]` extra는 없으며 이 형식과
관련된 모든 요소를 한 줄로 설치할 수는 없습니다.

<code-tabs name="install" />

컴파일하거나 수치 동등성을 확인하는 데 보드는 필요하지 않습니다. 지연 시간,
전력, 온도를 측정하려면 RK3588 보드가 필요하지만 아직 어떤 항목도 기록되지
않았습니다.

## 내보내기

<code-tabs name="export" />

컴파일을 시작하기 전에 정확한 모델 변형 목록을 기준으로 요청을 검증하고 캔버스도
검증합니다. 변형이 기록된 크기와 다른 `imgsz`를 전달하면 테스트되지 않은 항목을
조용히 컴파일하는 대신 오류가 발생합니다. LibreYOLO는 opset-19 ONNX 중간 형식을
작성하고 컴파일하며, 선택적으로 시뮬레이션한 뒤 중간 형식을 제거합니다.

RKNN 형식에는 이식 가능한 메타데이터 필드가 없으므로 메타데이터는
`<model>.rknn.metadata.json`이라는 사이드카입니다.

`verify=True`는 아티팩트를 컴파일한 동일한 세션에서 Toolkit2의 PC 시뮬레이터를
실행하고 동일한 입력에 대해 모든 출력을 ONNX Runtime과 비교하며, 출력별 오류
메트릭이 있는 `<model>.rknn.parity.json`을 작성합니다. 요소별 근접 조건을 이미
충족하지 않는 출력에는 코사인 유사도 0.9999 이상과 정규화 RMSE 0.02 이하라는
판정 기준을 적용합니다. 공급업체 부동소수점 빌드는 내부 텐서를 반정밀도로
낮추므로 디코딩된 바운딩 박스가 안정적이어도 엄격한 `allclose`는 성립하지
않습니다. 실패한 실행은 `<model>.rknn.failed.parity.json`을 작성하고 후보를
버리며 해당 경로의 이전 성공 내보내기는 그대로 둡니다.

다시 내보내지 않고 이미 보유한 ONNX 아티팩트를 비교하려면 다음을 실행합니다.

<code-tabs name="parity" />

Toolkit2 시뮬레이터는 `load_onnx`와 `build`가 생성한 인메모리 그래프를 실행합니다.
보드 없이는 대상별 `.rknn` 파일을 다시 불러올 수 없으므로 `verify=True`가 한
세션에서 컴파일, 내보내기, 시뮬레이션을 수행합니다.

## 아티팩트 실행

`libreyolo/backends`에는 RKNN 항목이 없으므로 `LibreYOLO()`는 `.rknn` 파일을
불러오지 않습니다. 컴파일된 아티팩트를 보드에 배포하고 Rockchip 자체 런타임으로
실행하며, 이 환경에서는 애플리케이션이 전처리, 디코딩, NMS, 좌표 크기 조정을
담당합니다.

`<model>.rknn.metadata.json`에는 클래스 이름, 입력 크기, 작업, 대상 플랫폼이
들어 있으며, 애플리케이션에서 LibreYOLO 후처리를 재현하는 데 필요한 정보입니다.
컴파일된 모델과 함께 배포합니다.

보드가 필요 없는 호스트 측 검사를 수행하려면 위와 같이 같은 고정 형상의 ONNX
아티팩트를 유지하고 시뮬레이터에서 비교합니다.

## 제약 조건

컴파일되는 네 가지 조합은 계열이 아니라 모델 변형입니다.

| 변형 | 작업 | 캔버스 | 대상 |
|---|---|---:|---|
| YOLO9-t | 탐지 | 640 | RK3588 |
| YOLO9-E2E-t | 탐지 | 640 | RK3588 |
| PicoDet-s | 탐지 | 320 | RK3588 |
| YOLO-NAS-s | 탐지 | 640 | RK3588 |

그 외 모든 항목은 컴파일 전에 거부되며 이 버전의 RKNN이 시뮬레이터로 테스트된
정확한 탐지 변형으로 제한된다는 메시지를 표시합니다. 다른 모델의 컴파일 전용
결과도 있지만 의도적으로 지원으로 제시하지 않습니다. 같은 측정 실행에서
RF-DETR에는 낮춰지지 않은 디코더 `GridSample` 노드가 2개 남았으며, D-FINE,
RT-DETR, RT-DETRv2, RT-DETRv4, DEIM, DEIMv2, EC는 컴파일과 시뮬레이션이
완료되었지만 디코딩된 출력이 실질적으로 잘못되었습니다.

배치 1, 정적 형상, opset 19를 사용합니다. RKNN은 LibreYOLO의 `half` 계약을
노출하지 않으므로 `half=True`가 거부되며, 대표 보정 및 작업 정확도 결과가 생길
때까지 `int8=True`도 거부됩니다.

다른 Rockchip 대상은 거부됩니다. 검증된 플랫폼은 `rk3588`뿐입니다.

전체 계열 및 작업 표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참조합니다. 조합 하나를 확인하려면 다음을 실행합니다.

<code-tabs name="support" />
