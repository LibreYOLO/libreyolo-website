---
title: Triton Inference Server
seo_title: NVIDIA Triton에서 LibreYOLO 모델 서빙
description: >-
  NVIDIA Triton을 통해 LibreYOLO ONNX 내보내기를 서빙합니다. 모델 저장소 레이아웃, 생성되는 config.pbtxt,
  HTTP 모델 URL을 대상으로 하는 예측을 설명합니다.
lead: >-
  Triton Inference Server는 모델 저장소를 호스팅하고 HTTP를 통해 추론 요청에 응답합니다. LibreYOLO는 ONNX
  그래프를 내보내고, 내보내기 메타데이터를 하나의 Triton 매개변수로 담은 config.pbtxt를 생성하며, 모델 URL을 불러올 수
  있는 모델 경로로 처리합니다.
keywords:
  - libreyolo triton 배포
  - triton inference server 사용법
  - config.pbtxt 설정
  - tritonclient http
  - triton 모델 저장소
  - 원격 yolo 추론
last_verified: 1.5.0
meta:
  - label: 호출
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: 도우미
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: 추가 설치
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: 프로토콜
    value: 'HTTP 및 HTTPS V2 추론만 지원합니다. gRPC, 인증, 공유 메모리, 모델 로드 및 언로드는 지원하지 않습니다.'
  - label: 타임아웃
    value: 연결 및 네트워크 타임아웃의 기본값은 30초입니다.
verification: >-
  dev 브랜치의 libreyolo/backends/triton.py, libreyolo/models/__init__.py,
  docs/triton.md, pyproject.toml을 확인했습니다. 컨테이너 명령은 docs/triton.md에 고정된 명령입니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: 저장소 레이아웃으로 내보내기
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: config.pbtxt 생성
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: 생성된 레이아웃
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: 서버 시작
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: 준비 상태 대기
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: 서버 중지
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: 서빙된 모델로 예측
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 로컬 모델과 비교
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: 버전 고정 또는 타임아웃 변경
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # 두 번째 경로 세그먼트가 모델 버전을 선택합니다. 없으면
        # Triton에 구성된 버전 정책이 선택합니다.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # 연결 및 네트워크 타임아웃의 기본값은 30초입니다.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## 설치

<code-tabs name="install" />

`triton` extra는 `tritonclient[http]`를 설치합니다. 이 통합은 HTTP 및 HTTPS V2
추론만 지원하므로 gRPC와 공유 메모리 extra는 의도적으로 제외됩니다. 서빙되는
아티팩트와 구성 생성기가 모두 ONNX 그래프를 사용하므로 `onnx`가 필요합니다.

## 모델 저장소 빌드

Triton이 예상하는 디렉터리 레이아웃에 동적 배치 축으로 내보냅니다.

<code-tabs name="repo" />

Triton은 모델 구성 응답에서 ONNX 사용자 지정 메타데이터를 보존하지 않으므로 전체
내보내기 메타데이터를 다른 방식으로 전달해야 합니다. `create_triton_config`는 이를
`config.pbtxt`의 `libreyolo_metadata`라는 단일 JSON 문자열 매개변수로 인코딩하고,
그래프 순서대로 입출력 선언을 생성하며, JSON 이스케이프를 처리하고, 모델을
`KIND_CPU`로 고정합니다.

도우미는 쓰기 전에 검증합니다. 정확히 하나의 ONNX 그래프 입력과 하나 이상의
출력, 해석 가능한 텐서 형상, `0`부터 `nc - 1`까지 모든 클래스 인덱스를 정의하는
`names` 맵이 있는 메타데이터가 필요합니다. 이러한 검사 중 하나라도 실패한 모델은
첫 요청이 아니라 구성 단계에서 거부됩니다.

`max_batch_size: 8`은 동적 내보내기와 일치하며 서버가 요청당 최대 8개 이미지를
배치 처리할 수 있게 합니다. 고정 배치 1 ONNX 그래프에는 `max_batch_size=0`을
사용합니다. 이 경우 LibreYOLO는 이미지를 순차적으로 전송합니다.

## 서버 시작

<code-tabs name="serve" />

명령은 Triton Server 26.04를 고정하며 Docker GPU 플래그를 의도적으로 생략합니다.
생성된 구성의 `KIND_CPU`가 어떤 경우에도 GPU 배치를 막기 때문입니다.

## 아티팩트 실행

Triton 모델 URL은 모델 경로입니다. `LibreYOLO()`는 로컬 경로를 처리하기 전에
`http` 또는 `https` 스킴을 확인하고 서버와 통신하는 백엔드를 반환하므로 호출
위치는 로컬 체크포인트와 동일하며 반환되는 `Results` 객체도 동일합니다.

<code-tabs name="run" />

URL 형식은 선택적 버전 세그먼트가 있는 `http(s)://host:port/model`입니다. 포트를
명시해야 합니다. 내장 자격 증명, 쿼리 문자열, 프래그먼트는 모두 거부되며 세그먼트가
둘보다 많은 경로도 거부됩니다.

배치는 서버가 결정하므로 `device`는 허용되지만 로그 한 줄과 함께 무시됩니다.

## 제약 조건

계약을 충족하지 못하면 백엔드는 성능이 저하된 결과 대신 직접적인 오류를
발생시킵니다. 모델 구성에 LibreYOLO 메타데이터가 없거나, 모델 입력이 둘 이상이거나,
구성된 출력과 모델 메타데이터가 일치하지 않거나, 지원하지 않는 입력 데이터 형식이
있거나, 서버 또는 모델이 준비되지 않은 경우가 해당합니다.

이 버전에서 계약 범위를 벗어나는 항목은 gRPC, 인증, 공유 메모리, API를 통한 모델
로드 또는 언로드입니다.

Triton 자체가 지원하는 모든 형식을 서빙할 수 있지만 여기서 메타데이터 매개변수와
생성된 구성은 ONNX에 맞춰져 있으므로 LibreYOLO 경로는 저장소로
[ONNX](/docs/export/onnx)를 내보내는 방식입니다. 요청과 응답 서버가 아니라 전체
비디오 파이프라인이 필요하면 [DeepStream](/docs/export/deepstream)을 참조합니다.
