---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net: LibreYOLO의 시선 추정'
description: >-
  LibreYOLO에서 2단계 시선 피치 및 요 추정에 L2CS-Net을 사용합니다. 설치, 예측, 내보내기를 지원하며 Gaze360
  체크포인트는 연구 전용입니다.
lead: >-
  L2CS-Net은 2단계 시선 추정기입니다. 얼굴 탐지기가 얼굴을 찾고 각도 구간 분류 헤드 2개가 있는 ResNet 트렁크가 얼굴마다
  피치와 요를 예측합니다. LibreYOLO는 이를 추론 전용으로 래핑합니다.
keywords:
  - L2CS-Net 사용법
  - 시선 추정
  - 아이트래킹
  - 피치 요 추정
  - Gaze360
  - 얼굴 탐지
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # face_detector를 지정하지 않으면 OpenCV 번들 얼굴 탐지기
        # (OpenCV 4에서는 Haar, OpenCV 5에서는 YuNet)를 사용하므로
        # L2CS 체크포인트 외에 추가로 내려받지 않고 실행할 수 있습니다.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 얼굴 소스
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # 이미 실행한 탐지기의 바운딩 박스를 L2CS에 전달합니다.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # 또는 특정 번들 얼굴 탐지기를 지정합니다.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: 내보낸 파일 사용하기
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # 내보낸 그래프는 ResNet 트렁크와 각도 구간 헤드 2개만 포함합니다.
        # 전처리된 448x448 얼굴 크롭을 입력받고 디코딩된 각도가 아닌 원시
        # (yaw_logits, pitch_logits)를 반환합니다. softmax, 구간 기댓값, 도 변환은
        # Python에 남습니다. libreyolo.models.l2cs.utils.bin_logits_to_angles를 참조합니다.
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## 설치

이미 보유한 체크포인트로 L2CS-Net을 생성, 예측, 내보내기하는 데는 extra가 필요하지
않습니다.

```bash
pip install libreyolo
```

LibreYOLO가 자동으로 가져올 수 있는 유일한 체크포인트는 Gaze360으로 학습한
ResNet-50입니다. LibreYOLO 조직이 아니라 저자의 Google Drive에 있으므로 일반 HTTP
미러 대신 `gdown`으로 내려받으며, 이 경로에는 `gaze` extra가 필요합니다.

```bash
pip install "libreyolo[gaze]"
```

이 extra가 없으면 조용히 실패하는 대신 LibreYOLO가 수동 내려받기 지침을 출력합니다.

## 예측

<code-tabs name="predict" />

L2CS-Net은 2단계 추정기입니다. 먼저 얼굴 탐지기를 실행하고 시선 헤드가 반환된 각
얼굴 크롭에서 피치와 요를 읽습니다. 그대로 사용하면 예측이 OpenCV 번들 탐지기로
대체되므로 L2CS 체크포인트가 있으면 추가로 내려받지 않고 기본 호출이 작동합니다.
`face_boxes`는 이미 실행한 탐지기의 바운딩 박스를 받고, `face_detector`는
`"auto"`, `"haar"`, `"yunet"`, LibreYOLO 탐지 모델 또는 일반 호출 가능 객체를
받습니다. `result.gaze`에는 탐지된 얼굴 바운딩 박스인 `result.boxes`와 행별로
정렬된 라디안 단위 피치와 요가 들어 있습니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

백본 깊이 5개가 하나의 입력 해상도를 공유하고 같은 인수를 받습니다. 공개된 유일한
체크포인트의 데이터셋인 Gaze360은 ResNet-50을 학습했습니다. 나머지 깊이 4개도
아키텍처상 지원하지만 불러올 공개 가중치는 없습니다.

## 내보내기

<export-matrix />

<code-tabs name="export" />

## 라이선스

<provenance-box>

LibreYOLO는 L2CS 체크포인트를 호스팅하거나 미러링하지 않습니다. 이 사이트의 다른
대부분 계열과 달리 LibreYOLO Hugging Face 조직에는 이 계열의 파일이 없습니다.
라이브러리가 자동으로 가져올 수 있는 유일한 체크포인트는 저자 자체 Google Drive
배포에서 직접 가져오며, 전송을 시작하기 전에 표시되는 Gaze360 라이선스 알림에 따라
제한됩니다. 위 요약이 암시하는 "huggingface.co/LibreYOLO에 다시 게시된" 사본이
아닙니다.

</provenance-box>
