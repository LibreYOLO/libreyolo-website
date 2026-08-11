---
title: 시선 추정
seo_title: LibreYOLO에서의 시선 추정
description: >-
  LibreYOLO에서 얼굴별 시선 피치와 요를 추정합니다. Python이나 CLI에서 예측하고, 각도를 라디안 단위로 읽으며, 시선 헤드를
  ONNX로 내보냅니다.
lead: >-
  시선 추정은 이미지의 모든 얼굴에 대한 시선 방향을 반환합니다. LibreYOLO는 이를 두 단계 작업으로 모델링합니다: 먼저 얼굴 탐지기가
  실행되고, 시선 헤드는 반환된 각 얼굴 크롭에서 피치와 요를 읽습니다.
keywords:
  - 시선 추정 파이썬
  - 아이 트래킹
  - 피치 요 요 시선
  - L2CS-넷
  - 응시 방향
  - 머리 자세
  - libreyolo 시선 작업
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # face_detector가 주어지지 않으면 예측은 OpenCV로 되돌아갑니다
        # 번들된 탐지기이므로 체크포인트를 넘어서는 것은 아무것도 다운로드되지 않습니다.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        gaze = result.gaze
        print(gaze.pitch, gaze.yaw)              # 라디안, 면마다 한 줄
        print(gaze.pitch_deg, gaze.yaw_deg)      # 같은 각도(도 단위)
        print(gaze.direction_3d)                 # (N, 3) 단위 벡터
    - label: CLI
      language: bash
      code: >
        # 파이썬 경로와 달리, CLI에는 자동 대체 기능이 없습니다: gaze

        # 모델은 명시적인 얼굴 탐지기가 필요하며, 그것은 반드시

        # 박스가 얼굴인 LibreYOLO 탐지기.

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: 얼굴 소스를 선택하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # 이미 실행한 탐지기의 시선 헤드 박스를 전달하십시오.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # 또는 번들로 제공되는 탐지기 중 하나의 이름을 말하십시오.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## 정의

시선 추정은 얼굴당 두 개의 각도를 반환합니다. `result.gaze`는 `Gaze` 형식의 `(N, 2)` 페이로드로, 0열은 피치, 1열은 요(야), 라디안 단위이며 `result.boxes`에서 탐지된 얼굴 상자와 행 단위로 정렬됩니다. 관례는 L2CS-Net이 사용하는 것과 같으며, 양의 요는 시선을 대상의 왼쪽으로 회전시키고 양의 피치는 시선을 아래로 회전시킵니다.

같은 페이로드는 각도를 위해 `pitch_deg` 및 `yaw_deg`를 노출하고, 카메라 프레임에서 단위 벡터 `direction_3d`와 `(N, 3)`를 `(x, y, z)` 열과 함께 노출합니다.

작업이 두 단계이기 때문에 예측은 두 모델에 의존합니다. 탐지기가 놓친 얼굴에는 시선 행이 없고, 잘못 배치된 박스는 잘못 잘린 얼굴에서 각도를 생성합니다. 표준 작업 키는 `gaze`이며, `gaze-estimation`는 이에 맞게 정규화됩니다.

## 모델들

[L2CS-Net](/docs/models/l2cs)은 이 작업을 수행하는 유일한 모델 계열입니다. 이 모델은 ResNet 트렁크를 2개의 병렬 각도-빈 분류 헤드와 결합하며, 하나는 피치용, 다른 하나는 요용으로 사용됩니다. 입력은 448x448 얼굴 크롭입니다. 아키텍처적으로 다섯 가지 백본 깊이를 지원하며, 그 중 하나인 ResNet-50은 공개된 체크포인트가 있습니다.

이 가중치에는 라이선스 제한이 있습니다. 이 가중치는 Gaze360에서 학습되었으며, 이 라이선스는 연구 및 비상업적 사용만 허용하고 재배포를 금지하므로 LibreYOLO는 이 계열에 대해 아무것도 미러링하지 않습니다. 라이브러리가 자동으로 가져올 수 있는 유일한 체크포인트는 라이선스 조건을 출력한 후, 저자들의 구글 드라이브 배포에서 직접 가져오는 것입니다(`gdown`). 배포하기 전에 [L2CS-Net](/docs/models/l2cs)를 읽으십시오.

해당 다운로드 경로에는 `gaze` 추가가 필요합니다:

```bash
pip install "libreyolo[gaze]"
```

그것이 없으면 라이브러리는 전송을 시도하는 대신 수동 다운로드 지침을 출력합니다. 이미 가지고 있는 체크포인트를 사용하여 예측하고 내보내는 데는 추가 작업이 전혀 필요하지 않습니다.

## 예측

<code-tabs name="predict" />

얼굴 소스는 세 가지 방법 중 하나로 선택됩니다. `face_boxes`는 이미 계산한 박스를 전달하고 탐지를 건너뜁니다. `face_detector`는 `"auto"`, `"haar"`, `"yunet"`, LibreYOLO 탐지 모델 또는 일반 호출 가능 객체를 수락하며, 생성자에서 설정하거나 호출당 설정할 수 있습니다. Python에서 설정되지 않은 경우, 예측은 OpenCV에 번들로 포함된 탐지기로 대체되므로 별도의 설정 없이 호출이 가능합니다. OpenCV 4에서는 바퀴 안에 포함된 Haar 캐스케이드로, 다운로드가 전혀 필요하지 않습니다; OpenCV 5에서는 Haar API가 제거되어 YuNet이 사용되며, 이는 OpenCV 동물원에서 작은 모델 파일을 한 번 가져옵니다.

CLI는 해당 폴백을 공유하지 않습니다. `libreyolo predict`는 `face_detector=` 없이 시선 모델을 거부하며, 그것이 취하는 값은 LibreYOLO 탐지기 이름 또는 체크포인트 경로입니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)을 참조하십시오.

## 학습

이 작업에서 어떤 계열도 LibreYOLO 안에서 학습하지 않습니다. `LibreL2CS.train()`는 다음과 같이 제안합니다: 업스트림 L2CS-Net 프로젝트에서 학습하고 생성된 상태 딕트를 여기에 로드하십시오.

## 검증

시선 실제값(gaze ground-truth) 데이터셋에 대한 검증은 범위에 포함되지 않으며, `val()`는 계산하지 않은 지표를 반환하지 않고 대신 예외를 발생시킵니다. 이 작업을 위한 `metrics/` 사전은 없습니다. 체크포인트가 학습된 데이터셋에서 업스트림에서 평가하십시오.

## 내보내기

<code-tabs name="export" />

Gaze 내보내기 계약은 ONNX, TorchScript, ExecuTorch, TensorRT 및 OpenVINO를 포함합니다. 라이브러리를 떠나는 것은 ResNet 트렁크와 두 개의 각도-빈 헤드뿐입니다: 그래프는 전처리된 448x448 얼굴 크롭을 입력으로 받아 원시 요(yaw) 및 피치(pitch) 로짓을 반환합니다. 얼굴 탐지, 크롭, 소프트맥스, 빈 기대값 및 각도로의 변환은 모두 Python, `libreyolo.models.l2cs.utils`에 남아 있습니다. 형식과 인수에 대해서는 [export](/docs/export)를 참조하십시오.
