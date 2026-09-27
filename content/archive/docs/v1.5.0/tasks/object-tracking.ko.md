---
title: 객체 추적
seo_title: LibreYOLO에서 객체 추적
description: >-
  LibreYOLO에서 ByteTrack, BoT-SORT, OC-SORT 또는 Deep OC-SORT를 사용하여 모든 검출, 분할 또는 포즈
  모델 위에서 비디오 프레임 전체의 객체를 추적합니다.
lead: >-
  트래킹은 비디오 프레임 전체에서 각 검출에 안정적인 아이덴티티를 할당합니다. LibreYOLO는 이를 자체 가중치를 가진 작업으로 모델링하지
  않습니다: 이는 선택된 트래커를 검출, 세분화 또는 포즈 모델의 프레임별 출력 위에서 실행하는 예측 모드인 model.track()입니다.
keywords:
  - 객체 추적 파이썬
  - 다중 객체 추적
  - 바이트트랙
  - 봇소트
  - 오크소트
  - 딥 오코르트
  - 트랙 아이디
  - 레이드 추적
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track()는 제너레이터입니다: 처리된 각 프레임마다 하나의 Results를 반환합니다.
        for result in model.track("video.mp4"):
            print(result.track_id)        # (N,) 정수 텐서, 박스와 정렬됨
            print(result.boxes.xyxy)
    - label: 트래커 선택
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (기본), "botsort", "ocsort" 또는 "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: 주석이 달린 비디오 저장
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # output_path가 없으면 파일은 runs/track/<video_stem>.mp4.에 저장됩니다
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: 트래커 조정
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # 설정 유형이 트래커를 선택하므로 여기서 tracker=는 중복됩니다.

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # 또는 동일한 필드를 키워드 인수로 전달하고 track()가 그것을 구성하게 하십시오.

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## 정의

추적은 LibreYOLO의 작업 키 중 하나가 아니며, 다운로드할 추적 체크포인트도 없습니다. 이는 `model.track(source)` 모델의 한 메서드로, 각 프레임에서 검출을 실행하고 시간에 따라 결과를 연관시킵니다. 이 메서드는 제너레이터로, 처리된 각 프레임마다 하나의 `Results`를 생성하며, `result.track_id`는 `result.boxes`에 맞춰진 `(N,)` 정수 텐서로 설정됩니다. 동일한 ID는 `result.boxes.id`에도 있습니다.

오직 확인된, 현재 추적 중인 객체만 제공됩니다. 추적에서 연관성이 끊어진 객체는 삭제되기 전에 설정된 프레임 수 동안 유지됩니다. ByteTrack와 BoT-SORT의 경우 `track_buffer`, 두 OC-SORT 변형의 경우 `max_age`이며, 따라서 그 시간 안에 복구된 객체는 원래의 ID를 유지합니다.

연관은 탐지 후에 발생하기 때문에, 프레임의 다른 페이로드는 영향을 받지 않고 유지됩니다: 추적된 `Results`는 탐지된 `Results`에서 일치하는 행으로 잘린 것이므로, 마스크와 키포인트도 박스와 함께 전달됩니다.

## 모델들

트래킹 실행에는 두 가지 독립적인 선택이 포함됩니다: 각 프레임에서 박스를 생성하는 모델과 박스를 연결하는 트래커입니다.

탐지, 분할 또는 포즈를 수행하는 모든 기본 LibreYOLO 모델은 `track()`를 노출하므로 탐지기 선택은 일반적인 선택과 동일합니다. 전체 목록은 [모델 인덱스](/docs/models)를 참조하거나 [YOLO9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr), [D-FINE](/docs/models/d-fine) 또는 [RTMDet](/docs/models/rtmdet)에서 시작할 수 있습니다. 결과에 연결할 박스가 없는 작업은 무의미한 ID를 반환하지 않고 호출을 거부합니다. 분류, 방향 박스, 포인트, 깊이, 표면 법선, 엣지, 의미론적 및 범주적 분할, 복원, OCR 및 신체 메시 모두 `track()`에서 예외를 발생시킵니다.

LibreYOLO의 모델 계층 중 두 개도 이를 거부합니다. `LibreSAM`를 통해 로드된 모델은 이미지 분할기이고, `LibreOpenVocab`를 통해 로드된 모델은 프레임별 탐지기입니다. 둘 다 `track()`에서 상속되며 대신 `predict()`와 함께 프레임별로 사용됩니다.

추적은 네이티브 PyTorch 모델에서 실행됩니다. `LibreYOLO("model.onnx")`를 통해 로드된 내보낸 아티팩트는 런타임 백엔드 객체를 반환하며, 이 객체는 `predict()`는 포함하지만 `track()`는 포함하지 않습니다.

라이브러리에는 `tracker` 인수로 선택된 네 개의 트래커가 함께 제공됩니다:

`"bytetrack"`가 기본값입니다. 이는 모션 전용으로, 칼만 필터와 세 단계 연관을 사용합니다: 먼저 높은 신뢰도의 검출, 그런 다음 낮은 신뢰도의 검출이 기존 트랙과 일치할 기회를 제공하는 두 번째 패스가 있으며, 이를 버리기 전에 마지막으로 확인되지 않은 트랙을 처리합니다. `TrackConfig`로 구성됩니다.

`"botsort"`는 ByteTrack의 세 단계 수명 주기를 유지하지만, 중심-너비-높이 칼만 상태를 사용하고 매칭 전에 카메라 움직임에 대해 예측된 트랙을 보정합니다. 이것은 BoT-SORT의 움직임 전용 변형이며, 외형 모델을 실행하지 않습니다. `BoTSortConfig`로 구성되며, 이 구성에는 `enable_cmc`, `cmc_method` 및 `cmc_downscale`가 추가됩니다.

`"ocsort"`는 또한 모션 전용이며, 연관 비용에 속도-방향 항을 추가하고, 각 트랙의 마지막 실제 관측치에 대해 두 번째 연관 패스를 수행하며, 트랙이 다시 발견될 때 가상 궤적을 따라 칼만 상태를 평활화합니다. `OCSortConfig`로 구성됩니다.

`"deepocsort"`는 외형 정보를 활용하여 OC-SORT를 확장합니다. 각 트랙은 재식별 임베딩의 신뢰도 가중 이동 평균을 유지하며, 코사인 유사도 항이 연관 비용에 포함되어, 신원이 긴 가림 현상과 교차 타겟에서도 유지됩니다. 프레임당 작은 임베딩 네트워크 전방 전달 한 번의 비용이 발생하며, OSNet 가중치는 첫 사용 시 다운로드됩니다. `DeepOCSortConfig`로 구성되어 있습니다.

## 예측

<code-tabs name="predict" />

`track_conf`는 첫 번째 연관 단계의 임계값을 설정합니다: ByteTrack과 BoT-SORT의 경우 `track_high_thresh`, OC-SORT와 Deep OC-SORT의 경우 `det_thresh`입니다. 이것은 `predict()`의 `conf`가 아니며, ByteTrack, BoT-SORT 및 OC-SORT의 경우 탐지기는 내부적으로 더 낮은 임계값에서 실행되어 약한 탐지가 복구 단계에서 계속 사용할 수 있도록 합니다. Deep OC-SORT는 탐지기를 `det_thresh` 자체에서 실행합니다. ByteTrack과 BoT-SORT의 경우 `track_conf`는 `track_low_thresh` 이상이어야 하며, 기본값은 0.1입니다.

트래커 설정은 두 가지 방법 중 하나로 이루어집니다. 구성 인스턴스를 `tracker_config=`에 전달하면, 해당 타입이 트래커를 선택하여 `tracker=`는 불필요해집니다. 또는 필드를 키워드 인수로 전달하고 `track()`가 지정한 트래커용 구성을 생성하도록 할 수 있습니다. 알 수 없는 키는 조용히 적용되지 않고 경고를 표시합니다. 어느 쪽이든, 일치하는 키가 명시적으로 설정되면 `track_conf`는 무시됩니다.

나머지 인수는 예측을 반영합니다: `iou`, `imgsz`, `classes`, `max_det`, `vid_stride`, `show`, `save` 및 `output_path`. 소스는 비디오 파일 경로입니다. 결과 처리는 [prediction](/docs/predict)를 참조하십시오.

## 학습

추적기는 학습되지 않았습니다. 네 대 중 세 대는 학습된 매개변수가 전혀 없는 순수한 모션 모델이며, Deep OC-SORT의 외형 네트워크는 처음 사용할 때 다운로드되는 공개된 재식별 체크포인트입니다. 추적 품질을 향상시키려면 탐지기를 개선하거나 위의 연관 임계값을 조정해야 합니다.
