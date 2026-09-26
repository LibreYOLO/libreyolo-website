---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'LibreYOLO의 OMDet-Turbo: 실시간 제로샷 탐지'
description: >-
  LibreYOLO에서 실시간 오픈 보캐뷸러리 탐지에 OMDet-Turbo를 사용합니다. openvocab extra를 설치하고 자유 텍스트
  보캐뷸러리로 예측합니다.
lead: >-
  OMDet-Turbo는 Om AI Lab이 개발한 실시간 오픈 보캐뷸러리 객체 탐지기로, 클래스 임베딩을 언어 작업 프롬프트와 분리합니다.
  LibreYOLO는 이를 오픈 보캐뷸러리 탐지기 계층의 예측 전용 계열로 래핑합니다.
keywords:
  - OMDet-Turbo 사용법
  - OmDet
  - 오픈 보캐뷸러리 객체 탐지
  - 실시간 객체 탐지
  - 제로샷 탐지
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 사용자 지정 NMS 임계값
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["traffic light", "bicycle"])

        # OMDet-Turbo는 이 계층에서 iou=를 적용하는 유일한 계열입니다. 자체
        # 후처리가 억제 임계값을 인수로 받으며 iou=를 설정하지 않으면
        # 기본값은 0.5입니다.
        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)
        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## 설치

OMDet-Turbo는 `openvocab` extra가 필요한 LibreYOLO의 오픈 보캐뷸러리 탐지기
계층으로 불러옵니다.

```bash
pip install "libreyolo[openvocab]"
```

이 extra는 해당 계층이 호출하는 Hugging Face 라이브러리인 `transformers`와
`timm`을 가져옵니다. OMDet-Turbo의 Swin 백본은 `transformers`의 `TimmBackbone`
래퍼를 통해 불러옵니다.

## 예측

OMDet-Turbo는 LibreYOLO가 `LibreYOLO()`로 불러오는 체크포인트가 아닙니다. 형제
팩토리인 `LibreOpenVocab`으로 불러오며, 처음 사용할 때 Hugging Face 스냅샷을
내려받아 `weights/` 아래에 캐시합니다.

<code-tabs name="predict" />

`set_classes()`는 유지되는 텍스트 보캐뷸러리를 설정합니다. 목록 전체를 바꾸려면
다시 호출하고 기본 COCO-80 레이블을 유지하려면 생략합니다. 빈 결과도 오류가 아니라
유효한 결과입니다. Grounding DINO와 달리 OMDet-Turbo는 클래스 임베딩을 언어 작업
프롬프트와 분리하므로 `transformers` 후처리가 반환하는 레이블은 구문 모호성 해소
단계 없이 쿼리한 클래스 목록으로 바로 매핑됩니다.

OMDet-Turbo에는 텍스트 토큰 임계값이 없습니다. `conf`만 탐지를 필터링하며
`text_threshold`를 전달하면 오류가 발생합니다. 이 계층에서 자체 NMS를
`post_process_grounded_object_detection` 내부에서 실행하는 유일한 계열이므로
`iou`는 경고 대신 적용됩니다. `imgsz`와 `augment=True`는 즉시 거부됩니다.
`transformers` 프로세서가 크기 조정을 담당하며 테스트 시점 증강은 이 계층의 범위를
벗어납니다. 단일 이미지의 `predict()`는 목록이 아닌 하나의 `Results`를
반환합니다. 여러 결과를 얻으려면 디렉터리나 이미지 목록을 전달하거나 비디오 소스에
`stream=True`를 사용합니다. 이 계열에는 CLI 경로가 없습니다. `libreyolo
predict`는 `LibreYOLO()`를 통해 `.pt` 체크포인트만 불러오므로 `LibreOpenVocab`
계열은 Python에서 실행합니다. 소스 유형과 스트리밍은 [예측](/docs/predict)을
참조합니다.

## 변형

체크포인트는 이 계층의 유일한 크기인 `t` 하나입니다. 고정된 업스트림 리비전의
`omlab/omdet-turbo-swin-tiny-hf`를 `transformers`의
`OmDetTurboForObjectDetection`을 통해 미러링하며, 미러링된 가중치 파일은 해당
업스트림 스냅샷과 바이트 단위로 동일합니다. 이 계열의 정확도 또는 지연 시간 수치는
아직 게시되지 않았습니다.

학습, 데이터셋 검증, 내보내기는 모두 이 계층의 범위를 벗어납니다. `train()`,
`val()`, `export()`는 모두 조건 없이 `NotImplementedError`를 일으킵니다. 공개된
체크포인트를 감싼 예측 전용 래퍼입니다.

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
