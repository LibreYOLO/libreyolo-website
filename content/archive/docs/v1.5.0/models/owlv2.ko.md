---
title: OWLv2
families:
  - owlv2
seo_title: 'LibreYOLO의 OWLv2: 제로샷 객체 탐지'
description: >-
  LibreYOLO에서 OWLv2를 사용하여 텍스트로 설명한 모든 객체를 탐지합니다. openvocab extra를 설치하고 자유 텍스트
  보캐뷸러리로 예측합니다.
lead: >-
  OWLv2는 Google Research가 개발한 오픈 보캐뷸러리 객체 탐지기로, CLIP 방식 인코더의 텍스트 임베딩을 기준으로 이미지
  영역의 점수를 계산합니다. LibreYOLO는 이를 오픈 보캐뷸러리 탐지기 계층의 예측 전용 계열로 래핑합니다.
keywords:
  - OWLv2 사용법
  - OWL-ViT
  - 오픈 보캐뷸러리 객체 탐지
  - 제로샷 탐지
  - 텍스트 조건부 탐지기
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 기본 보캐뷸러리
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        # set_classes()를 생략하면 계층의 기본 COCO-80 보캐뷸러리를 유지합니다.
        model = LibreOpenVocab("owlv2-l14")
        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## 설치

OWLv2는 `openvocab` extra가 필요한 LibreYOLO의 오픈 보캐뷸러리 탐지기 계층으로
불러옵니다.

```bash
pip install "libreyolo[openvocab]"
```

이 extra는 해당 계층이 호출하는 Hugging Face 라이브러리인 `transformers`와
`timm`을 가져옵니다.

## 예측

OWLv2는 LibreYOLO가 `LibreYOLO()`로 불러오는 체크포인트가 아닙니다. 형제 팩토리인
`LibreOpenVocab`으로 불러오며, 처음 사용할 때 Hugging Face 스냅샷을 내려받아
`weights/` 아래에 캐시합니다.

<code-tabs name="predict" />

`set_classes()`는 유지되는 텍스트 보캐뷸러리를 설정합니다. 목록을 바꾸려면 다시
호출하고 기본 COCO-80 레이블을 유지하려면 생략합니다. 각 레이블은 텍스트 타워에
도달하기 전에 고정 프롬프트 템플릿으로 래핑되며, 이는 `transformers`의
`Owlv2ForObjectDetection` 학습 방식과 일치합니다.

OWLv2에는 텍스트 토큰 임계값이 없습니다. `conf`만 탐지를 필터링하며
`text_threshold`를 전달하면 오류가 발생합니다. API 호환성을 위해 `iou`를
허용하지만 경고만 표시하고 아무것도 하지 않습니다. 여기서는 NMS를 실행하지 않기
때문입니다. `imgsz`와 `augment=True`는 즉시 거부됩니다. `transformers`
프로세서가 크기 조정을 담당하며 테스트 시점 증강은 이 계층의 범위를 벗어납니다.
단일 이미지의 `predict()`는 목록이 아닌 하나의 `Results`를 반환합니다. 여러
결과를 얻으려면 디렉터리나 이미지 목록을 전달하거나 비디오 소스에 `stream=True`를
사용합니다. 이 계열에는 CLI 경로가 없습니다. `libreyolo predict`는
`LibreYOLO()`를 통해 `.pt` 체크포인트만 불러오므로 `LibreOpenVocab` 계열은
Python에서 실행합니다. 소스 유형과 스트리밍은 [예측](/docs/predict)을 참조합니다.

## 변형

체크포인트는 `b16`(base, 패치 크기 16)과 `l14`(large, 패치 크기 14) 두
가지입니다. 크기를 지정하지 않으면 `b16`이 이 계층의 기본값입니다. 두 체크포인트
모두 `transformers`의 `Owlv2ForObjectDetection`을 통해 공식 Google Research
릴리스를 미러링하며, 업스트림 파일을 보존하는 LibreYOLO 호스팅 Hugging Face
스냅샷으로 한 번 내려받습니다. 이 계열의 정확도 또는 지연 시간 수치는 아직
게시되지 않았습니다.

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

