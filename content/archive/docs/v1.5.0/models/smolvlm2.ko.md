---
title: SmolVLM2
families:
  - smolvlm2
seo_title: 'LibreYOLO의 SmolVLM2: 오픈 보캐뷸러리 탐지'
description: >-
  LibreYOLO에서 SmolVLM2를 설치하고 오픈 보캐뷸러리를 설정하여 Hugging Face의 Apache-2.0 비전 언어 모델로
  예측하거나 대화합니다.
lead: >-
  SmolVLM2는 Hugging Face의 소형 비전 언어 모델입니다. LibreYOLO는 이를 오픈 보캐뷸러리 객체 탐지기로 래핑하고 자유
  형식 대화를 직접 노출합니다. 탐지할 클래스 목록을 제공하거나 질문할 수 있습니다.
keywords:
  - SmolVLM2 사용법
  - 비전 언어 모델
  - 오픈 보캐뷸러리 탐지
  - 소형 멀티모달 모델
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 대화
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # 탐지 편의 기능 아래의 탈출구: 바운딩 박스 쿼리뿐 아니라
        # 어떤 질문이든 사용할 수 있습니다.
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
source_hash: b30823b62d6347b5
---

## 설치

SmolVLM2는 자체 팩토리가 있는 체크포인트 기반 계열과 별도의 제품 영역인
LibreYOLO의 VLM 탐지기 계층에 속합니다. `vlm` extra가 필요하며, SmolVLM2 자체
프로세서의 의존성인 `num2words`도 함께 설치합니다.

```bash
pip install "libreyolo[vlm]"
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다.

<code-tabs name="predict" />

이 계열은 `LibreYOLO()`가 아니라 `LibreVLM()` 팩토리로 불러옵니다. VLM 계열은
체크포인트 로더를 선언하지 않으므로 다른 모델 페이지에서 설명하는 파일 접미사
라우팅이 적용되지 않습니다. `set_classes()`는 SmolVLM2에 찾도록 요청할
보캐뷸러리를 설정합니다. 이 설정은 유지되므로 다시 설정할 때까지 이후의 모든
`predict()` 및 `track()` 호출에 적용됩니다. LibreYOLO에서 SmolVLM2에는 파서
재정의가 필요하지 않습니다. 이 계층의 공유 기본값과 같은 채팅 템플릿 및 JSON
출력을 따르므로 탐지 프롬프트와 바운딩 박스 형식은 계열별로 다르지 않습니다. 모든
탐지에는 동일한 자리표시자 신뢰도가 지정되므로 `conf` 필터링은 순위 지정이 아니라
전체 적용 또는 전체 제외로 동작합니다. `iou`는 효과가 있으며 반복 생성기가 한
객체에 대해 거의 중복되는 바운딩 박스를 출력할 수 있으므로 이미 유지한 같은 클래스
바운딩 박스와 임계값보다 많이 겹치는 후속 바운딩 박스를 제거합니다. SmolVLM2는
`LibreVLM` 팩토리에 문서화된 동일한 탈출구인 `chat()`을 통해 자유 형식 질문에도
응답합니다. LibreYOLO CLI는 이 계층을 지원하지 않습니다. `libreyolo predict
model=...` 형식은 사용할 수 없습니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

레지스트리에는 SmolVLM2-500M-Video-Instruct 한 크기가 있으며
`LibreVLM("smolvlm2-500m")`로 불러옵니다. SmolVLM2는 이 계층의 전용 그라운딩
모델보다 약한 탐지기입니다. LibreYOLO 자체 래퍼는 이를 가장 강력한 오픈
보캐뷸러리 선택지가 아니라, 특별한 파싱 없이 새 계열이 여기서 작동할 수 있음을
보여주는 예로 설명합니다.

LibreYOLO는 SmolVLM2를 학습, 검증, 내보내기하지 않습니다. 이 계층의 모든
계열에서 `train()`, `val()`, `export()`는 모두 `NotImplementedError`를
일으킵니다(위의 지원 계층 참조). 사용자 지정 보캐뷸러리를 가중치에 포함해야 하면
업스트림에서 SmolVLM2를 파인튜닝하고 생성된 가중치를 불러옵니다. 모든 탐지에
동일한 자리표시자 신뢰도가 있으므로 COCO 방식의 검증 과정 대신 `predict()` 출력을
직접 확인합니다.

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />

