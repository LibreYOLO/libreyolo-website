---
title: Florence-2
families:
  - florence2
seo_title: 'LibreYOLO의 Florence-2: 오픈 보캐뷸러리 탐지'
description: >-
  LibreYOLO에서 Florence-2를 설치하고 오픈 보캐뷸러리를 설정하여 Microsoft의 MIT 라이선스 비전 모델로 바운딩 박스를
  예측합니다.
lead: >-
  Florence-2는 고정 탐지 헤드를 거치는 대신 작업 토큰으로 프롬프트하는 Microsoft의 비전 파운데이션 모델입니다.
  LibreYOLO는 이를 오픈 보캐뷸러리 객체 탐지기로 래핑하며, 예측할 때 클래스 목록을 제공합니다.
keywords:
  - Florence-2 사용법
  - 비전 언어 모델
  - 오픈 보캐뷸러리 탐지
  - 이미지 그라운딩
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 비디오
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # 라이브러리가 허용하는 모든 소스: 파일, 폴더, URL, 웹캠 인덱스,
        # RTSP 스트림 또는 .streams 목록
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## 설치

Florence-2는 자체 팩토리가 있는 체크포인트 기반 계열과 별도의 제품 영역인
LibreYOLO의 VLM 탐지기 계층에 속합니다. `vlm` extra가 필요합니다.

```bash
pip install "libreyolo[vlm]"
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다. LibreYOLO는
원본 `microsoft/Florence-2-*` 저장소 대신 florence-community가 다시 업로드한
체크포인트를 내려받습니다. 그 이유는 라이선스 절을 참조합니다.

<code-tabs name="predict" />

이 계열은 `LibreYOLO()`가 아니라 `LibreVLM()` 팩토리로 불러옵니다. VLM 계열은
체크포인트 로더를 선언하지 않으므로 다른 모델 페이지에서 설명하는 파일 접미사
라우팅이 적용되지 않습니다. `set_classes()`는 Florence-2에 이미지에서 찾도록
요청할 보캐뷸러리를 설정합니다. 이 설정은 유지되므로 다시 설정할 때까지 이후의
모든 `predict()` 및 `track()` 호출에 적용됩니다. 반환되는 `Results`의 `boxes`는
다른 모든 계열과 같은 형상이지만, 모든 탐지에는 동일한 자리표시자 신뢰도가
지정됩니다. 따라서 `conf` 필터링은 순위 지정이 아니라 전체 적용 또는 전체 제외로
동작하며 `iou`는 효과가 없습니다. Florence-2 래퍼가 중복 제거 단계 없이 파싱된
작업 토큰 출력에서 직접 탐지 목록을 만들기 때문입니다. Florence-2는 채팅 템플릿이
아니라 `<OPEN_VOCABULARY_DETECTION>` 작업 토큰으로 구동되므로 여기서 `chat()`은
`NotImplementedError`를 일으킵니다. LibreYOLO CLI는 이 계층을 지원하지 않습니다.
`libreyolo predict model=...` 형식은 사용할 수 없습니다. 소스, 스트리밍, 결과
처리는 [예측](/docs/predict)을 참조합니다.

## 변형

크기는 Florence-2-base와 Florence-2-large 두 가지이며 모두 768 px입니다.
`LibreVLM("florence-2-base")` 또는 `LibreVLM("florence-2-large")`로 불러옵니다.
LibreYOLO는 두 모델의 정확도를 비교하는 벤치마크를 게시하지 않았습니다.

LibreYOLO는 Florence-2를 학습, 검증, 내보내기하지 않습니다. 이 계층의 모든
계열에서 `train()`, `val()`, `export()`는 모두 `NotImplementedError`를
일으킵니다(위의 지원 계층 참조). 사용자 지정 보캐뷸러리를 가중치에 포함해야 하면
업스트림에서 Florence-2를 파인튜닝하고 생성된 가중치를 불러옵니다. 모든 탐지에
동일한 자리표시자 신뢰도가 있으므로 COCO 방식의 검증 과정 대신 `predict()` 출력을
직접 확인합니다.

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />

