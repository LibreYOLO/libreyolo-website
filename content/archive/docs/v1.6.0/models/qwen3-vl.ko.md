---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: 'LibreYOLO의 Qwen3-VL: 오픈 보캐뷸러리 탐지'
description: >-
  LibreYOLO에서 Qwen3-VL을 설치하고 오픈 보캐뷸러리를 설정하여 Alibaba의 Apache-2.0 비전 언어 모델로 예측하거나
  대화합니다.
lead: >-
  Qwen3-VL은 네이티브 2D 그라운딩을 지원하는 Alibaba의 비전 언어 모델입니다. LibreYOLO는 이를 오픈 보캐뷸러리 객체
  탐지기로 래핑하고 자유 형식 대화를 직접 노출합니다. 탐지할 클래스 목록을 제공하거나 질문할 수 있습니다.
keywords:
  - Qwen3-VL 사용법
  - 비전 언어 모델
  - 오픈 보캐뷸러리 탐지
  - 이미지 그라운딩
  - Alibaba
  - VLM
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 대화
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # 탐지 편의 기능 아래의 탈출구: 바운딩 박스 쿼리뿐 아니라

        # 어떤 질문이든 사용할 수 있습니다.

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: "801d97d089f1f957"
---

## 설치

Qwen3-VL은 자체 팩토리가 있는 체크포인트 기반 계열과 별도의 제품 영역인
LibreYOLO의 VLM 탐지기 계층에 속합니다. `vlm` extra가 필요합니다.

```bash
pip install "libreyolo[vlm]"
```

## 예측

가중치는 처음 사용할 때 Hugging Face에서 내려받아 로컬에 캐시됩니다. 인수 없이
호출한 `LibreVLM()`은 Qwen3-VL-4B를 기본값으로 사용합니다.

<code-tabs name="predict" />

이 계열은 `LibreYOLO()`가 아니라 `LibreVLM()` 팩토리로 불러옵니다. VLM 계열은
체크포인트 로더를 선언하지 않으므로 다른 모델 페이지에서 설명하는 파일 접미사
라우팅이 적용되지 않습니다. `set_classes()`는 Qwen3-VL에 찾도록 요청할
보캐뷸러리를 설정합니다. 이 설정은 유지되므로 다시 설정할 때까지 이후의 모든
`predict()` 및 `track()` 호출에 적용됩니다. 모든 탐지에는 동일한 자리표시자
신뢰도가 지정되므로 `conf` 필터링은 순위 지정이 아니라 전체 적용 또는 전체 제외로
동작합니다. 이 계열에서 `iou`는 효과가 있으며 반복 생성기가 한 객체에 대해 거의
중복되는 바운딩 박스를 출력할 수 있으므로 이미 유지한 같은 클래스 바운딩 박스와
임계값보다 많이 겹치는 후속 바운딩 박스를 제거합니다. Florence-2 및 Kosmos-2와
달리 Qwen3-VL은 `LibreVLM` 팩토리에 문서화된 동일한 탈출구인 `chat()`을 통해
자유 형식 질문에도 응답합니다. LibreYOLO CLI는 이 계층을 지원하지 않습니다.
`libreyolo predict model=...` 형식은 사용할 수 없습니다. 소스, 스트리밍, 결과
처리는 [예측](/docs/predict)을 참조합니다.

## 변형

크기는 Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct,
Qwen3-VL-8B-Instruct 세 가지이며 각각 `LibreVLM("qwen3-vl-2b")`,
`LibreVLM("qwen3-vl-4b")`, `LibreVLM("qwen3-vl-8b")`로 불러옵니다. 세 모델
모두 명목상 1024 px 입력을 선언하지만 Qwen 프로세서 자체의 스마트 크기 조정이
네트워크에 전달되는 실제 캔버스를 결정합니다. 따라서 이 수치는 이 사이트의 다른
계열처럼 고정 동작 해상도가 아닙니다. LibreYOLO는 세 크기의 정확도를 비교하는
벤치마크를 게시하지 않았습니다.

## 학습

`libreyolo[vlm-train]`을 설치하면 `LibreVLM("qwen3-vl-2b").train(data=...)`로 탐지 LoRA 어댑터를 학습할 수 있습니다. 비전 타워는 고정된 상태를 유지하며, 검증 손실로 최적 체크포인트 디렉터리를 선택합니다. 옵티마이저 상태를 복원하는 학습 재개와 탐지 mAP 검증은 지원하지 않습니다. 기본값과 다시 로드하는 방법은 [VLM 파인튜닝](/docs/train/vlm-fine-tuning)을 참조하십시오.

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
