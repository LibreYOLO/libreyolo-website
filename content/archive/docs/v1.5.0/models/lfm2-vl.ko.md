---
title: LFM2-VL
families:
  - lfm2vl
seo_title: 'LFM2-VL: LibreYOLO의 오픈 보캐뷸러리 탐지'
description: >-
  LibreYOLO에서 LFM2-VL을 사용해 온디바이스 오픈 보캐뷸러리 객체 탐지를 수행합니다. 어떤 텍스트 레이블로도 예측할 수 있으며
  학습, 검증, 내보내기는 지원하지 않습니다.
lead: >-
  LFM2-VL은 Liquid AI가 출시한 소형 온디바이스 비전 언어 모델입니다. LibreYOLO는 이를 오픈 보캐뷸러리 객체 탐지기로
  래핑합니다. 어떤 텍스트 레이블 목록이든 고정 헤드와 파인튜닝 없이 클래스 집합으로 사용할 수 있습니다.
keywords:
  - LFM2-VL 사용법
  - LFM2
  - Liquid AI
  - 비전 언어 모델
  - 오픈 보캐뷸러리 탐지
  - VLM
  - 엣지 VLM
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # 오픈 보캐뷸러리: 고정 클래스 헤드가 아니라 어떤 단어든 작동합니다. 다시
        # 설정할 때까지 이후의 모든 predict()/track() 호출에서 유지됩니다.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 원시 대화
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # 탐지 편의 기능 아래의 탈출구: 자유 형식 질문, 개수 세기 또는
        # 바운딩 박스 래퍼가 다루지 않는 모든 프롬프트
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## 설치

LFM2-VL에는 채팅 템플릿 백본용 `transformers`를 가져오는 `vlm` extra가
필요합니다.

```bash
pip install "libreyolo[vlm]"
```

## 예측

`LibreLFM2VL`은 `.pt` 체크포인트가 아니라 Python 클래스입니다.
`LibreYOLO()` 팩토리로 불러오지 않으며 `libreyolo` CLI도 이를 해석하지 않습니다.
`LibreVLM(...)` 팩토리(`from libreyolo import LibreVLM`)도 별칭을 통해 이 계열을
지원합니다. 예를 들어 `LibreVLM("lfm2-vl-450m")`을 사용할 수 있으며, 아래에서
사용하는 클래스가 이 팩토리의 생성 결과입니다. 가중치는 LibreYOLO 미러가 아니라
Liquid AI 자체 Hugging Face 저장소에서 가져옵니다. 처음 호출할 때 내려받아 로컬에
캐시하며, 내려받기 전에 일회성 라이선스 알림을 기록합니다.

<code-tabs name="predict" />

`result.boxes`에는 다른 모든 계열과 같은 방식으로 파싱된 탐지가 들어 있습니다.
신뢰도는 자리표시자입니다. LFM2-VL은 바운딩 박스별 점수를 출력하지 않으므로 모든
탐지에 동일한 상수 신뢰도가 지정되고, `conf=`는 해당 상수보다 낮은 행만 제거할 뿐
순위를 지정하지 않습니다. `iou`는 지정된 겹침을 초과하는 같은 클래스의 거의
중복된 바운딩 박스를 버립니다. 탐욕적 디코딩이 객체를 반복하는 데 따른 부수
효과이며 클래스별 NMS 과정은 아닙니다. `set_classes()`를 생략하면 보캐뷸러리는
COCO-80 이름을 기본값으로 사용합니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 변형

크기는 450m와 1.6b 두 가지이며 모두 온디바이스 배포용으로 설계된 Liquid AI의
LFM2.5-VL 릴리스에서 가져옵니다. LibreYOLO 벤치마크 하네스는 이 계열을 측정하지
않았으므로 두 크기를 비교할 공개 정확도 수치는 없습니다. 자체 컴퓨팅 예산에 맞춰
크기를 선택합니다.

LibreYOLO는 이 계열에서 예측만 노출합니다. `train()`, `val()`, `export()`는
모두 `NotImplementedError`를 일으킵니다. 업스트림에서 파인튜닝하고 결과를 대신
불러옵니다. 자리표시자 신뢰도 때문에 COCO mAP가 오해를 일으킬 수 있으므로
데이터셋 검증은 건너뛰며, 추적할 상태 딕셔너리가 없는 생성 모델의 내보내기는 범위를
벗어납니다.

## 라이선스

<provenance-box>

LFM Open License v1.0은 상업적 사용, 복제, 수정을 허용하지만 연간 매출 1,000만
달러 미만으로 제한합니다. 이 기준 이상의 법인은 본 계약에 따라 어떤 상업적 사용도
허가되지 않으며 Liquid AI에 직접 문의해야 합니다. 자격을 갖춘 비영리 단체는
비상업적 사용 또는 연구 목적에서 이 기준의 적용을 받지 않습니다. 모델은 Apache-2.0
`transformers` 라이브러리를 통해 불러오므로 LibreYOLO는 LiquidAI 소스 코드를
포함하지 않으며 가중치를 호스팅하거나 재배포하지도 않습니다. `LibreLFM2VL`은 처음
실행할 때 일치하는 크기를 Liquid AI 자체 Hugging Face 저장소에서 직접 내려받고,
내려받기 전에 일회성 알림을 기록합니다.

</provenance-box>

## 인용

<citation-block />
