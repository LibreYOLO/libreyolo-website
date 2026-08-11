---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything: 오픈 보캐뷸러리 탐지와 포인팅'
description: >-
  LibreYOLO에서 LocateAnything으로 오픈 보캐뷸러리 탐지와 포인팅을 수행합니다. 어떤 텍스트 레이블로도 예측할 수 있으며
  학습, 검증, 내보내기는 지원하지 않습니다.
lead: >-
  LocateAnything은 좌표 토큰을 한 번에 하나씩 생성하는 대신 바운딩 박스와 점을 병렬로 디코딩하는 NVIDIA의 비전 언어
  그라운딩 모델입니다. LibreYOLO는 이를 오픈 보캐뷸러리 탐지기와 포인터로 래핑합니다. 고정 헤드나 파인튜닝 없이 어떤 텍스트 레이블
  목록이든 클래스 집합으로 사용할 수 있습니다.
keywords:
  - LocateAnything 사용법
  - NVIDIA 비전 언어 모델
  - 오픈 보캐뷸러리 탐지
  - 점 탐지
  - VLM 그라운딩
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # 오픈 보캐뷸러리로 고정 클래스 헤드가 아니므로 어떤 단어든 사용할 수 있습니다.
        # 다시 설정할 때까지 이후 모든 predict()/track() 호출에 유지됩니다.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 점 프롬프팅
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        # task="point"는 박스 대신 일치한 객체마다 점 하나를 반환합니다.
        # 이미 불러온 모델에서는 model.set_task("point")로 작업을 전환합니다.
        model = LibreLocateAnything(size="3b", task="point")
        model.set_classes(["the person closest to the camera"])
        result = model(SAMPLE_IMAGE, save=True)

        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: 원시 채팅
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # 탐지 편의 기능 아래의 탈출구로, 자유 형식 질문과 개수 세기 또는
        # 박스 래퍼가 지원하지 않는 모든 프롬프트를 사용할 수 있습니다.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 378ea758e507a096
---

## 설치

LocateAnything에는 `vlm` extra가 필요합니다. 이 extra는 모델을 불러올 때 Hugging Face 원격 코드가 가져오는 `decord`, `lmdb`, `peft` 패키지와 `transformers`를 설치합니다.

```bash
pip install "libreyolo[vlm]"
```

## 예측

`LibreLocateAnything`은 `.pt` 체크포인트가 아니라 Python 클래스입니다. `LibreYOLO()` 팩토리로 불러오지 않으며 `libreyolo` CLI도 이를 해석하지 않습니다. `LibreVLM(...)` 팩토리(`from libreyolo import LibreVLM`)에서도 `LibreVLM("locate-anything")` 같은 별칭으로 이 계열에 접근할 수 있습니다. 아래에 사용하는 클래스는 이 팩토리가 생성하는 클래스입니다. 모델을 불러오면 Hugging Face에서 NVIDIA의 원격 모델 코드를 다운로드하고 실행합니다. 따라서 LibreYOLO는 변경 가능한 `main` 브랜치 대신 고정된 단일 커밋 리비전으로 다운로드를 고정하며 첫 다운로드 전에 라이선스 고지를 한 번 기록합니다.

<code-tabs name="predict" />

`result.boxes`(`detect` 작업)와 `result.points`(`point` 작업)는 다른 계열과 같은 방식으로 파싱된 출력을 담습니다. 신뢰도는 자리표시자입니다. LocateAnything은 박스별 점수를 출력하지 않으므로 모든 탐지에 같은 상수 신뢰도가 지정됩니다. `conf=`는 해당 상수보다 낮은 행만 제거할 뿐 순위를 매기지 않습니다. `set_classes()`를 생략하면 어휘는 기본적으로 COCO-80 이름을 사용합니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

공개된 크기는 3b 하나입니다. 두 작업이 동일한 가중치를 공유합니다. 기본값인 `detect`는 박스를 반환하고 `task="point"`는 일치한 객체마다 박스 대신 점 하나를 `result.points`에 반환합니다. 이미 불러온 모델에서는 `model.set_task("point")`로 두 작업을 전환할 수 있습니다. LibreYOLO 벤치마크 도구는 이 계열을 측정하지 않았으므로 비교할 수 있는 공개 정확도 수치가 없습니다.

LibreYOLO는 이 계열을 예측 전용으로 제공합니다. `train()`, `val()`, `export()`는 모두 `NotImplementedError`를 발생시킵니다. 파인튜닝은 업스트림에서 수행한 뒤 결과를 불러와야 합니다. 자리표시자 신뢰도로는 COCO mAP가 오해를 일으킬 수 있어 데이터셋 검증을 건너뛰며, 추적할 state dict가 없는 생성형 모델이므로 내보내기는 범위에 포함되지 않습니다.

## 라이선스

<provenance-box>

NVIDIA License는 사용, 복제, 수정을 허용하지만 NVIDIA와 그 계열사를 제외한 모든 사용자에게 모델과 파생물을 비상업적 사용, 연구 또는 평가로만 제한합니다. 수익 기준이나 유료 예외는 없습니다. LocateAnything-3B는 라이선스가 별도인 두 구성 요소도 결합합니다. 언어 백본은 Qwen Research License의 Qwen2.5-3B-Instruct이고 비전 인코더는 MIT 라이선스의 MoonViT-SO-400M입니다. LibreYOLO는 어느 구성 요소도 호스팅, 미러링 또는 재배포하지 않습니다. `LibreLocateAnything`은 처음 실행될 때 고정된 단일 커밋의 가중치와 필수 원격 코드를 Hugging Face의 `nvidia/LocateAnything-3B`에서 직접 다운로드합니다.

</provenance-box>

## 인용

<citation-block />
