---
title: 오픈 보캐뷸러리 API
seo_title: 'LibreOpenVocab API: 별칭과 인수'
description: >-
  LibreOpenVocab 팩토리, 그 네 개의 계열과 모든 별칭, set_classes, 계열별 기본 설정, 그리고
  text_threshold와 iou 규칙.
lead: >-
  LibreOpenVocab는 텍스트 조건 검출기를 위한 팩토리입니다. 클래스 목록은 고정된 헤드가 아니라 프롬프트이므로, 어휘는
  set_classes에 의해 설정되며 모델은 그에 대해 일반적인 검출 결과를 반환합니다.
keywords:
  - 리브레오픈보캡
  - 개방형 어휘 탐지
  - 그라운딩 디노
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - 클래스 설정
last_verified: 1.5.0
verification: >-
  별칭은 libreyolo/models/openvocab/__init__.py에서 읽고; 저장소, 크기 및 임계값은
  grounding_dino.py, owlv2.py, omdet_turbo.py 및 ov_deim.py에서; 호출 규칙은
  libreyolo/models/openvocab/base.py에서, 모두 v1.5.0 기준. 설계 의도는
  docs/adr/0008-open-vocab-detector-contract.md.에서.
snippets:
  install:
    - label: 배시
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## 설치

해당 계층에는 `openvocab` 추가가 필요합니다.

<code-tabs name="install" />

## 팩토리

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model`는 경로가 아니라 별칭입니다. 조회 전에 밑줄(_)은 하이픈(-)으로 변환되므로, CLI 인벤토리가 출력하는 계열-정의 이름(예: `omdet_turbo-t` 및 `grounding_dino-t`)은 그대로 로드됩니다. 알 수 없는 별칭은 알려진 모든 별칭을 나열하는 `ValueError`를 발생시킵니다.

생성자는 `size`, `nb_classes=80`, `names=None`, `device="auto"`, `task=None` 및 `text_threshold=None`를 받습니다. `names`를 전달하는 것은 로딩 직후 `set_classes`를 호출하는 것과 같습니다. 지원되지 않는 계열에 `text_threshold`를 전달하면 `TypeError`가 발생합니다.

<code-tabs name="usage" />

## 계열과 별칭

| 계열 | 별명 | 사이즈 | 무게 |
|---|---|---|---|
| 그라운딩 디노 | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

기본 별칭은 `grounding-dino-tiny`입니다.

`LibreGroundingDINO`, `LibreOWLv2` 및 `LibreOMDetTurbo`는 패키지 수준에서 내보낼 수 있으며 `size=`로 직접 구성할 수 있습니다. OV-DEIM은 위의 팩토리 별칭을 통해 접근할 수 있습니다.

## 클래스 설정

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

모든 이후 `predict()` 호출에 대한 어휘를 설정하고, 호출들이 체인될 수 있도록 모델을 반환합니다. 목록은 비어 있어서는 안 되며, 문자열만 포함해야 하고, 대소문자를 구분하지 않고 비교했을 때 항목이 고유해야 합니다; 빈 레이블은 거부됩니다. 단일 문자열을 전달하면 `TypeError`가 발생하는데, 이는 하나의 문자 클래스만 생성되기 때문입니다.

통화 후, `model.names`는 `0..N-1`를 주어진 순서대로 레이블에 매핑하고, `model.nb_classes`는 `N`입니다.

## 호출 인수

이 계층은 세 가지 차이점을 갖고 표준 예측 표면을 재사용합니다.

`conf`는 공유된 0.25가 아니라 계열 자체의 값으로 기본 설정됩니다:

| 계열 | 기본 설정 | 억압 |
|---|---|---|
| 그라운딩 디노 | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | 자체 후처리, 임계값 0.5, `iou=`를 준수합니다 |
| OV-DEIM | 0.25 | Top-K 선택을 통한 일대일 매칭, 억제 없음 |

`iou=`는 억제를 수행하는 계열에게만 의미가 있습니다. OMDet-Turbo는 임계값을 인수로 받아들이며, `iou=`가 설정되지 않은 경우 기본값은 0.5입니다. 나머지 세 가지는 아무것도 억제하지 않으므로, 그곳에 `iou=`를 전달하면 경고가 표시되며 무시됩니다.

`text_threshold=`는 Grounding DINO 전용으로, 기본값은 0.25입니다. 지속적인 값을 위해 생성 시 전달할 수 있으며, 호출 시마다 전달할 수도 있습니다. 호출 시마다 전달하는 값은 `stream=True`와 함께 사용할 수 없습니다. 스트리밍 결과가 지연 생성되기 때문입니다. 대신 생성자에서 설정하십시오. 다른 모든 계열은 이를 위해 `TypeError`를 증가시킵니다.

`imgsz=`는 `ValueError`를 발생시킵니다: 전처리 파이프라인이 이 단계의 크기 조정을 담당합니다. `augment=True`도 발생합니다, 여기서 테스트 시 증강은 범위를 벗어나기 때문입니다. 입력 크기는 참고용으로 계열별로 기록됩니다: Grounding DINO 800, OWLv2 960 및 1008, OMDet-Turbo 640, OV-DEIM 640.

## 지원되지 않음

`train()`, `val()`, `track()` 및 `export()` 모두 `NotImplementedError`를 발생시킵니다. 업스트림을 파인튜닝하고 결과 가중치를 로드합니다; 추적 대신 프레임별로 `predict()`를 실행합니다. 검증에는 전용 검증기가 필요합니다. 왜냐하면 공유된 검출 검증기는 모델을 이미지 텐서로 호출하는 반면 이 계층은 텍스트 조건 입력을 요구하기 때문입니다.
