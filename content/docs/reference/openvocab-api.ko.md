---
title: 오픈 보캐뷸러리 API
seo_title: 'LibreOpenVocab API: 별칭과 인자'
description: >-
  LibreOpenVocab 팩토리, 그 네 가지 패밀리와 모든 별칭, set_classes, 패밀리별 기본 설정, 텍스트 임계값과 iou
  규칙.
lead: >-
  LibreOpenVocab은 텍스트 조건 검출기를 위한 팩토리입니다. 클래스 목록은 고정된 헤드가 아니라 프롬프트이므로, 어휘는
  set_classes로 설정되며 모델은 이에 대해 일반적인 검출 결과를 반환합니다.
keywords:
  - LibreOpenVocab
  - 오픈 보캐뷸러리 검출
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: 1.5.0
verification: >-
  libreyolo/models/openvocab/__init__.py에서 읽은 별칭; grounding_dino.py, owlv2.py,
  omdet_turbo.py, ov_deim.py에서 가져온 저장소, 크기 및 임계값;
  libreyolo/models/openvocab/base.py에서 호출 규칙; 모두 v1.5.0 기준. 설계 의도는
  docs/adr/0008-open-vocab-detector-contract.md.에서 확인
snippets:
  install:
    - label: bash
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

계층에는 `openvocab` 추가가 필요합니다.

<code-tabs name="install" />

## 팩토리

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model`는 경로가 아니라 별칭입니다. 조회 전에 밑줄(_)은 하이픈(-)으로 변환되므로, CLI 인벤토리가 출력하는 패밀리-정의 이름인 `omdet_turbo-t` 및 `grounding_dino-t`는 그대로 로드됩니다. 알려지지 않은 별칭은 모든 알려진 별칭을 나열하는 `ValueError`를 발생시킵니다.

생성자는 `size`, `nb_classes=80`, `names=None`, `device="auto"`, `task=None` 및 `text_threshold=None`를 허용합니다. `names`를 전달하는 것은 로드 직후 `set_classes`를 호출하는 것과 동일합니다. 지원하지 않는 패밀리에 `text_threshold`를 전달하면 `TypeError`가 발생합니다.

<code-tabs name="usage" />

## 패밀리 및 별칭

| 패밀리 | 별칭 | 크기 | 무게 |
|---|---|---|---|
| 접지 DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

기본 별칭은 `grounding-dino-tiny`입니다.

`LibreGroundingDINO`, `LibreOWLv2` 및 `LibreOMDetTurbo`는 패키지 수준에서 내보내지며 `size=`로 직접 생성할 수 있습니다. OV-DEIM은 위의 팩토리 별칭을 통해 접근할 수 있습니다.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

이후의 모든 `predict()` 호출에 대한 어휘를 설정하고, 호출이 연쇄될 수 있도록 모델을 반환합니다. 리스트는 비어있지 않아야 하며, 오직 문자열만 포함해야 하고, 대소문자를 구분하지 않고 비교했을 때 항목이 고유해야 하며, 빈 레이블은 허용되지 않습니다. 단일 문자열을 전달하면 `TypeError`가 발생하는데, 이는 한 글자 클래스들로 나뉘게 되기 때문입니다.

호출 후, `model.names`는 `0..N-1`를 주어진 순서대로 레이블에 매핑하며, `model.nb_classes`는 `N`입니다.

## 호출 인수

이 계층은 표준 predict 기능을 재사용하지만 세 가지 차이가 있습니다.

`conf`은 공유된 0.25 대신 계열 자체 값을 기본값으로 사용합니다:

| 계열 | 기본 설정 | 억제 |
|---|---|---|
| DINO 접지 | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | 자체 후처리, 임계값 0.5, `iou=` 준수 |
| OV-DEIM | 0.25 | 상위-K 선택과 1대1 매칭, 억제 없음 |

`iou=`는 억제를 실행하는 계열에게만 의미가 있습니다. OMDet-Turbo는 임계값을 인자로 받아 `iou=`가 설정되지 않은 경우 기본값 0.5를 사용합니다. 나머지 세 개는 아무 것도 억제하지 않으므로, 그곳에 `iou=`를 전달하면 경고가 출력되고 무시됩니다.

`text_threshold=`는 오직 Grounding DINO에만 해당되며, 기본값은 0.25입니다. 지속적인 값을 위해 생성 시 전달할 수 있거나, 호출 시 전달할 수 있습니다. 호출 시 전달한 값은 `stream=True`와 결합할 수 없습니다. 스트리밍된 결과가 지연 생성되기 때문입니다. 대신 생성자에서 설정하십시오. 다른 모든 계열에서는 이에 대해 `TypeError`를 발생시킵니다.

`imgsz=`는 `ValueError`를 발생시킵니다: 이 계층에서 전처리 파이프라인이 크기 조정을 담당합니다. 테스트 시 증강 기능이 범위를 벗어나므로 `augment=True`도 발생합니다. 입력 크기는 참고용으로 계열별로 기록됩니다: Grounding DINO 800, OWLv2 960 및 1008, OMDet-Turbo 640, OV-DEIM 640.

## 지원되지 않음

`train()`, `val()`, `track()` 및 `export()` 모두 `NotImplementedError`를 상승시킵니다. 업스트림을 파인튜닝하고 결과 가중치를 로드하세요; 추적 대신 프레임당 `predict()`를 실행합니다. 검증에는 전용 검증기가 필요합니다. 왜냐하면 공유 검출 검증기는 이미지 텐서로 모델을 호출하지만 이 계층에서는 텍스트 조건 입력이 필요하기 때문입니다.
