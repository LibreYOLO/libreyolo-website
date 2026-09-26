---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'LibreYOLO의 SenseNova-Vision: 체크포인트 하나로 7개 작업'
description: >-
  하나의 프롬프트 기반 생성 체크포인트로 LibreYOLO에서 SenseNova-Vision을 사용하여 탐지, 분할, 판옵틱, 자세, 포인트,
  깊이, OCR을 수행합니다.
lead: >-
  SenseNova-Vision은 비전 작업을 공유 디코더의 프롬프트 기반 생성으로 표현하는 통합 멀티모달 모델입니다. 바운딩 박스, 포인트,
  키포인트, OCR 단어는 태그가 지정된 텍스트로 출력되고 깊이, 마스크, 판옵틱 맵은 디코더가 렌더링하는 이미지로 출력됩니다.
  LibreYOLO는 LibreVLM으로 모델을 불러오며 하나의 7B 체크포인트에서 7개 작업을 지원합니다.
keywords:
  - SenseNova-Vision 사용법
  - SenseTime 멀티모달 모델
  - 통합 멀티모달 모델
  - Bagel
  - 프롬프트 객체 탐지
  - 밀집 인식
  - 참조 분할
  - 판옵틱 분할
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task()는 불러온 같은 모델에서 작업을 전환합니다.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: 참조 분할 및 판옵틱
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="segment")
        # 분할은 참조 방식이며 클래스 목록이 아니라 대상 구문이 필요합니다.
        model.set_classes(["the person furthest to the right"])
        result = model.predict("street.jpg")
        mask = result.masks.data[0]

        model.set_task("panoptic")
        # 사용자 지정 보캐뷸러리가 없으면 판옵틱은 체크포인트가 튜닝된
        # COCO 판옵틱 카테고리를 기본값으로 사용합니다.
        result = model.predict("street.jpg")
        segment_map = result.panoptic.data
        for segment in result.panoptic.segments_info:
            print(segment)
    - label: '포인트, 자세, OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # 보캐뷸러리를 설정하지 않으면 자세 작업은 "person"을 기본값으로 사용합니다.
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## 설치

SenseNova-Vision에는 자체 extra가 필요합니다. 이 extra는 체크포인트의 대형 모델
디스패치에 필요한 `accelerate`를 가져오며, macOS가 아닌 플랫폼에서는 4비트 로드용
`bitsandbytes`도 가져옵니다.

```bash
pip install "libreyolo[sensenova]"
```

체크포인트는 Hugging Face의 LibreYOLO 자체 조직에 미러링되며 처음 사용할 때
자동으로 내려받습니다. CC BY-NC 4.0이 적용되어 비상업적 사용만 허용되며, 로더는
자동으로 내려받기 전에 항상 이 알림을 출력합니다. 아래 라이선스 절을 참조합니다.

## 예측

<code-tabs name="predict" />

모든 예측은 공유 Bagel-MoT 백본에서 확산 디코딩으로 이루어지므로 실시간 모델이
아니라 기능 모델입니다. 전용 탐지기나 분할기보다 이미지당 지연 시간이 눈에 띄게
길어집니다. `dtype="auto"`(기본값)는 메모리가 충분한 GPU에서 bf16으로 불러오고,
그 외 환경에서는 `bitsandbytes`가 필요한 4비트 NF4 양자화로 대체합니다. 충분히 큰
GPU에서 전체 정밀도를 강제하려면 `dtype="bf16"`을 전달합니다. 생성자에서
`noise_seed=42`를 사용하면 재현 가능한 밀집 출력을 위해 확산 샘플러의 시드를
설정하며, 시드를 비활성화하려면 `noise_seed=None`을 전달합니다.

7개 작업은 불러온 체크포인트 하나를 공유합니다. `set_task()`로 다시 불러오지 않고
작업을 전환합니다. `set_classes()`는 활성 보캐뷸러리를 설정합니다. 탐지, 포인트,
자세, 판옵틱은 클래스 목록을 허용하지만 분할은 참조 방식이므로 분리할 정확한 구문이
필요합니다. 각 작업은 서로 다른 페이로드가 채워진 표준 `Results` 객체를 반환합니다.
탐지에는 `boxes`, 포인트에는 `points`, 자세에는 `boxes`와 `keypoints`, OCR에는
`ocr`, 깊이에는 `depth_map`, 분할에는 `masks`, 판옵틱에는 `panoptic`과
`segments_info`가 채워집니다. 소스, 스트리밍, 결과 처리는
[예측](/docs/predict)을 참조합니다.

## 체크포인트

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
