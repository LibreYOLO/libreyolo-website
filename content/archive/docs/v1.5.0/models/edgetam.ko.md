---
title: EdgeTAM
families:
  - edgetam
seo_title: 'EdgeTAM: LibreYOLO의 온디바이스 프롬프트 분할'
description: >-
  LibreYOLO에서 온디바이스 속도를 위해 설계된 프롬프트 기반 포인트 및 바운딩 박스 분할에 EdgeTAM을 사용합니다.
  Apache-2.0이 적용된 체크포인트를 설치하고 예측합니다.
lead: >-
  EdgeTAM은 같은 프롬프트 기반 포인트 및 바운딩 박스 워크플로를 유지하면서 모바일 추론 속도를 위해 설계된 SAM 2의 온디바이스
  변형입니다. LibreYOLO는 LibreYOLO() 탐지기 팩토리와 별도의 전용 LibreSAM 팩토리를 통해 이미지 분할 경로를
  지원합니다.
keywords:
  - EdgeTAM 사용법
  - SAM 2
  - 프롬프트 분할
  - 인터랙티브 분할
  - 온디바이스 분할
  - 포인트 프롬프트
  - 바운딩 박스 프롬프트
  - Meta Reality Labs
last_verified: 1.5.0
snippets:
  predict:
    - label: 포인트 및 바운딩 박스 프롬프트
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # EdgeTAM은 "edge" 한 크기만 있습니다. 별칭: "edgetam", "edge-tam",
        # "edgetam-edge".
        model = LibreSAM("edgetam")

        # 포인트 프롬프트: 픽셀 좌표 [x, y], 레이블 1은 전경입니다.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # 마스크별 폴리곤
        print(result.boxes.xyxy)    # 마스크에서 파생된 꼭 맞는 바운딩 박스

        # 포인트 대신 바운딩 박스 프롬프트를 사용합니다.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # 프롬프트가 없으면 전체 이미지를 분할합니다. 단순화된 자동
        # 마스크 생성기이며 전체 기능을 갖춘 참조 구현은 아닙니다.
        result = model.predict(SAMPLE_IMAGE)
    - label: 한 번 인코딩하고 여러 번 프롬프트하기
      language: python
      code: |
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE

        model = LibreEdgeTAM()

        # 이미지 인코더의 비용이 큽니다. set_image()로 한 번 실행하면
        # 이후 모든 predict() 호출이 캐시된 임베딩을 재사용합니다.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: e6cce8faad18e73d
---

## 설치

EdgeTAM에는 `transformers`와 `timm`을 가져오는 `sam` extra가 필요합니다.

```bash
pip install "libreyolo[sam]"
```

## 예측

`LibreSAM(...)` 또는 계열별 `LibreEdgeTAM(...)`은 `LibreYOLO(...)`와 별도의
진입점입니다. 여기서는 공간 프롬프트 없이는 순전파가 의미 없으므로 탐지기가 아니라
프롬프트 가능 분할기를 반환합니다. 이 계열에는 `libreyolo predict` CLI 명령이
없으므로 Python API를 사용합니다. 이미지 분할만 지원하며 EdgeTAM의 비디오 추적은
여기서 범위를 벗어납니다.

<code-tabs name="predict" />

포인트 프롬프트는 객체 하나에 `[x, y]`, 여러 객체에 `[[x, y], ...]` 또는 NumPy
배열을 허용합니다. `labels`는 각 포인트를 `1`(전경) 또는 `0`(배경)으로 표시하며
기본값은 모두 전경입니다. 바운딩 박스 프롬프트는 `[x1, y1, x2, y2]` 또는 바운딩
박스 목록을 받으며 바운딩 박스마다 마스크 하나를 생성합니다. 두 프롬프트를 모두
생략하면 밀집 그리드로 프롬프트하고 신뢰할 수 있으며 겹치지 않는 마스크를 유지하여
전체 이미지를 분할합니다. 이 "모두 분할" 모드는 참조 자동 마스크 생성기보다
단순화되어 혼잡한 장면을 충분히 분할하지 못할 수 있으므로 실제 포인트 또는 바운딩
박스 프롬프트가 정확한 경로입니다. `conf`는 탐지 신뢰도가 아니라 예측한 마스크
품질(IoU)을 기준으로 필터링합니다. 모든 후보를 유지하려면 `0.0`을 전달합니다.
`multimask=True`는 가장 좋은 하나 대신 프롬프트마다 SAM의 전체 객체와 부분 객체
모호성 마스크 3개를 모두 반환합니다. `device=`는 모델을 이동하며 `set_image()`
세션이 활성 상태이면 캐시된 임베딩도 이동합니다. 프롬프트 가능 마스크에는 고정
클래스 집합이 없으므로 모든 마스크에는 이름이 `"object"`인 클래스 ID `0`이
지정됩니다. 이 계열에서 `train()`, `val()`, `export()`, `track()`은 모두
`NotImplementedError`를 일으킵니다. LibreYOLO는 여기서 이미지 추론을 지원합니다.
소스 유형은 [예측](/docs/predict)을 참조합니다.

## 변형

고정 입력 해상도를 사용하는 edge 한 크기만 있으므로, 다른 SAM 계층 대신 이 계열을
선택하는 것은 크기 선택이 아니라 하드웨어 선택입니다. EdgeTAM은 특히 제한된
온디바이스 추론을 위해 존재합니다.

## 체크포인트

이 계열에서 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box></provenance-box>

## 인용

<citation-block />
