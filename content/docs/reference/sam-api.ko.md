---
title: 프롬프트 가능한 세분화 API
seo_title: 'LibreSAM API: 프롬프트, 별칭 및 서명'
description: >-
  LibreSAM 팩토리, 크기 별칭, 포인트, 박스 및 개념-텍스트 프롬프트 유형, encode-once set_image 생명주기, 및
  계층에서 지원하지 않는 사항
lead: >-
  LibreSAM은 프롬프트 가능한 세분화를 위한 팩토리입니다. 순방향 패스는 호출 시 각 이미지에 대한 프롬프트가 필요하므로, 계층은
  프롬프트 없는 추론 실행기를 통하지 않고 자체 예측 영역을 가집니다.
keywords:
  - LibreSAM
  - 프롬프트 가능한 세분화
  - SAM 포인트 프롬프트
  - SAM 박스 프롬프트
  - set_image
  - 모든 것 세분화
  - libreyolo sam 추가
last_verified: 1.5.0
verification: >-
  팩토리 별명, 크기 및 저장소는 libreyolo/models/sam/model.py, sam2.py, edgetam.py, sam3.py,
  libreyolo/models/mobilesam/model.py 및 libreyolo/models/picosam3/model.py.에서
  읽습니다. 계약 및 기본값 프롬프트는 libreyolo/models/sam/base.py.에서 읽습니다. 설계 의도는
  docs/adr/0007-libresam-contract.md에서 읽으며, 모두 버전 v1.5.0입니다.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: 포인트 및 박스 프롬프트
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: '한 번 인코딩, 여러 번 프롬프트'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## 설치

이 등급에는 `sam`가 추가로 필요합니다.

<code-tabs name="install" />

## 팩토리

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model`은 경로가 아닌 크기 별명입니다. `**kwargs`는 계열 생성자에 도달하며, 이 생성자는 `device` 및 `multimask`를 받습니다. 알 수 없는 별명이 있으면 `ValueError`가 발생하며, 메시지에는 모든 알려진 별명이 나열됩니다.

<code-tabs name="usage" />

## 별명

| 계열 | 별명 | 크기 | 가중치 |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, 그리고 단축형 `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

기본값은 `base`입니다. SAM-1, SAM-2, EdgeTAM 및 MobileSAM은 명목상 1024 픽셀 캔버스에서 실행되며, SAM 3은 1008, PicoSAM3는 96에서 실행됩니다.

SAM 3 가중치는 게이트 처리되어 있습니다. 이들은 Meta의 맞춤 SAM 라이선스 하에 `facebook/sam3`에서 다운로드되며, 이는 MIT도 아니고 Apache-2.0도 아니며 LibreYOLO에 의해 재배포되지 않습니다. 로드하기 전에 저장소 페이지에서 조건에 동의하고 Hugging Face로 인증하십시오; 로더는 먼저 이 알림을 기록합니다.

계열 클래스도 내보내지므로 `LibreSAM1`, `LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` 및 `LibrePicoSAM3`를 `size=`로 직접 생성할 수 있습니다.

## 예측

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| 인수 | 기본값 | 의미 |
|---|---|---|
| `source` | `None` | 분할할 이미지; `None`는 `set_image()`가 캐시한 이미지를 재사용합니다 |
| `points` | `None` | 픽셀 좌표로 된 포인트 프롬프트 |
| `bboxes` | `None` | `[x1, y1, x2, y2]`로 박스 프롬프트, 또는 박스당 하나의 마스크를 위한 리스트 |
| `labels` | `None` | 포인트 레이블, `1`는 긍정, `0`는 부정, `points`와 맞도록 형상화; 생략 시 모두 긍정 |
| `masks` | `None` | 예약됨; 하나를 전달하면 `NotImplementedError` 증가 |
| `text` | `None` | 개념 프롬프트; SAM 3에만 해당 |
| `conf` | `None` | 예측된 마스크-IoU 하한 |
| `multimask` | `None` | 프롬프트별 모든 모호 마스크 반환; 기본값은 구성 설정 |
| `max_det` | `300` | 반환되는 마스크 제한 |
| `device` | `None` | 이 및 이후 호출에 대해 모델 이동, 캐시된 임베딩 무효화 |
| `color_format` | `"auto"` | 메모리 내 배열의 색상 형식 힌트 |
| `points_per_side` | `None` | 세그먼트 전용 그리드 밀도; 기본값은 32 |

반환은 일반적인 `Results`로 `masks`을 운반하며, 그 마스크에서 유래된 꽉 찬 `boxes`와 `"object"`라는 이름의 클래스 `0`가 추가됩니다.

## 프롬프트 형태

`points`는 한 객체에 대해 `[x, y]`, N 객체에 대해 `[[x, y], ...]`, 객체별로 그룹화된 점들에 대해 `[[[x, y], ...], ...]`와 같은 중첩된 형태를 허용합니다. Numpy 배열은 리스트가 사용될 수 있는 모든 곳에서 작동합니다. 좌표는 원본 이미지 상의 일반 픽셀입니다.

모든 공간 프롬프트를 생략하면 segment-everything이 실행되며, 이는 예측 IoU 임계값과 박스-IoU 중복 제거가 있는 그리드 자동 마스크 생성기입니다. 기본 `points_per_side` 값 32는 약 1024 디코더 패스를 실행하며, 이는 CPU에서는 느립니다; 상호 작용용으로는 낮추는 것이 좋습니다. 생성기는 안정성 점수 필터링, 다중 크롭, 마스크-IoU 중복 제거를 생략하므로, 이는 프롬프트 경로의 근사치이며 완벽한 일치는 아닙니다.

## 신뢰도

`conf`는 탐지 신뢰도가 아닌 마스크 품질 점수인 예측 마스크-IoU로 필터링합니다. `None`는 프롬프트 경로의 모든 마스크를 유지하고 segment-everything에서 계열 그리드 임계값을 적용합니다. `0.0`는 어느 모드에서도 필터링을 비활성화합니다.

SAM 3의 텍스트 경로에서는, `conf`가 대신 프롬프트 가능한 개념 분할 탐지 점수입니다. `None`는 표준 0.3 임계값을 의미하며, `0.0`는 모든 후보를 유지합니다.

## 텍스트 프롬프트

`text=`는 SAM 3 전용입니다; 모든 공간 프롬프트 계열은 이를 위해 `NotImplementedError`를 올립니다. 텍스트는 포인트 및 박스와 상호 배타적입니다. 반환된 `names`는 클래스 `0`를 요청된 개념에 매핑합니다. `source=None`와 함께한 텍스트 호출은 캐시된 이미지를 다시 인코딩합니다. 이는 트래커와 개념 인코더가 캐시를 공유하지 않기 때문입니다.

키워드 `exemplars=`는 향후 이미지-샘플 확장을 위해 예약되어 있으며 구현되지 않았습니다.

## 한 번 인코딩 수명 주기

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image`는 무거운 이미지 인코더를 한 번 실행하고 임베딩을 캐시하므로, 이후의 모든 `predict()`는 `source=None`와 함께 저렴하게 실행됩니다. 두 방법 모두 모델을 반환하므로 호출을 체인으로 연결할 수 있습니다. `device=`를 `predict`에 전달하면 모델이 이동하고 캐시가 무효화됩니다.

## PicoSAM3

PicoSAM3는 `bboxes=`만 허용합니다. 포인트, 텍스트, 마스크, 멀티마스크 및 세그먼트-모두 프롬프트는 오류를 발생시킵니다. 박스는 10% 확장되고 96픽셀 ROI 네트워크를 통해 실행되며, PicoSAM3는 ONNX로만 내보내는 계층의 유일한 계열입니다.

## 지원되지 않음

`train()`, `val()` 및 `track()`는 계층의 모든 계열에서 `NotImplementedError`를 증가시킵니다. 프롬프트 가능한 마스크는 점수를 매길 고정된 클래스 세트가 없으므로 mAP는 여기서 의미가 없습니다. `export()`는 SAM-1, SAM-2, SAM 3, EdgeTAM 및 MobileSAM에 대해 증가합니다.

SAM-2, SAM 3 및 EdgeTAM용 비디오 및 메모리 경로는 이 버전의 범위에 포함되지 않으며, SAM 3 이미지 예시 및 마스크 프롬프트도 마찬가지입니다.
