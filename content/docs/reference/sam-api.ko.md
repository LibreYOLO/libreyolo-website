---
title: 프롬프트 가능한 분할 API
seo_title: 'LibreSAM API: 프롬프트, 별칭 및 서명'
description: >-
  LibreSAM 팩토리, 그 크기 별칭, 포인트, 박스 및 개념-텍스트 프롬프트 유형, 한 번 인코딩하는 set_image 수명 주기,
  그리고 해당 등급에서 지원하지 않는 것.
lead: >-
  LibreSAM은 프롬프트 가능한 세분화를 위한 팩토리입니다. 순전파(forward pass)에는 호출 시에 이미지별 프롬프트가 제공되어야
  하므로, 해당 계층은 프롬프트 없는 추론 러너를 통해 경로를 우회하지 않고 자체 예측 표면을 소유합니다.
keywords:
  - 리브레샘
  - 프롬프트 가능 분할
  - SAM 포인트 프롬프트
  - SAM 박스 프롬프트
  - 이미지 설정
  - 모든 것을 분할하다
  - 리브레욜로 샘 엑스트라
last_verified: 1.5.0
verification: >-
  팩토리 별칭, 크기 및 저장소는 libreyolo/models/sam/model.py, sam2.py, edgetam.py, sam3.py,
  libreyolo/models/mobilesam/model.py 및 libreyolo/models/picosam3/model.py.에서
  읽습니다. 프롬프트 계약 및 기본값은 libreyolo/models/sam/base.py.에서 읽습니다. 설계 의도는
  docs/adr/0007-libresam-contract.md에서 읽으며, 모두 v1.5.0 기준입니다.
snippets:
  install:
    - label: 배시
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
    - label: 한 번 인코딩하고 여러 번 프롬프트하기
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

그 계층에는 `sam` 추가가 필요합니다.

<code-tabs name="install" />

## 팩토리

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model`는 경로가 아니라 크기 별칭입니다. `**kwargs`는 `device`와 `multimask`를 받는 계열 생성자에 도달합니다. 알 수 없는 별칭은 `ValueError`를 발생시키며 메시지는 모든 알려진 별칭을 나열합니다.

<code-tabs name="usage" />

## 별명

| 계열 | 별명 | 사이즈 | 무게 |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, 그리고 약식 `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| 엣지TAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| 모바일SAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

기본값은 `base`입니다. SAM-1, SAM-2, EdgeTAM 및 MobileSAM은 명목상 1024 픽셀 캔버스에서 작동하며, SAM 3는 1008, PicoSAM3는 96에서 작동합니다.

SAM 3 가중치는 제한되어 있습니다. 이들은 Meta의 맞춤형 SAM 라이선스 하에 `facebook/sam3`에서 다운로드되며, 이 라이선스는 MIT나 Apache-2.0이 아니며 LibreYOLO에서 재배포되지 않습니다. 로드하기 전에 저장소 페이지에서 약관에 동의하고 Hugging Face로 인증해야 합니다; 로더는 먼저 해당 공지를 기록합니다.

계열 클래스도 내보내지므로, `LibreSAM1`, `LibreSAM2`, `LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` 및 `LibrePicoSAM3`를 `size=`로 직접 생성할 수 있습니다.

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

| 논쟁 | 기본값 | 의미 |
|---|---|---|
| `source` | `None` | 분할할 이미지; `None`는 `set_image()`가 캐시한 이미지를 재사용합니다 |
| `points` | `None` | 픽셀 좌표로 지점 프롬프트 |
| `bboxes` | `None` | 상자 프롬프트를 `[x1, y1, x2, y2]`로 하거나, 상자당 하나의 마스크를 위해 그것들의 목록으로 하십시오 |
| `labels` | `None` | 포인트 레이블, `1` 양성 및 `0` 음성, `points`에 맞게 형성; 생략 시 모두 양성 |
| `masks` | `None` | 예약됨; 하나를 통과하면 `NotImplementedError`가 올라갑니다 |
| `text` | `None` | 개념 프롬프트; SAM 3만 |
| `conf` | `None` | 예측된 마스크-IoU 최저값 |
| `multimask` | `None` | 프롬프트마다 모든 모호성 마스크를 반환합니다; 기본값은 건설 설정입니다 |
| `max_det` | `300` | 반환된 마스크 상한 |
| `device` | `None` | 향후 호출과 이 작업을 위해 모델을 이동하여 캐시된 임베딩을 무효화합니다 |
| `color_format` | `"auto"` | 메모리 내 배열을 위한 색상 형식 힌트 |
| `points_per_side` | `None` | segment-everything의 그리드 밀도; 기본값은 32 |

반환은 `masks`를 운반하는 일반적인 `Results`이며, 해당 마스크에서 유래된 타이트 `boxes`와 `"object"`라는 이름의 `0` 클래스를 포함합니다.

## 프롬프트 모양

`points`는 한 객체에 대해서는 `[x, y]`, N 객체에 대해서는 `[[x, y], ...]`, 객체별로 그룹화된 점들에 대해서는 `[[[x, y], ...], ...]` 형태를 수용합니다. Numpy 배열은 리스트가 작동하는 모든 곳에서 작동합니다. 좌표는 원본 이미지의 일반 픽셀입니다.

모든 공간 프롬프트를 생략하면 세그먼트-에브리씽(segment-everything)이 실행되며, 이는 예측 IoU 임계값과 박스 IoU 중복 제거를 갖춘 그리드 자동 마스크 생성기입니다. 기본 `points_per_side` 값 32는 대략 1024번의 디코더 패스를 실행하며, 이는 CPU에서는 느립니다. 대화형 사용을 위해 값을 낮추십시오. 생성기는 안정성 점수 필터링, 다중 크롭, 마스크 IoU 중복 제거를 생략하므로, 이는 프롬프트 경로의 근사치일 뿐 실제와 일치하지는 않습니다.

## 자신감

`conf`는 마스크 품질 점수인 예측 마스크-IoU에 따라 필터링하며, 이는 검출 신뢰도가 아닙니다. `None`는 프롬프트된 경로의 모든 마스크를 유지하고 segment-everything에서 family grid 임계값을 적용합니다. `0.0`는 두 모드 중 어느 쪽에서도 필터링을 비활성화합니다.

SAM 3의 텍스트 경로에서 `conf`는 대신 Promptable Concept Segmentation 탐지 점수입니다. `None`는 표준 0.3 임계값을 의미하고, `0.0`는 모든 후보를 유지합니다.

## 텍스트 프롬프트

`text=`는 SAM 3 전용입니다; 모든 공간-프롬프트 계열는 이를 위해 `NotImplementedError`를 발생시킵니다. 텍스트는 포인트 및 박스와 상호 배타적입니다. 반환된 `names`는 클래스 `0`를 요청된 개념과 매핑합니다. `source=None`가 있는 텍스트 호출은 캐시된 이미지를 다시 인코딩합니다. 이는 트래커와 개념 인코더가 캐시를 공유하지 않기 때문입니다.

키워드 `exemplars=`는 미래의 이미지 예시 확장을 위해 예약되어 있으며 구현되지 않았습니다.

## 한 번 인코딩하는 수명 주기

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image`는 무거운 이미지 인코더를 한 번 실행하고 임베딩을 캐시하므로, 이후의 모든 `predict()`를 `source=None`와 함께 사용하는 비용이 적습니다. 두 가지 방법 모두 모델을 반환하므로 호출을 연결할 수 있습니다. `device=`를 `predict`에 전달하면 모델이 이동하고 캐시가 무효화됩니다.

## PicoSAM3

PicoSAM3는 `bboxes=`만 허용합니다. 점, 텍스트, 마스크, 멀티마스크 및 모든 세그먼트 프롬프트가 상승합니다. 상자는 10퍼센트 확장되고 96픽셀 ROI 네트워크를 통해 실행되며, PicoSAM3는 ONNX로만 내보내는 계층에서 하나의 계열입니다.

## 지원되지 않음

`train()`, `val()` 및 `track()`는 해당 티어의 모든 계열에서 `NotImplementedError`를 발생시킵니다. 프롬프트 가능한 마스크는 점수를 매길 고정된 클래스 세트가 없으므로, 여기서 mAP는 의미가 없습니다. `export()`는 SAM-1, SAM-2, SAM 3, EdgeTAM 및 MobileSAM에서 상승합니다.

SAM-2, SAM 3 및 EdgeTAM의 비디오 및 메모리 경로는 이 버전의 범위에 포함되지 않으며, SAM 3 이미지 예시 및 마스크 프롬프트도 역시 포함되지 않습니다.
