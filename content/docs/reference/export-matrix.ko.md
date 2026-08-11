---
title: 전체 내보내기 매트릭스
seo_title: LibreYOLO 내보내기 지원 매트릭스 및 규칙
description: >-
  LibreYOLO가 패밀리, 작업 및 형식 조합이 내보내지는지 여부를 결정하는 방법: 12가지 형식, 3가지 계층, 대체 규칙 및 동등성
  임계값
lead: >-
  내보내기 지원은 삼중항(패밀리, 작업, 형식)에 대한 조회입니다. 이 페이지에서는 해당 매트릭스의 구조, 명시적 항목이 다루지 않는 셀을
  채우는 규칙, 관심 있는 조합을 조회하는 방법을 설명합니다.
keywords:
  - libreyolo 내보내기 지원
  - 내보내기 매트릭스
  - onnx tensorrt openvino tflite
  - libreyolo formats 명령어
  - 내보내기 동등성 임계값
  - NotImplementedError 내보내기
last_verified: 1.5.0
verification: >-
  libreyolo/export/support.py에서 읽은 형식, 계층, 폴백 순서, 작업 및 패밀리 블록과 NCNN 블록;
  libreyolo/export/exporter.py에서 읽은 별칭 및 공유 인수;
  docs/adr/0011-export-support-tiers.md에서 읽은 계층 정의; docs/export_support.md에서 읽은
  패리티 임계값, 모두 v1.5.0 기준. 조합별 셀은 여기에 전사되지 않음; 아래 스니펫으로 조회하십시오.
snippets:
  usage:
    - label: '매트릭스 조회, 모델 필요 없음'
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: '내보내기, 거절 읽기'
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.export.support import get_support

        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.export(format="onnx"))

        # 호출 전에 확인: 차단된 조합은 사전 점검에서 오류 발생
        # 메시지에 해당 이유가 표시됨
        blocked = get_support("domedetr", "detect", "onnx")
        print(blocked.tier)
        print(blocked.reason)
source_hash: 83de3289634888c6
---

## 매트릭스의 형태

매트릭스는 `(family, task, format)`를 키로 함. 패밀리 키는 모델 레지스트리의 정식 이름이며, 작업 키는 `libreyolo.tasks.TASKS`에서 가져오고, 열두 가지 형식이 있음:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)`는 추가로 두 개의 별칭을 허용합니다: `tensorrt`의 경우 `engine`, 그리고 `tflite`의 경우 `litert`로, 이는 현재 TensorFlow Lite의 이름입니다. 형식과 `.tflite` 접미사는 변경되지 않았습니다.

<code-tabs name="usage" />

셀은 세 가지 키의 함수이기 때문에 전체 그리드는 크고 각 릴리스마다 변경됩니다. 수동으로 작성되지 않고 생성되며 라이브러리 저장소의 `docs/export_support.md`에 존재합니다. 복사본을 읽는 대신 Python 또는 CLI에서 매트릭스를 쿼리하세요.

## 세 가지 계층

| 계층 | 의미 |
|---|---|
| `validated` | 숫자 홀짝은 CI에서 또는 문서화된 야간 실행에서 다룹니다 |
| `available` | 변환이 구현되었지만 숫자 런타임 패리티 증거는 기록되지 않았습니다 |
| `blocked` | 프리플라이트는 추적 전에 이유와 함께 `NotImplementedError`를 발생시킵니다 |

검증되고 사용 가능한 조합은 모두 승인 또는 일반 경고 없이 진행됩니다. 그들의 기록된 증거와 제약 조건은 생성된 문서에서 계속 볼 수 있습니다. 차단된 조합은 종속성 검사, 보정 로딩, 추적 또는 아티팩트 생성 전에 실패합니다.

검증된 항목을 추가하려면 패리티 테스트와 `since` 필드가 필요합니다.

A `SupportEntry`는 네 개의 필드를 포함한다: `tier`, `reason` 문자열, `since` 릴리스, 그리고 `constraint` 문자열. 제약 조건은 통합 시점에서 중요한 부분이다: 체크 마크는 일반적으로 고정 입력 캔버스, 배치 1, FP32, 그리고 명시된 런타임 버전이라는 조건에서만 적용된다.

## 셀이 어떻게 결정되는가

`get_support(family, task, fmt)`는 이 순서로 해결된다. 일치하는 첫 번째 규칙이 적용된다.

1. 알 수 없는 작업이나 12개 형식 외의 형식은 `blocked`를 반환한다.
2. 명시적인 `(family, task, format)` 항목은 기록된 대로 반환된다.
3. 패밀리 전체 차단은 그 패밀리의 이유와 함께 `blocked`를 반환한다.
4. 작업 전체 차단은 그 작업의 이유와 함께 `blocked`를 반환한다.
5. `ncnn`의 경우, NCNN 차단 목록에 있는 패밀리는 `blocked`를 반환한다.
6. `mnn`는 `blocked`를 반환합니다: 이 패밀리와 작업에 대한 런타임 계약이 없습니다.
7. `rknn`는 `blocked`를 반환합니다. 이 버전에서 RKNN은 RK3588에서 시뮬레이터로 테스트된 정확한 탐지 변종에만 제한됩니다: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s 및 PicoDet-s.
8. `tensorrt`와 `openvino`는 `available`를 반환합니다: 변환기 경로는 존재하지만 해당 패밀리와 작업에 대한 런타임 동일성은 기록되지 않았습니다.
9. `tflite`, `paddle`, `coreai` 및 `coreml`는 `blocked`를 각각 다른 이유와 함께 반환합니다.
10. 다른 모든 것은 `available`를 반환합니다: 변환은 구현되어 있지만 수치 런타임 동일성은 기록되지 않았습니다.

8단계에서 10단계까지의 비대칭성은 의도된 것입니다. TensorRT와 OpenVINO는 ONNX로 일반적으로 변환하므로, 목록에 없는 조합도 시도해볼 가치가 있습니다. TFLite, Paddle, Core AI 및 CoreML은 각각 패밀리별 경로가 필요하므로, 목록에 없는 조합은 초대가 아니라 거부를 의미합니다.

## 차단된 작업

명시적인 항목이 없는 모든 패밀리에 대해 이러한 작업은 차단됩니다.

| 작업 | 이유 |
|---|---|
| `ocr` | 지역별 동적 크롭이 있는 두 개의 네트워크는 단일 그래프 내보내기 계약에 맞지 않습니다 |
| `point` | 해당 패밀리는 공유 포인트 히트맵 및 백엔드 피크 디코딩 계약에 연결되어 있지 않습니다 |
| `semantic` | 해당 패밀리는 공유된 밀집 로짓 및 백엔드 argmax 계약에 연결되어 있지 않습니다 |
| `mesh` | 바디-메시 그래프 출력, 메타데이터 및 런타임 계약이 정의되지 않았습니다 |
| `normal` | 해당 패밀리는 고정 캔버스 밀집 유닛 노멀 및 백엔드 재정규화 계약에 연결되어 있지 않습니다 |
| `panoptic` | 파노프틱 내보내기에는 백엔드 런타임 계약이 없습니다 |
| `gaze` | 해당 패밀리는 공유된 두-헤드 로짓 및 백엔드 기대 디코딩 계약에 연결되어 있지 않습니다 |

명시적 항목이 이를 재정의하며, 예를 들어 연결된 시맨틱 패밀리가 여전히 내보내는 방식입니다.

## 차단된 패밀리

| 패밀리 | 차단 이유 |
|---|---|
| `depth_anything3` | 모든 형식; 그 깊이 그래프가 내보내기된 런타임 계약에 없습니다 |
| `domedetr` | 모든 형식. PAQI는 이미지당 쿼리 수를 설정하므로, 추적된 그래프는 추적된 이미지에만 유효합니다. 내보낼 수 있는 DETR에는 D-FINE을 사용하십시오. |
| `eomt` | 런타임 파싱이 없는 인스턴스 및 파노라마 내보내기 |
| `l2cs` | ONNX, TorchScript, ExecuTorch, TensorRT 및 OpenVINO 외의 모든 것 |
| `hrnet` | ONNX, TorchScript, OpenVINO 및 TensorRT 외의 모든 것 |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | 모든 형식; 프롬프트 가능 모델 내보내기는 v1 런타임 계약의 범위 밖입니다. |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | 모든 형식; 오픈 보캐블러리 런타임 내보내기는 v1의 범위 밖입니다. |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | 모든 형식; 생성형 VLM 내보내기는 v1 범위 밖입니다. |

PicoSAM3는 프롬프트 가능 계층에서 예외입니다: 원시 96픽셀 ROI 네트워크를 ONNX로 내보냅니다.

## NCNN에서 차단됨

DETR 스타일 디코더는 NCNN에서 구현하지 않은 샘플링 연산이 필요하므로, 명시적 항목이 없으면 이러한 계열은 `ncnn`에서 차단됩니다: Deformable DETR, DETR, DINO-DETR, D-FINE, LW-DETR, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR 및 EC. 거부된 이름으로 ONNX, OpenVINO, TorchScript 및 TensorRT가 대안으로 언급됩니다.

## 동등성 임계값

검증된 셀은 내보낸 결과물이 기본 모델을 다음 범위 내에서 재현했음을 의미합니다:

| 작업 그룹 | 임계값 |
|---|---|
| 탐지 및 OBB | 일치된 박스 IoU 0.95 이상, 점수 MAE 0.01 이하 |
| 분할 및 파노프틱 | 마스크 IoU 0.95 이상 |
| 포즈 | 네이티브 해상도에서 키포인트 L2 2픽셀 이하 |
| 분류 | 로짓 코사인 0.999 이상 및 top-1 클래스와 동일 |
| 깊이 및 복원 | 네이티브 출력 대비 PSNR 40dB 이상 |
| 표면 법선 | 평균 각도 오차 0.1도 이하 |
| 포인트 | 피크 위치가 한 출력 셀 내에서 동일 |

DETR 쿼리 행은 순서없는 집합이므로 DETR 계열 동등성은 쿼리 행을 위치가 아닌 집합으로 정렬합니다.

## 내보내기

<code-tabs name="export" />

차단된 조합은 사전 점검(preflight)에서 `NotImplementedError`를 발생시키며, 메시지에는 기록된 이유가 표시됩니다. `validated_alternatives(family, task)`는 해당 쌍에 대해 검증된 형식을 반환하며, 이는 거부 옆에 출력하면 유용한 정보입니다.

모든 수출업자가 공유하는 인수는 [모델 API 페이지](/docs/reference/model-api)에 나열되어 있습니다. 형식별 인수는 개별 형식 페이지에 있습니다.

## 제약 조건 읽기

검증된 셀은 형식 일반이 아니라 측정된 한 구성에 대한 주장입니다. `FP32, batch 1, fixed 520x520 input`와 같은 제약 조건 문자열은 해당 모양과 정밀도에서 패리티가 기록되었음을 의미합니다. 다른 해상도나 배치 크기로 내보내더라도 여전히 아티팩트가 생성되지만, 단지 그 수가 나온 구성은 아닙니다.
