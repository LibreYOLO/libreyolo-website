---
title: 전체 내보내기 행렬
seo_title: LibreYOLO 내보내기 지원 매트릭스 및 규칙
description: >-
  LibreYOLO가 계열, 작업 및 형식 조합이 내보낼지 여부를 결정하는 방법: 열두 가지 형식, 세 가지 계층, 폴백 규칙 및 패리티
  임계값.
lead: >-
  내보내기 지원은 삼중항(계열, 작업, 형식)에 대한 조회입니다. 이 페이지에서는 해당 매트릭스의 형태, 명시적 항목이 다루지 않는 셀을
  채우는 규칙, 그리고 관심 있는 조합에 대해 이를 조회하는 방법을 설명합니다.
keywords:
  - libreyolo 내보내기 지원
  - 매트릭스 내보내기
  - onnx 텐서RT 오픈비노 tflite
  - libreyolo 명령 형식
  - 내보내기 균형 임계값
  - NotImplementedError 내보내기
last_verified: 1.5.0
verification: >-
  형식, 계층, 대체 순서, 작업 및 계열 블록과 NCNN 블록은 libreyolo/export/support.py에서 읽음; 별칭과 공유
  인수는 libreyolo/export/exporter.py에서 읽음; 계층 정의는
  docs/adr/0011-export-support-tiers.md에서 읽음; 패리티 임계값은 docs/export_support.md에서
  읽음, 모두 v1.5.0 기준. 조합별 셀은 여기서 기록되지 않음; 아래 스니펫으로 조회하십시오.
snippets:
  usage:
    - label: '행렬을 조회하십시오, 모델은 필요 없습니다'
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
    - label: 내보내기하고 거절을 읽다
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.export.support import get_support

        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.export(format="onnx"))

        # 전화하기 전에 확인: 차단된 조합은 비행 전 검사에서 증가합니다
        # 그리고 그 메시지는 이 이유를 담고 있습니다.
        blocked = get_support("domedetr", "detect", "onnx")
        print(blocked.tier)
        print(blocked.reason)
source_hash: 83de3289634888c6
---

## 행렬의 형태

매트릭스는 `(family, task, format)`로 키가 지정됩니다. 계열 키는 모델 레지스트리의 정식 이름이며, 작업 키는 `libreyolo.tasks.TASKS`에서 가져오고, 형식은 열두 가지가 있습니다:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)`는 추가로 두 개의 별칭도 허용합니다: `tensorrt`에 대해 `engine`, 그리고 TensorFlow Lite의 현재 이름인 `tflite`에 대해 `litert`입니다. 형식과 `.tflite` 접미사는 변경되지 않았습니다.

<code-tabs name="usage" />

셀은 세 개의 키의 함수이기 때문에 전체 그리드는 크고 릴리스마다 변경됩니다. 그것은 수작업으로 작성되는 것이 아니라 생성되며, 라이브러리 저장소의 `docs/export_support.md`에 있습니다. 사본을 읽는 대신 Python이나 CLI에서 매트릭스를 조회하십시오.

## 세 단계

| 계층 | 의미 |
|---|---|
| `validated` | 숫자 동등성은 CI나 문서화된 야간 실행에서 다뤄집니다 |
| `available` | 변환이 구현되었지만, 숫자 런타임 동등성 증거는 기록되지 않았습니다 |
| `blocked` | Preflight는 추적하기 전에 이유와 함께 `NotImplementedError`를 제기합니다 |

검증되고 사용 가능한 조합은 모두 승인 또는 일반 경고 없이 진행됩니다. 기록된 증거와 제약 조건은 생성된 문서에서 계속 표시됩니다. 차단된 조합은 종속성 검사, 교정 로딩, 추적 또는 아티팩트 생성 전에 실패합니다.

검증된 항목을 추가하려면 패리티 테스트와 `since` 필드가 필요합니다.

`SupportEntry`에는 네 개의 필드가 있습니다: `tier`, `reason` 문자열, `since` 릴리스, 그리고 `constraint` 문자열. 제약 조건은 통합 시 중요한 부분입니다: 체크 표시가 적용되는 것은 명시된 조건에서만 적용되며, 일반적으로 고정된 입력 캔버스, 배치 1, FP32, 그리고 특정 이름의 런타임 버전입니다.

## 세포가 어떻게 결정되는지

`get_support(family, task, fmt)`는 이 순서대로 처리됩니다. 일치하는 첫 번째 규칙이 적용됩니다.

1. 알 수 없는 작업이나 열두 가지 형식 외의 형식은 `blocked`를 반환합니다.
2. 명시적인 `(family, task, format)` 항목은 기록된 대로 반환됩니다.
3. 계열 전체 차단은 해당 계열의 사유와 함께 `blocked`를 반환합니다.
4. 작업 전체 블록은 해당 작업의 이유와 함께 `blocked`를 반환합니다.
5. `ncnn`의 경우 NCNN 블록 목록에 있는 계열은 `blocked`를 반환합니다.
6. `mnn`는 `blocked`를 반환합니다: 이 계열과 작업에 대한 런타임 계약이 없습니다.
7. `rknn`는 `blocked`를 반환합니다. 이 버전의 RKNN은 RK3588에서 시뮬레이터로 테스트된 정확한 검출 변형인 YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s 및 PicoDet-s로 제한됩니다.
8. `tensorrt`와 `openvino`는 `available`를 반환합니다: 변환기 경로는 존재하지만 해당 계열와 작업에 대한 런타임 균형이 기록되지 않았습니다.
9. `tflite`, `paddle`, `coreai` 및 `coreml`는 각각 나름의 이유와 함께 `blocked`를 반환합니다.
10. 나머지는 `available`로 반환됩니다: 변환은 구현되었지만, 숫자 런타임 동등성은 기록되지 않았습니다.

8단계에서 10단계까지의 비대칭은 의도적입니다. TensorRT와 OpenVINO는 ONNX에서 일반적으로 변환하므로, 목록에 없는 조합도 시도할 가치가 있습니다. TFLite, Paddle, Core AI 및 CoreML은 각각 계열별 경로가 필요하므로, 목록에 없는 조합은 초대가 아니라 거부를 의미합니다.

## 차단된 작업

명시적인 항목이 없는 모든 계열에 대해 이러한 작업은 차단됩니다.

| 작업 | 이유 |
|---|---|
| `ocr` | 동적 지역별 크롭을 사용하는 두 개의 네트워크는 단일 그래프 내보내기 계약에 맞지 않습니다 |
| `point` | 계열은 공유 포인트 히트맵 및 백엔드 피크 디코딩 계약에 연결되어 있지 않습니다 |
| `semantic` | 그 계열은 공유된 밀집 로짓과 백엔드 아그맥스 계약에 연결되어 있지 않습니다 |
| `mesh` | 바디 메시 그래프 출력, 메타데이터 및 런타임 계약이 정의되지 않았습니다 |
| `normal` | 그 계열은 고정 캔버스 밀집 단위 법선 및 백엔드 재정규화 계약에 연결되어 있지 않습니다 |
| `panoptic` | 파노픽 익스포트에는 백엔드 런타임 계약이 없습니다 |
| `gaze` | 계열은 공유된 두 헤드 로짓과 백엔드 기대 디코딩 계약에 연결되어 있지 않습니다 |

명시적인 항목은 이를 무시하며, 예를 들어 유선 의미 체계 계열이 여전히 내보내는 방식이 바로 이것입니다.

## 차단된 계열

| 계열 | 차단됨 |
|---|---|
| `depth_anything3` | 모든 형식; 해당 깊이 그래프는 내보낸 런타임 계약에 없습니다 |
| `domedetr` | 모든 형식. PAQI는 이미지별 쿼리 수를 설정하므로, 추적된 그래프는 추적된 이미지에 대해서만 유효합니다. 내보낼 수 있는 DETR에는 D-FINE을 사용하십시오. |
| `eomt` | 런타임 파싱이 없는 인스턴스 및 판옵틱 내보내기 |
| `l2cs` | ONNX, TorchScript, ExecuTorch, TensorRT 및 OpenVINO 이외의 모든 것 |
| `hrnet` | ONNX, TorchScript, OpenVINO 및 TensorRT 외의 모든 것 |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | 모든 형식; 프롬프트 가능 모델 내보내기는 v1 런타임 계약의 범위에 포함되지 않습니다 |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | 모든 형식; 오픈-보캐뷸러리 런타임 내보내기는 v1의 범위에서 벗어납니다 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | 모든 형식; 생성형 VLM 내보내기는 v1의 범위에 포함되지 않습니다 |

PicoSAM3는 프롬프트 가능한 계층에서 예외입니다: 그것은 원시 96 픽셀 ROI 네트워크를 ONNX로 내보냅니다.

## NCNN으로 차단됨

DETR 스타일 디코더는 NCNN이 구현하지 않는 샘플링 연산이 필요하므로, 명시적인 안내가 없는 한 `ncnn`에서는 다음 계열가 차단됩니다: Deformable DETR, DETR, DINO-DETR, D-FINE, LW-DETR, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR 및 EC. 거부된 이름으로는 ONNX, OpenVINO, TorchScript 및 TensorRT가 대안으로 언급됩니다.

## 동등 임계값

검증된 셀은 내보낸 산출물이 다음 범위 내에서 원본 모델을 재현했음을 의미합니다:

| 작업 그룹 | 임계값 |
|---|---|
| 탐지 및 OBB | 매치된 박스 IoU가 0.95 이상, 점수 MAE가 0.01 이하 |
| 분할 및 파노프틱 | 마스크 IoU 0.95 이상 |
| 자세 | 기본 해상도에서 키포인트 L2가 2픽셀 이하 |
| 분류 | 로짓 코사인 0.999 이상 및 최상위 1위 클래스와 동일 |
| 깊이와 복원 | 원본 출력에 대해 40dB 이상의 PSNR |
| 표면 법선 | 평균 각도 오차 0.1도 미만 |
| 점 | 하나의 출력 셀 내에서 피크 위치가 동일함 |

DETR 쿼리 행은 순서 없는 집합이므로, DETR 계열 패리티는 쿼리 행을 위치적으로가 아니라 집합으로 정렬합니다.

## 내보내기

<code-tabs name="export" />

차단된 조합은 사전 점검에서 `NotImplementedError`를 발생시키며, 메시지에는 기록된 사유가 포함됩니다. `validated_alternatives(family, task)`는 해당 쌍에 대해 검증된 형식을 반환하며, 이는 거부 옆에 출력하면 유용한 정보입니다.

모든 내보내기자가 공유하는 인수는 [모델 API 페이지](/docs/reference/model-api)에 나열되어 있습니다. 형식별 인수는 각 형식 페이지에 있습니다.

## 제약 읽기

검증된 셀은 하나의 측정된 구성에 대한 주장이지, 형식 일반에 대한 주장이 아닙니다. `FP32, batch 1, fixed 520x520 input`와 같은 제약 문자열은 그 형태와 정밀도에서 패리티가 기록되었음을 의미합니다. 다른 해상도나 배치 크기로 내보내더라도 여전히 인공물이 생성되지만, 단지 그 숫자가 나온 구성은 아닙니다.
