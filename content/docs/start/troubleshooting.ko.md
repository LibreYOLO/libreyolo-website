---
title: 문제 해결
seo_title: 일반적인 LibreYOLO 오류 수정
description: >-
  LibreYOLO가 가장 자주 발생시키는 오류, 각 오류의 의미 및 수정 방법. 잘못된 출력을 생성하지만 오류를 발생시키지 않는 두 가지
  실패 포함.
lead: '메시지별로 그룹화된 오류. 마지막 두 항목은 반대 문제를 다룹니다: 코드가 실행되고 그럴듯한 값을 반환하지만 잘못된 경우.'
keywords:
  - libreyolo 오류
  - 모듈을 찾을 수 없음 오류 libreyolo
  - libreyolo cuda 메모리 부족
  - libreyolo NotImplementedError
  - libreyolo 문제 해결
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

오류는 표시되는 텍스트별로 그룹화됩니다. 여기에 메시지가 없으면 [FAQ](/docs/faq)가 실패가 아닌 질문에 답하며 `libreyolo models`는 설치한 것이 실제로 로드할 수 있는 것을 보고합니다.

## 가져오지 않은 패키지를 명명할 때 ModuleNotFoundError

일부 패밀리는 선택적인 추가 기능이 필요합니다. 메시지는 추가 기능 자체보다는 누락된 패키지를 지정하므로, 트레이스백만 보고 문제를 바로 해결하기는 항상 명확하지 않습니다.

`libreyolo models`를 실행하십시오. 의존성이 누락된 모든 패밀리는 해당 패밀리를 활성화하는 정확한 pip 명령과 함께 출력되므로 패키지를 다시 추가 기능과 매핑할 필요가 없습니다. `libreyolo models --json`는 객체와 동일한 내용을 출력합니다.

[설치 페이지](/docs/install)에는 모든 추가 기능과 그 범위가 나와 있습니다.

## ONNX 추론에는 onnxruntime이 필요

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

기본 패키지는 런타임에 의존하지 않습니다. 원하는 런타임은 하드웨어에 따라 다르기 때문입니다. CPU용은 `onnxruntime`를, CUDA용은 `onnxruntime-gpu`를 설치하세요. 두 패키지는 동일한 `onnxruntime` 모듈을 제공하므로 둘 중 하나만 설치하고 둘 다 설치하지 마세요.

## ONNX 모델을 찾을 수 없습니다.

```
FileNotFoundError: ONNX model not found: <path>
```

경로는 스크립트가 아니라 작업 디렉터리를 기준으로 해결됩니다. 또한 export가 다른 곳에 조용히 기록될 때도 나타납니다: `export()`는 기록된 경로를 반환하므로, 이름을 추측하지 말고 반환 값을 캡처하세요.

## train()에서 NotImplementedError 발생

모든 패밀리가 학습되지 않습니다. 일부는 예측, 검증 및 내보내기를 위해 포팅되었으며, 그들의 `train()`는 실행을 시뮬레이션하는 대신 예외를 발생시킵니다.

[FAQ 항목](/docs/faq)에서 그 이유를 설명합니다. 학습 스크립트를 작성하기 전에 특정 패밀리를 확인하려면, 해당 모델 페이지에서 학습 여부를 확인할 수 있습니다.

## export()에서 NotImplementedError 발생

한 패밀리는 작업을 지원할 수 있지만 여전히 이를 내보내지 않을 수 있습니다. EoMT는 사람들이 직면하는 경우입니다: `export()`는 의미적 작업을 수락하고 `segment`와 `panoptic`에 대해 증가시키는데, 이는 쿼리-마스크 런타임 계약이 정의되지 않았기 때문입니다.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

모든 패밀리 페이지에는 어떤 작업과 형식 조합이 검증되었는지 보여주는 내보내기 매트릭스가 있습니다.

## CUDA 메모리 부족

먼저 `batch`를 줄이고, 그 다음 `imgsz`를 줄이세요. 두 가지 모두 크기에 따라 메모리가 대략 변하지만, 배치는 모델이 보는 것을 변경하지 않고 줄일 수 있는 것입니다.

학습이 아니라 검증에서 실패하면, 검증은 자체 배치 크기로 실행되므로 그것도 낮추세요.

Windows에서, 디스플레이 GPU는 메모리 부족이 아닌 무작위 CUDA 오류처럼 보이는 두 번째 실패 모드를 가지고 있습니다: 드라이버가 지정된 시간 초과보다 오래 응답하지 않는 GPU를 재설정하여 실행 중인 작업을 종료합니다. 모니터를 구동하는 카드에서 긴 커널이 이를 발생시킬 수 있습니다.

## 가중치를 다운로드할 수 없습니다.

가중치는 처음 사용할 때 Hugging Face에서 가져와 로컬에 캐시됩니다. [FAQ](/docs/faq)에서는 캐시 위치와 완전히 오프라인으로 실행하는 방법을 다룹니다.

다운로드가 404 오류를 반환하면, 전달한 파일 이름을 확인하십시오. URL은 파일 이름과 작업 접미사를 포함하여 파생되므로, 게시된 체크포인트와 일치하지 않는 이름은 존재하지 않는 URL을 생성합니다. 각 모델 페이지의 체크포인트 표에는 정확한 게시 파일명이 나와 있습니다.

## Windows에서 학습이 멈추거나 재시작됩니다.

Windows에는 `fork`가 없으므로, dataloader 작업자는 스크립트를 다시 가져오는 것으로 시작합니다. `if __name__ == "__main__":` 가드가 없으면 각 작업자가 학습 호출을 다시 실행하게 되며, 이는 데드락이 발생하거나 끝없이 프로세스를 생성하게 됩니다.

```python
def main():
    ...  # 모델을 구축하고 train()을 호출합니다.

if __name__ == "__main__":
    main()
```

`workers=0`를 설정해도 이를 피할 수 있지만, 처리량이 감소합니다. 가드가 더 나은 해결책입니다.

## 오류를 발생시키지 않는 두 가지 실패

이 페이지의 나머지 부분은 오류에 대한 내용입니다. 이 두 가지는 코드가 실행되고 올바르게 보이는 것을 반환하기 때문에 더 문제입니다.

### 단일 결과 인덱싱

`predict()`는 한 장의 이미지에 대해 하나의 `Results`를 반환하고, 여러 장의 이미지에는 리스트를 반환합니다. 단일 이미지 반환을 인덱싱하면 *탐지*가 선택되며, 이미지는 선택되지 않습니다:

```python
result = model.predict("image.jpg")   # a 결과
result.boxes                          # 모든 탐지, 올바름
result[0].boxes                       # 하나의 탐지, 조용히
```

아무 것도 증가하지 않습니다. `Results`를 인덱싱하는 것은 유효한 작업으로, 부분 집합을 반환하기 때문입니다. 리스트 형태를 대상으로 작성된 코드는 조용히 이미지 당 하나의 박스를 보고합니다. 리스트임을 아는 것만 인덱싱하세요.

### 속성으로 측정 지표 읽기

`val()`는 속성 접근이 가능한 객체가 아닌, 측정 지표 이름을 키로 하는 일반 사전을 반환합니다:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # 올바름
metrics.box.map               # AttributeError
```

키는 `metrics/` 및 `speed/`로 네임스페이스가 지정되어 있습니다. 작업에 따라 세트가 다르므로, 생성된 사전을 한번 출력해 보세요.

## 학습 전에 데이터셋 확인

대부분의 학습 실패는 데이터셋 문제입니다. `libreyolo doctor data.yaml`는 검출 데이터셋에 대해 헬스 체크를 실행하고 심각도별로 결과를 보고하며, 이는 첫 번째 에포크부터 트레이스백을 읽는 것보다 더 빠릅니다.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

점검 목록은 [의사 명령](/docs/cli/doctor)을 참조하십시오.
