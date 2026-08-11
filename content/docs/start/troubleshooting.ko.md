---
title: 문제 해결
seo_title: 일반적인 LibreYOLO 오류 수정
description: >-
  LibreYOLO가 가장 자주 발생시키는 오류, 각 오류가 의미하는 것, 그리고 해결 방법. 오류를 발생시키는 대신 잘못된 출력을 내는 두
  가지 실패도 포함.
lead: '오류는 보는 메시지별로 그룹화됩니다. 마지막 두 항목은 반대 문제를 다룹니다: 실행은 되지만 그럴듯한 값을 반환하고 잘못된 코드입니다.'
keywords:
  - libreyolo 오류
  - modulenotfounderror libreyolo
  - libreyolo CUDA 메모리 부족
  - libreyolo 미구현 오류
  - libreyolo 문제 해결
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

오류는 보는 텍스트별로 그룹화됩니다. 메시지가 여기에 없다면, [FAQ](/docs/faq)는 실패가 아닌 질문에 답변하며, `libreyolo models`는 설치한 프로그램이 실제로 로드할 수 있는 내용을 보고합니다.

## ModuleNotFoundError: 가져오지 않은 패키지 지정

일부 계열는 선택적 추가가 필요합니다. 메시지는 추가가 아니라 누락된 패키지의 이름을 나타내므로, 추적 정보만으로는 항상 수정 방법이 명확하지 않습니다.

`libreyolo models`를 실행하십시오. 의존성이 없는 모든 계열는 해당 의존성을 활성화하는 정확한 pip 명령어와 함께 출력되므로, 패키지를 다시 extra에 매핑할 필요가 없습니다. `libreyolo models --json`는 객체와 동일하게 출력됩니다.

[설치 페이지](/docs/install)에는 모든 추가 기능과 그 내용이 나와 있습니다.

## ONNX 추론에는 onnxruntime이 필요

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

기본 패키지는 런타임에 의존하지 않습니다. 어떤 것을 사용할지는 하드웨어에 따라 달라집니다. CPU용으로는 `onnxruntime`를 설치하고, CUDA용으로는 `onnxruntime-gpu`를 설치하십시오. 두 패키지 모두 동일한 `onnxruntime` 모듈을 제공하므로 하나만 설치하고 둘 다 설치하지 마십시오.

## ONNX 모델 탐색 실패

```
FileNotFoundError: ONNX model not found: <path>
```

경로는 스크립트가 아닌 작업 디렉토리를 기준으로 해석됩니다. 이는 export가 다른 곳에 조용히 쓸 때에도 나타납니다: `export()`는 쓴 경로를 반환하므로, 이름을 가정하지 말고 반환 값을 캡처하십시오.

## train()에서 NotImplementedError

모든 계열이 학습을 하는 것은 아닙니다. 일부는 예측, 검증 및 내보내기 용도로만 포팅되며, 그들의 `train()`는 실행하는 척하기보다는 오히려 상승합니다.

[FAQ 항목](/docs/faq)에서 이유를 설명합니다. 학습 스크립트를 작성하기 전에 특정 계열를 확인하려면, 해당 모델 페이지에서 학습 여부를 확인할 수 있습니다.

## export()에서 NotImplementedError

한 계열이 작업을 지원할 수 있지만 여전히 그것을 내보내지 않을 수 있습니다. EoMT는 사람들이 부딪히는 경우입니다: `export()`는 의미론적 작업을 수락하고 `segment`와 `panoptic`에 대해 호출을 발생시키는데, 이는 그들이 필요한 쿼리-마스크 런타임 계약이 정의되지 않았기 때문입니다.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

각 계열의 페이지에는 어떤 작업과 형식 조합이 검증되었는지를 보여주는 내보내기 행렬이 표시됩니다.

## CUDA 메모리 부족

먼저 `batch`를 줄이고, 그 다음 `imgsz`를 줄이십시오. 둘 다 자신의 크기에 따라 메모리를 대략적으로 변경하지만, 배치는 모델이 보는 것을 바꾸지 않고 줄일 수 있는 것입니다.

만약 학습이 아니라 검증에서 실패하면, 검증은 자체 배치 크기로 실행되므로 그것도 낮춰야 합니다.

Windows에서는 디스플레이 GPU에 또 다른 고장 모드가 있으며, 이는 메모리 부족보다는 무작위 CUDA 오류처럼 보입니다. 드라이버가 일정 시간 이상 응답하지 않는 GPU를 재설정하여 실행 중이던 작업을 중단시키기 때문입니다. 모니터를 구동하는 카드에서 긴 커널이 이를 유발할 수 있습니다.

## 가중치 다운로드 실패

가중치는 처음 사용할 때 Hugging Face에서 가져와 로컬에 캐시됩니다. [FAQ](/docs/faq)에는 캐시 위치와 완전히 오프라인으로 실행하는 방법이 나와 있습니다.

다운로드가 404 오류가 발생하면 전달한 파일 이름을 확인하십시오. URL은 해당 파일 이름에서 파생되며, 작업 접미사도 포함되므로 게시된 체크포인트와 일치하지 않는 이름은 존재하지 않는 URL을 생성합니다. 각 모델 페이지의 체크포인트 표에는 정확한 게시된 파일 이름이 나와 있습니다.

## Windows에서 학습이 중단되거나 다시 시작됨

Windows에는 `fork`가 없기 때문에 dataloader 작업자는 스크립트를 다시 가져오는 것으로 시작합니다. `if __name__ == "__main__":` 보호가 없으면 각 작업자가 학습 호출을 다시 실행하게 되며, 이는 교착 상태를 일으키거나 끝없이 프로세스를 생성하게 됩니다.

```python
def main():
    ...  # 모델을 구축하고 train()을 호출하십시오

if __name__ == "__main__":
    main()
```

`workers=0`를 설정하는 것도 이를 피할 수 있지만, 처리량에는 비용이 듭니다. 가드가 더 나은 해결책입니다.

## 성장시키지 않는 두 가지 실패

이 페이지의 나머지 부분은 오류에 관한 것입니다. 이 두 가지는 더 심각합니다, 왜냐하면 코드가 실행되고 겉보기에는 맞아 보이는 결과를 반환하기 때문입니다.

### 단일 결과 인덱싱

`predict()`는 하나의 이미지에 대해 하나의 `Results`를 반환하고, 여러 이미지에 대해서는 리스트를 반환합니다. 단일 이미지 반환값에서 인덱싱하면 이미지가 아니라 *탐지(detection)*를 선택합니다:

```python
result = model.predict("image.jpg")   # 결과
result.boxes                          # 모든 탐지, 정확함
result[0].boxes                       # 원 검출, 조용히
```

아무것도 올라가지 않습니다. `Results`를 인덱싱하는 것은 유효한 연산으로, 부분 집합을 반환합니다. 리스트 형태에 대해 작성된 코드는 조용히 이미지당 하나의 박스를 보고합니다. 알고 있는 리스트만 인덱싱하십시오.

### 속성으로서의 읽기 지표

`val()`는 속성 접근이 가능한 객체가 아니라, 메트릭 이름을 키로 하는 일반 딕셔너리를 반환합니다:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # 정확한
metrics.box.map               # 속성 오류
```

키는 `metrics/`와 `speed/` 네임스페이스가 지정되어 있습니다. 작업에 따라 집합이 다르므로 작업이 생성한 내용을 확인하려면 사전을 한 번 출력해 보십시오.

## 학습하기 전에 데이터셋 확인하기

대부분의 학습 실패는 데이터셋 문제입니다. `libreyolo doctor data.yaml`는 검출 데이터셋에 대해 상태 점검을 수행하고 심각도별로 결과를 보고하는데, 이는 첫 번째 에포크부터 트레이스백을 읽는 것보다 더 빠릅니다.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

검사 카탈로그는 [doctor command](/docs/cli/doctor)를 참조하십시오.
