---
title: 체크포인트와 가중치
seo_title: LibreYOLO 체크포인트와 가중치
description: >-
  LibreYOLO가 모델 가중치를 찾고, 다운로드하며 검증하는 방법, 가중치가 호스팅되는 위치, 네트워크 없이 실행하는 방법, 안전하게
  체크포인트를 로드하는 요소
lead: >-
  LibreYOLO 체크포인트는 state dict와 그것을 식별하는 데 필요한 메타데이터를 담고 있는 torch.save 사전입니다. 이
  페이지에서는 파일 출처, 위치, 로드 방법을 다룹니다.
keywords:
  - libreyolo 가중치
  - libreyolo 체크포인트
  - libreyolo 가중치 다운로드
  - libreyolo 오프라인
  - libreyolo 허깅 페이스
  - 체크포인트 메타데이터
last_verified: 1.5.0
meta:
  - label: 호스팅 위치
    value: '체크포인트당 하나의 Hugging Face 저장소:'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: 로컬 캐시
    value: weights/ under the working directory
    mono: true
  - label: 메타데이터 스키마
    value: v1.0
snippets:
  load:
    - label: 자동 다운로드
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 단순 파일 이름은 weights/LibreYOLO9t.pt로 해석되며 이미 존재하지 않는 경우 해당 위치에서 다운로드됩니다.
        # 다운로드됨.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: 명시적 경로
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 디렉터리 구성 요소가 포함된 경로는 작성된 그대로 사용되며 네트워크에서 가져오지 않습니다.
        # 절대로 가져오지 않음.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # 모델을 구성하지 않고 메타데이터를 읽으며, 스키마를 만족하는지 여부를 보고합니다.
        # 보고함.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # 문제 목록을 반환합니다. 비어 있으면 파일이 v1.0을 만족함을 의미합니다.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## 체크포인트가 검색되는 위치

디렉터리 구성 요소가 없는 모델 참조(예: `LibreYOLO9t.pt`)는 현재 작업 디렉터리를 기준으로 `weights/`에 대해 해석됩니다. `weights/LibreYOLO9t.pt`가 존재하면 사용됩니다; 작업 디렉터리 자체에 해당 이름의 파일이 존재하면 대신 사용됩니다; 그렇지 않으면 `weights/LibreYOLO9t.pt`가 다운로드 대상이 됩니다.

디렉토리(절대 또는 상대 경로)가 포함된 참조는 문자 그대로 사용됩니다. 이는 가중치가 중앙에 위치하고 아무 것도 가져오지 않아야 할 때 사용하는 형식입니다.

<code-tabs name="load" />

## 자동 다운로드

해결된 경로가 존재하지 않을 때, LibreYOLO는 파일명을 분석하여 패밀리, 크기 및 작업을 복구하고, 해당 패밀리에서 다운로드 URL을 요청합니다. 대부분의 패밀리는 Hugging Face의 LibreYOLO 조직에서 이를 구축하며, 각 체크포인트는 파일 이름을 따서 자신의 저장소를 갖습니다:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

데이터셋 변형 접미사는 저장소 이름의 일부로 남아 있으므로, 패밀리 기본값과 다르게 학습된 체크포인트는 기본값을 덮어쓰지 않고 자체 저장소로 해결됩니다.

전송 자체는 방어적입니다. 잘린 가중치 파일은 나중에 도움이 되지 않는 오류로 실패하기 때문입니다. 다운로드는 `.part` 파일로 스트리밍되며, 완료된 경우에만 원자적으로 제자리에 이동되므로 중단된 과정이 최종 경로에 반쯤 작성된 체크포인트를 남길 수 없습니다. 중단된 전송은 HTTP 검증기를 사용하여 바이트 오프셋부터 재개되며, 서버가 객체가 변경되었다고 표시하면 처음부터 다시 시작됩니다. 실패는 지수 백오프로 세 번 재시도됩니다. 동일한 경로를 목표로 하는 동시 프로세스는 잠금 파일을 가져가므로, 두 개의 학습 실행이 동시에 시작되면 한번만 다운로드됩니다. 계열이 LibreYOLO 조직이 아닌 제3자 호스트에서 가져오는 경우, 체크섬을 고정하고 불일치 시 파일을 거부할 수 있습니다.

`HF_TOKEN`가 설정되어 있거나 `~/.cache/huggingface/token`에 토큰이 캐시되어 있는 경우, 이는 베어러 토큰으로 첨부됩니다. 이는 `huggingface.co` URL에만 첨부되므로, 다른 호스트에서 다운로드하는 패밀리는 절대 받지 않습니다.

모든 패밀리가 자동으로 다운로드하는 것은 아닙니다. 일부는 공개된 가중치를 재배포할 수 없기 때문에 의도적으로 URL을 반환하지 않으며, 그 경우 오류에서 대신 제공해야 할 내용을 설명합니다. 다른 일부는 전송이 시작되기 전에 라이선스 공지를 출력합니다. 그 공지는 체크포인트의 조건이 코드보다 더 제한적이라는 런타임 신호이며, 스크롤해서 지나가기보다는 읽을 가치가 있습니다.

## Hugging Face 조직

게시된 가중치는 [huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO)에서 실시간으로 확인할 수 있으며, 체크포인트마다 별도의 저장소가 있습니다. 각 저장소에는 라이선스가 있으며, 한 패밀리 안에서도 라이선스가 동일하지 않을 수 있습니다: 코드는 MIT 라이선스인 패밀리라도 일부 가중치는 아닐 수 있습니다. 저장소가 권위 있는 출처입니다. 각 모델 페이지에는 해당 패밀리의 게시된 체크포인트와 라이선스가 Checkpoints 및 Licensing 섹션 아래에 나와 있습니다.

## 오프라인 작업

파일이 로컬에 있으면 라이브러리 자체는 네트워크 접근을 요구하지 않습니다. 두 가지 접근 방식이 있습니다:

작업이 실행되는 위치 옆에 `weights/` 디렉토리를 미리 채워둡니다. 연결된 기기에서 한 번 체크포인트를 가져온 다음 디렉토리를 복사하는 것으로 충분하며, 위의 확인 단계에서 이를 찾아 네트워크에 접속하지 않습니다.

또는 공유 위치에 대한 절대 경로를 전달할 수 있습니다. 디렉터리 구성 요소가 있는 참조는 그대로 사용되므로, 관리된 가중치의 읽기 전용 마운트도 유효한 설정입니다. 프로세스가 변환해야 하는 체크포인트 옆에 쓸 수 없는 경우, 변환은 실패하는 대신 개인 임시 디렉터리로 대체됩니다.

데이터셋은 별도의 규칙을 따릅니다: `~/datasets` 하에서 해결되거나, 해당 변수가 설정된 경우 `LIBREYOLO_DATASETS_DIR`로 명명된 디렉터리 아래에서 해결됩니다.

## 안전성 로딩

체크포인트는 피클이며, 피클은 열릴 때 임의의 코드를 실행할 수 있습니다. LibreYOLO는 모든 가중치 파일을 신뢰할 수 없는 것으로 간주하고 PyTorch의 `weights_only=True` 경로로 로드하며, 이는 언피클러가 텐서와 소수의 안전한 타입만 사용할 수 있도록 제한합니다. 이는 LibreYOLO가 다운로드한 파일뿐만 아니라 사용자가 전달하는 파일에도 적용됩니다. 해당 인수를 지원하지 않는 너무 오래된 PyTorch 빌드에서는 안전하지 않게 수행되지 않고 로드가 거부됩니다.

일부 상위 학습 체크포인트는 제한된 언픽클러가 거부하는 객체를 포함합니다. 예를 들어, 학습을 수행한 프레임워크의 구성 객체가 있습니다. 이러한 객체는 LibreYOLO에 필요하지 않은 메타데이터이므로, 변환 과정에서 각 차단된 클래스는 아무 것도 실행하지 않으면서 언픽클러를 만족시키는 비활성 대체물로 교체되고, 변환된 파일에는 텐서만 남습니다. 민감한 모듈 이름은 스텁 처리되지 않고 아예 거부되며, 재시도 루프는 제한되어 있어서 차단된 클래스가 무한히 반복되도록 설계된 파일이 실패하도록 합니다. 나머지 경로는 [기존 가중치 가져오기](/docs/migrate)를 참조하십시오.

## 체크포인트 메타데이터

LibreYOLO 체크포인트는 `model` 키가 PyTorch 상태 사전을 보유하는 딕셔너리입니다. 스키마 v1.0에서는 9개의 키가 필요하며, 이 키들을 통해 팩토리는 이름을 분석하거나 텐서 모양에서 추측하지 않고 파일을 식별할 수 있습니다.

| 키 | 의미 |
|---|---|
| `model` | PyTorch 상태 사전 |
| `schema_version` | 메타데이터 계약 버전. v1.0은 문자열 `1.0` 사용 |
| `libreyolo_version` | 파일을 생성한 LibreYOLO 버전 |
| `model_family` | 등록된 패밀리 식별자, 예: `yolo9` |
| `size` | 해당 패밀리 내 변형, 예: `t` 또는 `r18` |
| `task` | 하나의 표준 작업 이름 |
| `nc` | 양의 클래스 수 |
| `names` | 클래스 인덱스를 레이블에 매핑한 것으로, `0`부터 `nc - 1`까지 포함합니다. |
| `imgsz` | 양의 입력 해상도 |

추가 구조가 있는 작업은 해당 키와 함께 기록됩니다. Pose 체크포인트는 `num_keypoints`와 `keypoint_dim`를 추가하며, 키포인트별 OKS 시그마를 추가할 수 있습니다. OCR 체크포인트는 전체 CTC 문자 집합을 포함하여 파일이 자체적으로 완결되도록 합니다. 복원 체크포인트는 손상 유형 및 업스케일 비율을 기록할 수 있습니다. 트레이너 체크포인트는 `epoch`, 옵티마이저 상태 및 EMA 가중치와 같은 재개 상태를 추가하며; 공개된 추론 가중치는 이를 포함해서는 안 됩니다.

모든 아홉 개 키를 만족하는 파일은 메타데이터 경로를 통해 로드됩니다. 만족하지 않는 파일은 레이아웃을 인식하는 패밀리가 있으면 변환되거나, 아니면 누락된 항목을 이름으로 경고하며 호환성 경로를 통해 로드됩니다.

## 체크포인트 검사

<code-tabs name="inspect" />

`libreyolo metadata`은(는) 모델을 절대 생성하지 않으므로, 설치되지 않은 패밀리의 파일과 확실하지 않은 파일에서도 작동합니다.
