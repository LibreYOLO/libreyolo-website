---
title: 체크포인트와 가중치
seo_title: LibreYOLO 체크포인트 및 가중치
description: >-
  LibreYOLO가 모델 가중치를 찾고 다운로드하며 검증하는 방법, 모델이 호스팅되는 위치, 네트워크 없이 실행하는 방법, 체크포인트가
  안전하게 로드되는 이유.
lead: >-
  LibreYOLO 체크포인트는 상태 사전과 그것을 식별하는 데 필요한 메타데이터를 담고 있는 torch.save 사전입니다. 이 페이지에서는
  이러한 파일이 어디에서 오는지, 어디에 저장되는지, 그리고 어떻게 로드되는지 다룹니다.
keywords:
  - libreyolo 가중치
  - libreyolo 체크포인트
  - libreyolo 가중치 다운로드
  - libreyolo 오프라인
  - 리브레욜로 허깅 페이스
  - 체크포인트 메타데이터
last_verified: 1.5.0
meta:
  - label: 호스팅됨
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

        # 빈 파일 이름은 weights/LibreYOLO9t.pt로 확인되며
        # 이미 존재하지 않는 경우 거기에 다운로드됩니다.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: 명시적 경로
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 디렉터리 구성 요소가 있는 경로는 작성된 대로 정확히 사용됩니다 그리고
        # 네트워크에서 절대 가져오지 않습니다.
        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")
        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # 모델을 구성하지 않고 메타데이터를 읽고 보고합니다
        # 그것이 스키마를 만족하는지 여부.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: 제이슨
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


        # 문제 목록을 반환합니다. 비어 있으면 파일이 v1.0을 만족한다는 의미입니다.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## 체크포인트를 찾는 곳

디렉터리 구성 요소가 없는 모델 참조, 예를 들어 `LibreYOLO9t.pt`는 현재 작업 디렉터리를 기준으로 `weights/`에 대해 해결됩니다. `weights/LibreYOLO9t.pt`가 존재하면 그것이 사용되며; 작업 디렉터리 자체에 해당 이름의 파일이 존재하면 대신 그것이 사용됩니다; 그렇지 않으면 `weights/LibreYOLO9t.pt`가 다운로드 대상이 됩니다.

디렉터리, 절대 경로 또는 상대 경로를 포함하는 참조는 문자 그대로 읽습니다. 이는 가중치가 중앙에 위치하고 아무것도 가져오면 안 될 때 사용하는 형식입니다.

<code-tabs name="load" />

## 자동 다운로드

해결된 경로가 존재하지 않을 때, LibreYOLO는 파일 이름을 분석하여 계열, 크기 및 작업을 복구하고, 해당 계열에게 다운로드 URL을 요청합니다. 대부분의 계열는 Hugging Face의 LibreYOLO 조직에서 이를 구성하며, 각 체크포인트는 파일 이름을 따서 명명된 고유한 저장소를 가지고 있습니다:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

데이터셋 변형 접미사는 저장소 이름의 일부로 남아 있으므로, 기본 계열가 아닌 다른 것으로 학습된 체크포인트는 기본 체크포인트를 덮어쓰지 않고 자체 저장소로 해석됩니다.

전송 자체는 방어적입니다. 잘린(weight) 파일은 나중에 도움이 되지 않는 오류와 함께 실패하기 때문입니다. 다운로드는 `.part` 파일로 스트리밍되며 완료될 때만 원자적으로 제자리에 이동되므로 중단된 프로세스가 최종 경로에 절반만 작성된 체크포인트를 남길 수 없습니다. 중단된 전송은 HTTP 검증기를 사용하여 바이트 오프셋에서 다시 시작되며, 서버가 객체가 변경되었음을 나타내면 0에서 다시 시작됩니다. 실패는 지수 백오프로 세 번 재시도됩니다. 동일한 경로를 대상으로 하는 동시 프로세스는 잠금 파일을 획득하므로 두 학습 실행이 동시에 시작되더라도 한 번만 다운로드됩니다. 계열이 LibreYOLO 조직이 아닌 제3자 호스트에서 파일을 가져오는 경우, 체크섬을 고정하고 불일치 시 파일을 거부할 수 있습니다.

`HF_TOKEN`가 설정되어 있거나 `~/.cache/huggingface/token`에 토큰이 캐시되어 있는 경우, 그것은 베어러 토큰으로 첨부됩니다. 이것은 `huggingface.co` URL에만 첨부되므로, 다른 호스트에서 다운로드하는 계열는 결코 그것을 받지 않습니다.

모든 계열가 자동으로 다운로드되는 것은 아닙니다. 일부는 배포된 가중치를 재배포할 수 없기 때문에 일부러 URL을 제공하지 않으며, 오류가 발생하면 대신 무엇을 제공해야 하는지 설명합니다. 다른 일부는 전송이 시작되기 전에 라이선스 공지를 출력합니다. 이 공지는 체크포인트의 조건이 코드의 조건보다 더 좁다는 런타임 신호이며, 스크롤하여 지나가지 말고 읽을 가치가 있습니다.

## 허깅페이스 조직

게시된 가중치는 [huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO)에서 확인할 수 있으며, 체크포인트마다 하나의 저장소가 있습니다. 각 저장소에는 라이선스가 있으며, 동일한 계열일지라도 라이선스가 일률적이지 않습니다: 코드가 MIT인 계열이라도 일부 가중치는 MIT가 아닐 수 있습니다. 저장소가 권위 있는 출처입니다. 모든 모델 페이지에는 해당 계열의 게시된 체크포인트와 그 라이선스가 체크포인트 및 라이선스 섹션에 나와 있습니다.

## 오프라인 작업 중

파일이 로컬에 있으면 라이브러리와 관련하여 네트워크 접근이 필요하지 않습니다. 두 가지 방법이 있습니다:

작업이 실행되는 위치 옆에 `weights/` 디렉토리를 미리 채워 넣으십시오. 연결된 컴퓨터에서 체크포인트를 한 번 가져오고 나서 디렉토리를 복사하는 것만으로 충분합니다. 위의 확인 단계가 이를 찾아내며 네트워크에는 도달하지 않습니다.

또는 공유 위치에 대한 절대 경로를 전달할 수 있습니다. 디렉토리 구성 요소가 있는 참조는 주어진 대로 사용되므로, 선별된 가중치의 읽기 전용 마운트는 유효한 설정입니다. 프로세스가 변환해야 하는 체크포인트 옆에 쓸 수 없는 경우, 변환은 실패하는 대신 개인 임시 디렉토리로 대체됩니다.

데이터셋은 별도의 규칙을 따릅니다: `~/datasets` 아래에서 해결되거나, 해당 변수가 설정된 경우 `LIBREYOLO_DATASETS_DIR`라는 디렉터리 아래에서 해결됩니다.

## 적재 안전

체크포인트는 피클이며, 피클은 열릴 때 임의의 코드를 실행할 수 있습니다. LibreYOLO는 모든 가중치 파일을 신뢰할 수 없는 것으로 취급하며 PyTorch의 `weights_only=True` 경로로 로드하는데, 이는 언피클러가 텐서와 소수의 안전한 타입들만 허용하도록 제한합니다. 이는 LibreYOLO가 다운로드한 파일뿐만 아니라 사용자가 전달하는 파일에도 적용됩니다. 해당 인자를 지원하지 않는 오래된 PyTorch 빌드에서는 로드가 안전하지 않게 실행되지 않고 거부됩니다.

일부 업스트림 학습 체크포인트는 제한된 언픽클러가 거부하는 객체를 포함하고 있습니다. 예를 들어, 학습된 프레임워크에서 가져온 구성 객체가 있습니다. 이러한 객체는 LibreYOLO에 필요하지 않은 메타데이터이므로, 변환 과정에서 각 차단된 클래스는 아무 것도 실행하지 않고 언픽클러를 만족시키는 비활성 대체물로 대체되며, 변환된 파일에는 텐서만 살아남습니다. 민감한 모듈 이름은 스텁 처리되지 않고 즉시 거부되며, 재시도 루프는 제한되어 있어, 끝없이 차단된 클래스 시리즈를 도입하도록 설계된 파일은 안전하게 실패합니다. 나머지 경로에 대해서는 [기존 가중치 가져오기](/docs/migrate)를 참조하십시오.

## 체크포인트 메타데이터

LibreYOLO 체크포인트는 `model` 키가 PyTorch 상태 딕셔너리를 갖고 있는 딕셔너리입니다. 스키마 v1.0에서는 아홉 개의 키가 필요하며, 이 키들을 통해 팩토리는 파일 이름을 분석하거나 텐서 모양으로 추측하지 않고도 파일을 식별할 수 있습니다.

| 열쇠 | 의미 |
|---|---|
| `model` | 파이토치 상태 사전 |
| `schema_version` | 메타데이터 계약 버전. v1.0은 문자열 `1.0`를 사용합니다 |
| `libreyolo_version` | 파일을 생성한 LibreYOLO 버전 |
| `model_family` | `yolo9`와 같은 등록된 계열 식별자 |
| `size` | 그 계열 내의 변형, 예를 들어 `t` 또는 `r18` |
| `task` | 하나의 정본 작업 이름 |
| `nc` | 양성 클래스 수 |
| `names` | 클래스 인덱스를 레이블로 매핑한 것, `0`부터 `nc - 1`까지 포함 |
| `imgsz` | 양의 입력 해상도 |

추가 구조가 있는 작업은 해당 키와 함께 이를 기록합니다. Pose 체크포인트는 `num_keypoints` 및 `keypoint_dim`를 추가하며, 키포인트별 OKS 시그마를 추가할 수 있습니다. OCR 체크포인트는 전체 CTC 문자 집합을 포함하여 파일이 자체적으로 완전하도록 합니다. 복원 체크포인트는 열화 유형과 업스케일 계수를 기록할 수 있습니다. 트레이너 체크포인트는 `epoch`, 옵티마이저 상태, EMA 가중치 등 복원 상태를 추가하며, 공개된 추론 가중치는 이를 포함해서는 안 됩니다.

모든 아홉 개 키를 만족하는 파일은 메타데이터 경로를 통해 로드됩니다. 만족하지 않는 파일은 레이아웃을 인식하는 계열가 있으면 변환되거나, 그렇지 않으면 누락된 항목을 경고하는 이름과 함께 호환성 경로를 통해 로드됩니다.

## 검문소를 점검하기

<code-tabs name="inspect" />

`libreyolo metadata`는 모델을 절대 생성하지 않으므로, 설치되지 않은 계열의 파일이나 확실하지 않은 파일에서도 작동합니다.
