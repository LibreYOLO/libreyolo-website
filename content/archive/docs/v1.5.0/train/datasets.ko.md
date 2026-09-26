---
title: 데이터셋
seo_title: LibreYOLO의 학습 데이터셋
description: >-
  YAML LibreYOLO가 읽는 데이터셋, 기대하는 폴더 구성, 자동 다운로드 작동 방식, 그리고 학습 전에 데이터셋을 확인하는
  doctor 명령어.
lead: >-
  LibreYOLO 데이터셋은 루트, 그 분할 및 클래스 이름을 명명하는 YAML 파일입니다. 레이블 파일이 어디에 있는지를 포함한 나머지
  모든 것은 관례에 따라 해당 파일에서 파생됩니다.
keywords:
  - yolo 데이터셋 형식
  - data.yaml
  - 맞춤형 데이터셋 학습
  - yolo 레이블 형식
  - 코코 JSON 데이터셋
  - 데이터셋 자동 다운로드
  - 리브리욜로 의사
  - 클래스 불균형 확인
  - 학습 검증 분할 누수
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 번들된 이름, 상대 경로 또는 절대 경로 모두 작동합니다.
        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: 데이터셋 확인
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: 경고에도 CI 작업 실패
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: 이미지 디코드 단계를 건너뛰기
      language: bash
      code: |
        # 레이블과 YAML만 읽습니다. 손상, 중복 및 분할 누수
        # 모든 검사는 픽셀이 필요하므로 무시됩니다.
        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## 데이터셋에서 포인트를 학습시키다

`data=`는 YAML 경로 또는 패키지와 함께 제공되는 구성 이름을 사용합니다.

<code-tabs name="train" />

이름은 고정된 순서로 확인됩니다: 존재하는 절대 경로, 그 다음 작업 디렉터리를 기준으로 주어진 이름, 그 다음 `.yaml`가 추가된 동일한 이름, 그 다음 번들된 설정 디렉터리. 아무것도 일치하지 않으면 오류는 검색된 모든 디렉터리를 나열하고 번들된 설정을 표시합니다.

## 번들 구성

열세 개의 데이터셋 구성은 패키지 내 `libreyolo/config/datasets/` 아래에 있습니다.

| 설정 | 작업 | 노트 |
|---|---|---|
| `coco8.yaml` | 탐지하다 | 8개의 이미지, 일반 URL에서 다운로드 |
| `coco128.yaml` | 탐지하다 | 128개의 이미지 |
| `coco1000.yaml` | 탐지하다 | 800 학습, 200 검증 |
| `coco5000.yaml` | 탐지하다 | 4000 학습, 1000 검증 |
| `coco.yaml` | 탐지하다 | 전체 COCO 2017 |
| `coco-val-only.yaml` | 탐지하다 | val2017만 |
| `coco8-pose.yaml` | 자세 | 8개의 이미지, COCO-17 키포인트 |
| `coco-pose.yaml` | 자세 | COCO 2017 키포인트 |
| `ade20k.yaml` | 의미론의 | 150수업 |
| `cityscapes.yaml` | 의미론의 | 수업 19개, 수동으로 다운로드 |
| `cocostuff.yaml` | 의미론의 | 182개 수업, 수동으로 다운로드 |
| `gopro.yaml` | 복원하다 | 디블러링 쌍 |
| `sr8.yaml` | 복원하다 | 초해상도 쌍 |

`coco8.yaml`와 `coco128.yaml`만 일반 다운로드 URL을 제공합니다. 나머지는 아래에 설명된 옵트인 과정이 필요한 Python 다운로드 블록을 제공하거나, 데이터가 이미 디스크에 있어야 한다고 예상합니다.

## 데이터셋이 디스크에 위치한 곳

YAML `path` 키는 데이터셋 루트를 지정합니다. 절대 `path`는 작성된 대로 사용됩니다. 상대 경로는 먼저 datasets 디렉토리 아래에서 찾고, 그 다음 YAML 파일 자체 옆에서 찾으며, 다운로드하려는 데이터셋은 datasets 디렉토리 아래에 저장됩니다.

그 디렉터리는 `~/datasets`이며, `LIBREYOLO_DATASETS_DIR` 환경 변수에 의해 재정의됩니다. 이에 대한 설정 파일은 없습니다.

## YAML 키

```yaml
path: my-dataset        # 데이터셋 루트
train: images/train     # 학습하는 데 필요한
val: images/val         # 검증이 필요함
test: images/test       # 선택 사항
nc: 3                   # 선택 사항; 이름과 일치해야 함
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # 선택 사항
```

`train`, `val` 및 `test`는 각각 이미지 디렉토리, 한 줄에 하나의 이미지 경로를 나열한 `.txt` 파일, 또는 둘을 혼합한 목록을 받습니다. `.txt` 목록의 줄은 상대 경로일 수 있으며, 이 경우 목록 파일 자체 디렉토리를 기준으로 해석됩니다. 또한 `#`로 시작하는 줄은 건너뜁니다.

`names`는 목록이거나 정수 키 맵일 수 있습니다. `nc`는 선택 사항입니다. 두 개가 모두 존재하고 불일치할 경우 의사는 이를 오류로 보고합니다.

## 디렉토리 구조 및 레이블 파일

탐지, 분할, 포즈 및 방향이 지정된 상자는 모두 하나의 레이아웃을 공유합니다. 레이블 경로는 이미지 경로에서 `images` 디렉토리 구성 요소를 `labels`로 바꾸고 확장자를 `.txt`로 변경하여 파생됩니다:

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

`images` 경로 구성 요소 전체만 재작성되므로, `images_old`라는 디렉터리는 그대로 남습니다.

탐지 행은 원본 이미지의 가로 및 세로에 대해 모두 `[0, 1]`로 정규화된 다섯 개의 필드입니다:

```text
<class_id> <cx> <cy> <w> <h>
```

레이블 파일이 없거나 비어 있으면 이미지에 객체가 없다는 의미이며, 이는 객체를 올리는 대신 배경으로 학습됩니다. 필드가 다섯 개 이상인 행은 폴리곤으로 읽히며, 그 박스는 폴리곤의 범위가 됩니다. 따라서 탐지 학습을 위한 세분화 내보내기가 문제 없이 로드됩니다. 의사는 몇 개의 행이 해당 경로를 거쳤는지 보고합니다.

## 기타 작업

세분화는 적어도 세 개의 점을 가진 폴리곤 행 `<class_id> <x1> <y1> ... <xN> <yN>`와 동일한 레이아웃을 유지합니다. 다섯 필드 탐지 행은 허용되며 직사각형 인스턴스를 의미합니다.

Pose는 YAML에 `kpt_shape: [K, D]`와 선택적인 `flip_idx` 순열을 추가합니다. 각 행에는 정확히 `5 + K * D` 필드가 있습니다: 상자, 그리고 `K` 키포인트의 `x y` 또는 `x y v`, 가시성 `0`, `1` 또는 `2`.

오리엔티드 박스는 정확히 아홉 개의 필드를 사용하며, 클래스와 이어서 정규화된 좌표의 네 개 코너 점이 표시됩니다. 파일에는 각도가 저장되지 않습니다.

시멘틱 분할은 각 이미지를 동일한 해상도의 단일 채널 마스크와 연결하며, `masks_dir`(기본 `masks`)를 `images`로 대체하여 해결합니다. 픽셀 값 `255`는 무시를 의미합니다. `label_mapping`는 로드 시 소스 ID를 학습 ID로 다시 매핑합니다.

분류는 레이블 파일 대신 ImageFolder 트리를 사용하며, `train/`와 `val/` 각각에는 클래스별로 하나의 디렉토리가 포함되어 있습니다. 클래스-인덱스 매핑은 정렬된 폴더 이름 순서입니다.

복원은 변질된 입력을 `input_dir`와 `target_dir`를 통해 동일한 해상도의 깨끗한 목표와 짝지어 줍니다. 깊이, 표면 법선 및 엣지는 각각 자체 디렉토리 키를 통해 이미지와 밀집 맵을 짝지어 줍니다.

깊이 척도 규약과 판옵틱 세그먼트 ID PNG 인코딩을 포함한 전체 작업별 계약은 라이브러리 저장소에서 `docs/dataset_schema.md`입니다.

## 원본 COCO JSON

COCO JSON 주석 파일은 직접 사용할 수 있습니다. `annotations` 매핑을 추가하면 분할 경로가 이미지 루트가 됩니다:

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

`names`가 존재할 때, JSON 카테고리 이름은 그것과 일치해야 하며, `names`는 모델이 예측하는 레이블 ID를 정의합니다. `names`가 없으면 COCO 카테고리 ID가 정렬되어 `0..N-1`에 밀집하게 매핑됩니다.

이 경로는 분할마다 하나의 이미지 디렉토리를 예상합니다. 경로 목록이나 `.txt` 이미지 목록은 다른 세트를 조용히 로드하는 대신 오류를 발생시킵니다.

## 자동 다운로드

데이터셋은 해당 `train` 또는 `val` 경로가 비어 있지 않은 디렉토리나 존재하는 파일로 해석될 때 존재하는 것으로 간주됩니다. 그렇지 않고 YAML에 `download` 키가 있는 경우, 그 값이 다음에 일어날 일을 결정합니다.

`http` 또는 `https` URL이 가져와지고, 그것이 zip이면 데이터셋 루트에 압축이 풀립니다. 다른 모든 것은 내장 Python 스크립트로 처리되며 `allow_download_scripts=True`일 때만 실행됩니다. 그것이 없으면 스크립트는 경고와 함께 건너뛰어지고 디스크에 있는 내용으로 계속 학습이 진행됩니다.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

플래그는 네트워크 게이트가 아니라 코드 실행 게이트입니다. URL 다운로드는 어쨌든 발생하며, `download: |` 블록이 그것을 필요로 합니다. 플래그가 켜져 있을 때 CLI는 경고를 출력하며, 의사는 절대로 그것을 활성화하지 않습니다.

## 학습하기 전에 데이터셋을 확인

`libreyolo doctor`는 탐지 데이터셋을 읽고 GPU가 사용되기 전에 무엇이 잘못될 수 있는지 보고합니다. 오류를 찾으면 1로 종료하므로 CI 게이트로 작동합니다.

<code-tabs name="doctor" />

검사는 여섯 가지 범주로 나뉩니다:

| 계열 | 찾는다 |
|---|---|
| `config` | `names`와 일치하지 않는 `names`, `nc` 누락, 누락되었거나 비어 있는 분할, 중복된 클래스 이름 |
| `files` | 레이블 파일이 없는 이미지, 이미지가 없는 레이블, 분할에 나열된 누락된 이미지, 스템 충돌 |
| `labels` | 잘못된 행, `[0, nc)` 밖의 클래스 ID, `[0, 1]` 밖의 좌표, 면적이 0인 박스, 너무 작거나 큰 박스, 중복 박스, 바이트가 동일한 레이블 파일 |
| `balance` | 샘플이 거의 없거나 없는 클래스, 클래스 불균형 비율, 한 분할에만 존재하는 클래스, 배경 이미지 비율 |
| `images` | 디코딩할 수 없는 파일, EXIF 회전, 이상한 채널 배치, 균일한 이미지, 정확한 중복 및 거의 중복 |
| `splits` | 두 분할에서 동일한 이미지가 정확히 또는 거의 동일하게 나타나는 것 |

`--only`와 `--skip`는 체크 ID 또는 계열 접두사를 사용하므로 `skip=images,labels.tiny_object`는 유효합니다. `--fast`는 픽셀을 디코딩해야 하는 모든 체크를 제외하는데, 이는 `images` 및 `splits` 계열입니다.

알아둘 만한 두 가지 동작이 있습니다. `--strict`는 오류뿐만 아니라 경고도 종료 코드를 실패하게 만듭니다. 그리고 닥터는 탐지 데이터셋만 다루며, 포즈, 세그먼트 또는 오리엔티드 박스 데이터셋은 잘못된 계약과 비교되지 않고, 탐지한 내용을 이름으로 표시하는 메시지와 함께 거부됩니다.

## 관련된

- 데이터가 준비되면 `train()`가 취하는 인수에 대한 [하이퍼파라미터](/docs/train/hyperparameters).
- [검증 및 지표](/docs/train/validation) - `val` 또는 `test` 분할에서 평가하기 위함.
