---
title: 기존 가중치 가져오기
seo_title: LibreYOLO에서 업스트림 가중치 로드
description: >-
  Point LibreYOLO를 업스트림 프로젝트의 체크포인트로 지정합니다. 자동 변환은 로드 시점에 이를 다시 래핑하며, 클래스 수와 이름을
  유지합니다.
lead: >-
  LibreYOLO는 모델 계열을 업스트림 프로젝트에서 가져오기 때문에, 그들이 공개한 체크포인트는 거의 바로 로드할 수 있습니다. 그들이
  부족한 것은 메타데이터입니다. 자동 변환은 로드 시점에 이를 제공합니다.
keywords:
  - libreyolo 가중치 변환
  - 업스트림 체크포인트 로드
  - libreyolo 마이그레이션
  - pth를 libreyolo로 변환
  - 자동 변환
last_verified: 1.5.0
meta:
  - label: 진입점
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: 출처 옆에 쓰여 있는
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: 스크립트 변환기
    value: weights/ in the repository
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 이미 가지고 있는 체크포인트 경로를 대신 사용하십시오. 인식된
        # 업스트림 레이아웃은 즉석에서 변환되어 옆에 기록됩니다
        # 소스, 그리고 나서 로드됨.
        model = LibreYOLO("path/to/upstream-checkpoint.pth")

        # 클래스 수와 이름은 텐서와 파일 자체에서 나옵니다
        # 메타데이터, 그래서 파인튜닝은 COCO의 것이 아니라 자체 레이블 세트를 유지합니다.
        print(model.family, model.size, model.task, model.nb_classes)
        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 결과를 확인하십시오
      language: bash
      code: |
        # 변환된 파일은 게시된 파일과 동일한 스키마를 만족합니다.
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

이 페이지는 다른 프로젝트의 체크포인트에 관한 것입니다. 만약 이전 버전의 LibreYOLO에서 자신의 코드를 옮기고 있다면, [1.5.0으로 업그레이드](/docs/upgrade)를 참조하십시오.

## 외부 파일을 로드하면 무슨 일이 일어나는가

`LibreYOLO()`는 먼저 제한된, 가중치 전용 경로를 통해 모든 가중치 파일을 로드합니다. 결과에 전체 LibreYOLO 메타데이터가 포함되어 있으면 그대로 사용됩니다. 포함되어 있지 않으면, 파일은 다른 시도를 하기 전에 자동 변환기로 들어갑니다. 제한된 로드가 완전히 실패하면, 이는 체크포인트에 서드파티 객체가 피클된 경우 발생하며, 이때 자동 변환기는 해당 객체를 무효화하는 로더와 함께 시도됩니다.

자동 변환은 네 가지 작업을 수행합니다. 먼저 입력 프로젝트가 사용한 레이아웃에서 텐서 딕셔너리를 풀어냅니다. 그런 다음 모든 등록된 계열에게 결과 키를 인식하는지 묻고, 입력 프로젝트의 이름과 LibreYOLO 포트의 이름이 다를 경우 이름을 재매핑합니다. 그 다음에는 승자를 메타데이터 스키마 v1.0을 만족하는 체크포인트로 감싸고, 텐서 자체에서 크기, 작업, 클래스 수를 읽습니다. 마지막으로 결과를 원본 파일 옆에 저장하고 그 파일을 불러옵니다.

<code-tabs name="convert" />

변환은 조용하게 이루어지지 않습니다. 변환된 파일은 계열, 소스 이름, 출력 이름 및 결과 클래스 수와 함께 기록되므로 실행 로그에는 정확히 무엇이 로드되었는지가 기록됩니다.

## 그것이 풀어내는 레이아웃들

업스트림 체크포인트는 그들의 가중치를 몇 가지 일반적인 위치에 중첩시키며, 변환기는 텐서를 포함하고 있는 위치가 나올 때까지 순서대로 시도합니다: `ema.module` 아래의 EMA 블록 또는 평평한 `ema`, `module.` 접두어가 제거된 `ema_state_dict`, 그 다음 `params_ema`, `params`, `ema_net`, `net`, `model`, `state_dict`, 그리고 마지막으로 객체 자체. 여러 위치를 시도하는 것은, 단지 카운터만을 포함하는 `ema` 블록이 아래의 실제 가중치를 가리는 것을 방지합니다.

래퍼 접두사도 제거됩니다: 분산 학습에서는 `module.`, 컴파일된 모델에서는 `_orig_mod.`, 일부 재배포를 포함하는 `model.model.` 중첩 추가가 있습니다.

## 그것이 무엇을 읽고 어디에서 읽는지

크기, 작업 및 클래스 수는 파일 이름이 아니라 텐서에서 가져오므로, 세부 조정된 체크포인트는 아키텍처의 기본값 대신 자체 클래스 수로 변환됩니다. 클래스 이름은 존재할 경우 체크포인트 자체 메타데이터에서 가져오며, 이름이 `args` 또는 `hyper_parameters` 블록에 있으면 그 블록에서 가져오고, 탐지된 클래스 수에 맞게 잘라내어 기본 레이블 세트를 유지한 세부 조정에서도 더 이상 헤드가 갖고 있지 않은 인덱스를 포함하지 않도록 합니다.

밀집 작업은 만들어진 레이블을 부여하는 대신 명시적으로 처리됩니다. 깊이 체크포인트는 `depth`라는 하나의 클래스를 얻고, 복원 체크포인트는 `image`라는 하나의 클래스를 얻습니다. 포즈 체크포인트는 텐서나 계열 중 하나에서 키포인트 수를 제공해야 합니다. 둘 다 제공하지 않으면, 불완전한 파일을 작성하는 대신 변환이 거부됩니다.

RF-DETR은 자체 인식기를 갖습니다. 이는 크기 탐지가 전체 체크포인트를 필요로 하고, 그 헤드가 LibreYOLO가 80클래스 COCO 규칙을 사용하는 반면 91개의 출력을 가지기 때문입니다. 체크포인트는 정확히 80개의 이름을 포함하거나, 클래스 수를 80으로 선언하거나, COCO를 데이터셋으로 명시하거나, 클래스 또는 데이터셋 메타데이터가 전혀 없는 경우 80클래스로 정규화됩니다. 이름, 명시적인 80이 아닌 클래스 수 또는 COCO가 아닌 데이터셋 힌트로 식별되는 진정한 90클래스 모델은 그대로 보존됩니다.

## 변환된 파일이 저장되는 위치

출력은 소스 옆에 작성되며, 소스 이름을 따서 명명됩니다:

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

`upstream-checkpoint.pth`로 저장된 작은 YOLOv9 탐지기는 따라서 `upstream-checkpoint-LibreYOLO9t.pt`가 됩니다. 파일 이름을 계열명이 아닌 소스명을 따라 지정하면, 한 디렉토리 내에서 동일한 계열과 크기의 두 개의 세부 조정(fine-tune)이 서로 덮어쓰지 않으며, 공식 체크포인트와도 충돌하지 않습니다. 파일은 매번 로드될 때 다시 작성되므로 원본과 비교해 오래되지 않습니다. 디렉토리가 읽기 전용인 경우, 변환된 파일은 새 개인 임시 디렉토리로 이동되며, 로그에는 위치가 표시됩니다.

그때부터는 일반적인 LibreYOLO 체크포인트입니다: 메타데이터 경로를 통해 로드되며, `libreyolo metadata`가 이를 유효하다고 보고합니다.

## 도움이 필요한 경우

두 계열은 일반 인식기의 범위 밖에 있습니다. 시선(gaze) 계열은 완전히 제외됩니다: 추론 전용이며 공개 가중치에는 재배포 제한이 있습니다. RF-DETR은 위에서 설명한 전용 인식기가 처리하므로 제외됩니다.

원시 업스트림 PIDNet 체크포인트는 거부되며, `weights/convert_pidnet_weights.py`를 가리키는 오류가 발생합니다. 해당 스크립트는 체크포인트가 필요로 하는 Cityscapes 시맨틱 메타데이터를 작성합니다.

D-FINE과 DEIM은 같은 아키텍처 키를 공유하므로, 텐서만으로는 둘을 구분할 수 없습니다. 둘 다 파일을 주장하고 구별 표시가 있는 형제 계열가 없을 경우, 파일 이름이 결정합니다: `dfine_hgnetv2_n_coco.pth` 또는 `deim_hgnetv2_n_coco.pth` 형태의 이름이 이를 결정하며, 의미 없는 이름은 추측하지 않고 그 설명과 함께 거부됩니다. `LibreDFINE` 또는 `LibreDEIM`를 직접 인스턴스화하는 것도 이를 해결합니다.

여러 계열이 하나의 파일을 정당하게 주장할 때, 하위 클래스는 자신이 정제하는 기본 클래스를 이기고 나머지는 레지스트리 순서가 결정합니다. 그 순서는 각 계열의 검사가 얼마나 구체적인지를 기록하기 때문입니다. 파일 이름은 D-FINE과 DEIM이 동점일 때만 참고되므로, 파일 이름이 넓은 일치를 정밀한 일치보다 우선하게 할 수는 없습니다.

## 스크립트 변환기

이 저장소는 `weights/` 아래에 계열별 변환 스크립트를 포함하고 있으며, 반복되는 배관 작업을 위한 공유 헬퍼도 포함합니다. 이는 런타임 경로가 거부하는 파일에 대한 경로, 로드 시가 아니라 미리 체크포인트를 생성하기 위한 경로, 그리고 메타데이터가 텐서에서 추론되지 않고 제공되어야 하는 계열들을 위한 경로입니다.

그 스크립트들은 설치된 패키지가 아니라 저장소의 일부이므로, 하나를 사용하려면 클론해야 합니다:

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

모든 스크립트는 스키마 v1.0을 만족하는 체크포인트를 작성하며, 이는 자동 변환이 충족하는 기준과 발행된 가중치가 충족하는 기준과 동일합니다. 해당 스키마가 무엇을 포함하는지에 대해서는 [체크포인트 및 가중치](/docs/weights)를 참조하십시오.
