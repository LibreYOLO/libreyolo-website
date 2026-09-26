---
title: libreyolo doctor
seo_title: libreyolo doctor 명령 레퍼런스
description: >-
  학습 전에 탐지 데이터셋을 확인합니다: 기본값이 있는 인자, 건너뛰거나 골라서 실행할 수 있는 검사 계열, CI가 게이트로 삼을 수 있는
  종료 코드.
lead: >-
  탐지 데이터셋에 상태 검사를 모아 실행하고, 학습 실행에 지장을 줄 만한 항목을 보고합니다: 누락된 파일, 깨진 레이블, 손상된 이미지,
  분할(split) 간 데이터 누수, 클래스 불균형.
keywords:
  - libreyolo doctor cli
  - 데이터셋 상태 점검
  - yolo 데이터셋 검증
  - 데이터셋 누수 확인
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo doctor
    mono: true
  - label: 필수
    value: data
    mono: true
  - label: 출력
    value: stdout에 출력되는 발견 항목 보고서. 오류가 발견되면 종료 코드 1
snippets:
  examples:
    - label: 기본
      language: bash
      code: |
        # download=true를 주면 번들된 coco8.yaml이 이미지가 없을 때 직접 내려받습니다.
        libreyolo doctor coco8.yaml download=true
    - label: '빠른 검사, 이미지 디코딩 없음'
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: 선택한 검사에 CI 게이트 적용
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## 개요

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

데이터셋은 위치 인자이며, 대안으로 `data=<path>` 형태도 받습니다. 둘을 서로
다른 값으로 함께 주면 `config_conflict`로 종료합니다. 나머지는 모두
`key=value` 쌍이고 POSIX 형식도 동작하므로, `imgsz=1024`와 `--imgsz 1024`는
같은 인자입니다.

## 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `data` | | 위치 인자. YOLO 탐지 형식의 데이터셋 YAML, 예를 들어 `coco8.yaml`. 필수 |
| `imgsz` | `640` | 아주 작은 객체처럼 픽셀 기준 검사에 사용하는 학습 이미지 크기 |
| `fast` | `false` | 이미지 디코딩을 건너뛰며, 그러면 손상, 중복, 누수 검사가 빠집니다 |
| `skip` | | 건너뛸 검사 id 또는 계열을 쉼표로 구분해 지정합니다. 예: `images,labels.tiny_object` |
| `only` | | 이것만 실행할 검사 id 또는 계열을 쉼표로 구분해 지정합니다 |
| `strict` | `false` | CI 게이트를 위해 경고도 종료 코드를 실패로 만듭니다 |
| `download` | `false` | 데이터셋이 없을 때 URL 기반 다운로드를 허용합니다. 스크립트는 절대 실행하지 않습니다 |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 출력 억제 |
| `help_json` | `false` | 명령 스키마를 JSON으로 덤프하고 종료 |

### 검사 계열

`skip`과 `only`는 전체 검사 id 또는 계열 접두사를 받으므로, `images`는 모든
`images.*` 검사를 선택합니다.

| 계열 | 대상 |
|---|---|
| `config` | 데이터셋 YAML 자체: `names` 누락, `names`와 대조한 `nc`, 분할 누락, 해석할 수 없는 `path`, 중복된 클래스 이름 |
| `files` | 이미지와 레이블 짝 맞추기: 레이블 누락, 이미지 누락, 짝 없는 레이블, 지원하지 않는 확장자, 대소문자 충돌 |
| `labels` | 레이블 내용: 구문, 폴리곤 라인, 범위를 벗어난 클래스 id, 범위를 벗어난 좌표, 퇴화된 박스, 아주 작은 객체, 지나치게 큰 박스, 극단적인 종횡비, 중복 박스, 객체가 몰린 이미지, 동일한 파일 |
| `images` | 픽셀 데이터: 손상된 파일, EXIF 방향, 특이한 색상 모드, 아주 작거나 극단적인 크기, 단색 이미지, 정확 중복과 근사 중복 |
| `splits` | 분할 간 누수, 정확 일치와 근사 일치 |
| `balance` | 클래스 분포: 인스턴스가 없거나 적은 클래스, 불균형, 분할 커버리지, 배경 비율, 분할 편중 |

## 예제

<code-tabs name="examples" />

## 참고

### 종료 코드

오류가 발견되지 않으면 `0`, 발견 항목 중 하나라도 오류이면 `1`입니다.
`strict=true`이면 경고도 종료 코드를 `1`로 올리며, CI 게이트가 원하는 설정이
바로 이것입니다.

사용법 문제에는 별도의 코드가 있습니다: `skip`이나 `only`에 알 수 없는 검사
id나 계열이 있으면 `2`, 데이터셋을 찾을 수 없으면 `3`, 데이터셋이 탐지 형태가
아니면 `3`입니다.

### 스캔 전 선택 해석

`skip`과 `only`는 디스크에서 무언가를 읽기 전에 검사 레지스트리를 기준으로
해석되므로, 오타는 긴 이미지 검사가 끝난 뒤가 아니라 즉시 실패합니다. 아무것도
일치하지 않는 선택자는 오류이며, 메시지에 알려진 계열 목록이 나옵니다.

`skip`과 `only`, `fast`를 조합한 결과 실행할 검사가 하나도 남지 않으면, 조용히
통과하는 대신 이 또한 오류가 됩니다.

### 다운로드

`download=true`가 아니면 데이터셋을 가져오지 않으며, 수행하는 것은 URL
다운로드뿐입니다. 데이터셋 YAML에 포함된 Python 다운로드 스크립트는 플래그가
무엇이든 이 명령이 실행하지 않습니다.

### 범위

검사는 탐지 데이터셋을 위해 작성되었습니다. 레이블이 자세, 분할, 회전 바운딩
박스 형태인 데이터셋은 잘못된 규칙으로 채점되는 대신 식별되어 `data_invalid`로
거부됩니다.

### 출력

사람이 읽는 보고서는 stdout으로 나가고, `json=true`는 이를 요약 개수, 데이터셋
통계, 모든 발견 항목, 건너뛴 검사 목록을 담은 구조화된 객체로 대체합니다.

관련: 이 명령을 먼저 실행하도록 의도된 대상인
[`libreyolo train`](/docs/cli/train).
