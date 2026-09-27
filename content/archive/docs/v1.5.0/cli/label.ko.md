---
title: libreyolo label
seo_title: libreyolo label 명령 레퍼런스
description: '로컬 바운딩 박스 어노테이션 도구 실행: 기본값이 있는 인자, AI 어시스트 스위치, 그리고 네트워크 인터페이스에 바인딩할 때 노출되는 것.'
lead: >-
  바운딩 박스를 그리고 편집하는 로컬 웹 도구를 실행합니다. LibreYOLO 네이티브 레이블 파일을 기록하므로, 여기서 어노테이션한
  데이터셋은 변환 단계 없이 그대로 학습됩니다.
keywords:
  - libreyolo label cli
  - 바운딩 박스 어노테이션 툴
  - yolo 레이블링 툴
  - 자동 레이블링 cli
  - libreyolo label 공유
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo label
    mono: true
  - label: 출력
    value: 'stdout에 서버 URL이 출력되고, 레이블은 이미지 옆에 labels/*.txt로 기록됩니다'
snippets:
  examples:
    - label: 기본
      language: bash
      code: |
        # 프로젝트 홈을 열고, 브라우저에서 데이터셋을 선택하거나 새로 만듭니다.
        libreyolo label
    - label: '수동 전용, 고정 포트'
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: 팀원 참여 허용
      language: bash
      code: |
        libreyolo label share=true
source_hash: bddad245877793b1
---

## 개요

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

인자는 `key=value` 쌍이며, POSIX 형식도 동작하므로 `port=9200`과
`--port 9200`은 같은 인자입니다.

## 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `data` | | 바로 열 데이터셋 YAML 또는 폴더. 설정하지 않으면 프로젝트 홈에서 시작합니다 |
| `host` | `127.0.0.1` | 바인딩할 호스트 또는 인터페이스 |
| `port` | `8000` | 바인딩할 포트. 사용 중이면 다음 빈 포트로 넘어갑니다 |
| `device` | `auto` | AI 자동 레이블링에 사용할 장치: `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | AI 자동 레이블링을 끄고 수동 레이블링 도구만 남깁니다 |
| `no_browser` | `false` | 브라우저를 자동으로 열지 않습니다 |
| `share` | `false` | `0.0.0.0`에 바인딩해 같은 네트워크의 팀원이 접속할 수 있게 합니다 |
| `json` | `false` | stdout에 JSON 출력 |
| `quiet` | `false` | stderr 억제 |
| `verbose` | `false` | stderr 상세 출력 |

## 예제

<code-tabs name="examples" />

## 참고 사항

### 기록하는 내용

박스는 LibreYOLO 네이티브 `labels/*.txt` 파일로 저장되며, 이 형식은
`libreyolo train`이 읽는 형식이라 이후에 아무것도 변환할 필요가 없습니다. 이
버전은 바운딩 박스만 다룹니다. 이미지 사이를 이동할 때마다 편집 내용이 저장됩니다.

### 데이터셋 열기

`data`를 지정하지 않으면 도구는 프로젝트 홈에서 시작하고, 데이터셋은 브라우저에서
선택하거나 새로 만듭니다. `data=path/to/data.yaml`을 넘기면 해당 데이터셋이 바로
열리고, 시작 줄에 이미지 수, 클래스 수, 그리고 데이터셋에 쓸 수 있는지가
표시됩니다. 읽기 전용 데이터셋도 열리며, 왜 쓸 수 없는지 알려줍니다.

### 공유, 그리고 `host`의 역할

`share=true`는 와일드카드 주소에 바인딩해 같은 네트워크의 다른 컴퓨터에서도 도구에
접근할 수 있게 하고, 프로젝트를 전환하거나 삭제하고 연산을 시작하는 관리 작업은 이
컴퓨터에 남습니다.

`host`를 특정 인터페이스로 설정하면 이야기가 달라지고 더 위험해집니다: 호스트가
네트워크 클라이언트와 구분되지 않게 되어 모든 클라이언트가 관리 권한을 갖습니다.
이렇게 하면 명령이 stderr에 경고를 출력합니다. `share=true`를 쓰는 편이 낫습니다.

### 포트와 종료

포트가 사용 중이면 다음 포트로 넘어가며, 요청한 포트에서 최대 스무 개까지
시도합니다. 스무 개가 모두 실패하면 `io_error`로 종료합니다. stdout에 출력되는
URL은 실제로 바인딩된 포트를 사용합니다. `share=true`를 지정하면 결과에 팀원이
열어야 할 주소인 `lan_url`도 함께 담깁니다.

명령은 Ctrl+C를 누를 때까지 포그라운드에서 서비스를 제공합니다.

관련 항목: 학습 전에 레이블링한 데이터셋을 확인하려면
[`libreyolo doctor`](/docs/cli/doctor), 그 데이터셋으로 학습하려면
[`libreyolo train`](/docs/cli/train)을 사용합니다.
