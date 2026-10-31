---
title: libreyolo monitor
seo_title: libreyolo monitor 명령 레퍼런스
description: >-
  학습 실행을 위한 실시간 대시보드를 띄웁니다: 기본값이 딸린 인자, 서버가 디스크에서 읽는 파일, 그리고 서버 하나로 여러 실행을 다루는
  방법.
lead: >-
  학습 실행이 디스크에 기록한 산출물을 읽어 웹 대시보드를 제공합니다. 학습 프로세스에 붙지 않으므로 진행 중인 실행, 끝난 실행, 죽은
  실행이 모두 표시됩니다.
keywords:
  - libreyolo monitor cli
  - yolo 학습 대시보드
  - 학습 진행 상황 실시간 확인
  - libreyolo monitor 포트
  - 학습 지표 시각화
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo monitor
    mono: true
  - label: 출력
    value: stdout에 서버 URL을 출력한 뒤 프로세스가 포그라운드에 남습니다
snippets:
  examples:
    - label: 기본
      language: bash
      code: |
        # runs/를 감시하고 그 아래의 모든 실행을 나열합니다.
        libreyolo monitor
    - label: 다른 runs 루트
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: '단일 실행, 고정 포트, 브라우저 없음'
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
source_hash: 4aa178141d451728
---

## 개요

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

디렉터리는 위치 인자입니다. 나머지는 모두 `key=value` 쌍이며 POSIX 형식도
동작하므로 `port=9100`과 `--port 9100`은 같은 인자입니다.

## 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `run_dir` | `runs` | 위치 인자. 감시할 runs 루트, 또는 바로 열 단일 실행 디렉터리. 어느 쪽이든 루트 아래의 모든 실행이 나열됩니다 |
| `host` | `127.0.0.1` | 바인딩할 호스트 또는 인터페이스 |
| `port` | `8420` | 바인딩할 포트. 이미 사용 중이면 다음 빈 포트로 넘어갑니다 |
| `no_browser` | `false` | 브라우저를 자동으로 열지 않습니다 |
| `json` | `false` | stdout으로 JSON 출력 |
| `quiet` | `false` | stderr 억제 |
| `verbose` | `false` | stderr에 상세 출력 |

## 예시

<code-tabs name="examples" />

## 참고

### 서버 하나, 여러 실행

서버는 단일 실행이 아니라 runs 루트를 감시하고 각 실행을 URL로 구분하므로, 한
컴퓨터의 여러 실행이 포트 하나를 공유합니다. 인덱스를 보려면 루트 URL을, 실행별로
보려면 실행마다 탭 하나씩을 열면 됩니다; 각 URL의 `?run=` 파라미터가 어느 실행인지
알려 줍니다.

명령을 단일 실행 디렉터리로 지정하면 서버는 그 디렉터리의 부모를 루트로 삼으므로,
형제 실행도 인덱스에 그대로 나타나고, 이름을 지정한 실행으로는 바로 딥링크됩니다.

### 읽는 대상

대시보드는 `libreyolo train`이 기록하는 `status.json`, `metrics.jsonl`,
`train.log` 파일과 해당 실행의 이미지로 구성됩니다. 학습 프로세스 자체에서는
아무것도 읽지 않으므로, 끝났거나 죽은 실행도 진행 중인 실행과 똑같이 표시됩니다.

### 사전 조건과 포트

실행이 최소 하나는 이미 있어야 합니다. 인자가 없고 `runs/` 디렉터리도 없으면
명령은 `source_not_found`로 종료합니다; 지정한 디렉터리에 실행이 하나도 없을 때도
마찬가지입니다.

포트가 이미 사용 중이면 다음 포트로 넘어가며, 요청한 포트에서 최대 스무 개까지
시도합니다. 스무 개가 모두 실패하면 `io_error`로 종료합니다. stdout에 출력되는
URL에는 실제로 바인딩된 포트가 들어갑니다.

명령은 Ctrl+C를 누를 때까지 포그라운드에서 서비스합니다. `json=true`는 URL, 감시
중인 루트, 찾은 실행 개수를 `schema_version`이 포함된 객체 하나로 출력합니다.

관련: [`libreyolo train`](/docs/cli/train), 이 명령의 `project`와 `name` 인자가 이
실행 디렉터리들이 어디에 생기는지 결정합니다.
