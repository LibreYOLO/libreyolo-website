---
title: libreyolo ui
seo_title: libreyolo ui 명령 레퍼런스
description: '로컬 추론 웹 UI 실행: 바인드 주소, 포트 동작, 장치 선택, 명령이 종료되는 방식.'
lead: '드롭하거나 붙여넣은 이미지를 받아 선택한 모델로 추론하고, 그 결과를 브라우저에 보여주는 로컬 웹 서버를 시작합니다.'
keywords:
  - libreyolo ui 명령어
  - libreyolo 웹 ui
  - 로컬 추론 웹 ui
  - 드래그 앤 드롭 추론
  - libreyolo ui 포트
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo ui
    mono: true
  - label: 출력
    value: '표준 출력에 서버 URL이 출력되고, 이후 프로세스는 포그라운드에 남습니다'
snippets:
  examples:
    - label: 기본
      language: bash
      code: |
        libreyolo ui
    - label: '포트 고정, 브라우저 열지 않기'
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: 'CPU에서 실행, 기계 판독용'
      language: bash
      code: |
        libreyolo ui device=cpu json=true
source_hash: b0eebd33fd0f463b
---

## 형식

```bash
libreyolo ui [key=value ...]
```

인자는 `key=value` 쌍이며 POSIX 형식도 동작하므로, `port=9000`과 `--port 9000`은
같은 인자입니다.

## 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `host` | `127.0.0.1` | 바인드할 호스트 또는 인터페이스 |
| `port` | `8000` | 바인드할 포트. 사용 중이면 다음 빈 포트로 올라감 |
| `device` | `auto` | 장치: `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | 브라우저를 자동으로 열지 않음 |
| `json` | `false` | 표준 출력으로 JSON 출력 |
| `quiet` | `false` | 표준 에러 출력 억제 |
| `verbose` | `false` | 자세한 표준 에러 출력 |

## 예제

<code-tabs name="examples" />

## 참고

기본 바인드 주소는 루프백이므로, UI는 이 컴퓨터에서만 접근할 수 있습니다.

요청한 포트가 사용 중이면 명령은 다음 포트를 시도하고, 요청한 값에서 20 포트
뒤까지 계속 올라갑니다. 20개 모두 실패하면 `io_error`로 종료하면서 다른 포트를
지정하라고 안내합니다. 표준 출력에 표시되는 URL은 실제로 바인드된 포트이므로,
요청한 포트를 가정하지 말고 그 URL을 읽으십시오.

`no_browser=true`가 아닌 한, 바인드 직후 그 URL로 브라우저 탭이 열립니다.

그다음 명령은 Ctrl+C를 누를 때까지 포그라운드에서 서비스하며, Ctrl+C를 누르면
서버가 정상적으로 종료됩니다. 분리 모드(detached mode)는 없으므로, 터미널을
되찾으려면 셸의 기능으로 백그라운드에 보내십시오.

`json=true`를 지정하면 서버가 시작되기 전에 URL과 장치가 `schema_version`이 포함된
하나의 객체로 출력되며, 스크립트는 이렇게 바인드된 포트를 가져옵니다.

관련 항목: 박스를 그리고 레이블을 저장하려면 [`libreyolo label`](/docs/cli/label),
학습 실행을 관찰하려면 [`libreyolo monitor`](/docs/cli/monitor). 둘 다 포트와
브라우저 동작이 같은 로컬 웹 서버입니다.
