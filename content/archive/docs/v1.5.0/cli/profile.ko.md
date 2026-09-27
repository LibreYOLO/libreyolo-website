---
title: libreyolo profile
seo_title: libreyolo profile 명령 레퍼런스
description: '학습과 추론 속도를 측정하고 그 결과를 읽는 법: profile의 모든 하위 명령과 인자, 기본값, 그리고 각 관점이 보고하는 내용.'
lead: >-
  학습 스텝이나 추론 호출에서 시간이 어디로 가는지 측정하고, 자체 완결형 프로파일을 기록하며, 그 프로파일을 여러 관점으로 다시 읽어 주는
  명령 그룹입니다.
keywords:
  - libreyolo profile cli
  - yolo 학습 성능 프로파일링
  - 추론 지연 시간 측정
  - gpu 커널 프로파일링
  - libreyolo profile compare
last_verified: 1.5.0
meta:
  - label: 명령
    value: libreyolo profile
    mono: true
  - label: 출력
    value: runs/profile 아래의 profile.json과 profile_trace.json
    mono: true
snippets:
  examples:
    - label: 추론 측정
      language: bash
      code: |
        # source 인자를 주지 않으면 함께 제공되는 샘플 이미지를 사용합니다.
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: 판정 읽기
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: 두 측정 결과 비교
      language: bash
      code: >
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project
        runs/profile/a

        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4
        --project runs/profile/b


        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
source_hash: b967e869fd9ba418
---

## 사용법

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

이 그룹은 `key=value` 인자를 받지 않습니다. 하위 명령은 위치 인자와 POSIX
플래그를 사용하므로 `--weights LibreYOLO9t.pt`이지 `weights=LibreYOLO9t.pt`가
아닙니다. 하위 명령 없이 `libreyolo profile`을 실행하면 목록을 출력합니다.

하위 명령 두 개가 측정을 수행하고 프로파일을 기록하며, 나머지는 그 프로파일을
읽습니다. `run`과 `infer`는 모두 동일한 자체 완결형 `profile.json`을 내놓기
때문에, 읽기 쪽 하위 명령은 어느 쪽에서든 동작합니다.

## profile run

짧은 학습을 프로파일링하면서 실행하고 프로파일을 기록합니다.

```bash
libreyolo profile run <data> [--flag value ...]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `data` | | 위치 인자. 데이터셋 YAML 또는 이름, 예를 들어 `coco128`. 필수 |
| `--weights` | `LibreYOLO9t.pt` | 모델 가중치 파일 또는 이름 |
| `--size` | `t` | 모델 크기 변형 |
| `--batch` | `16` | 마이크로 배치. `-1`은 VRAM의 약 70%에 맞춰 자동으로 정합니다 |
| `--imgsz` | `640` | 학습 이미지 크기 |
| `--workers` | `8` | 데이터로더 워커 수 |
| `--amp` | `true` | 모델 계열의 AMP 경로를 사용합니다. `--no-amp`로 비활성화합니다 |
| `--steps` | `20` | 프로파일링되는, 즉 측정되는 스텝 수 |
| `--warmup` | `5` | 측정 전 워밍업 스텝 수 |
| `--repeat` | `1` | 평균과 표준편차를 얻기 위해 N회 반복 |
| `--device` | `0` | 장치 |
| `--project` | `runs/profile` | 출력 디렉터리 루트 |
| `--json` | `false` | stdout으로 JSON 출력 |

측정 구간은 `--warmup`에 `--steps`를 더한 만큼의 반복입니다. 그 구간을 채우기에
너무 작은 데이터셋은 프로파일을 만들지 못하고 명령이 코드 `3`으로 종료하면서,
빠져나갈 세 가지 방법을 알려 줍니다: 더 큰 데이터셋, 더 적은 스텝, 또는 더 작은
배치.

`--repeat` 값이 1보다 크면 집계된 `runs/profile/profile_repeat.json`을 기록하는데,
그 안의 스칼라 지표는 여러 시행에 걸쳐 평균을 낸 값이고 커널 목록은 마지막
시행에서 가져옵니다. 이 값은 `compare`에서 유의성 판정을 받기 위한 전제
조건이기도 합니다: 단일 실행은 판정을 내줄 수 없습니다.

## profile infer

추론 경로를 프로파일링하고 프로파일을 기록합니다.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `source` | | 위치 인자. 이미지 또는 디렉터리. 생략하면 함께 제공되는 샘플 이미지 |
| `--weights` | `LibreYOLO9t.pt` | 모델 가중치 파일 또는 이름 |
| `--size` | `t` | 모델 크기 변형 |
| `--batch` | `1` | 순전파 한 번당 이미지 수 |
| `--imgsz` | `640` | 입력 이미지 크기 |
| `--half` | `false` | 순전파를 오토캐스트로 실행하며 CUDA에서만 동작합니다. `--no-half`로 비활성화합니다 |
| `--amp-dtype` | `float16` | CUDA 오토캐스트 dtype: `float16` 또는 `bfloat16` |
| `--warmup` | `20` | 측정 전 워밍업 반복 횟수 |
| `--runs` | `100` | 측정되는 반복 횟수 |
| `--repeat` | `1` | 평균과 표준편차를 얻기 위해 N회 반복 |
| `--conf` | `0.25` | 신뢰도 임계값이며, NMS가 하는 작업량을 바꿉니다 |
| `--iou` | `0.45` | NMS IoU 임계값 |
| `--max-det` | `300` | 이미지당 최대 탐지 수이며, NMS가 하는 작업량을 바꿉니다 |
| `--device` | `0` | 장치 |
| `--trace` | `true` | 커널과 op를 파고들 수 있도록 Chrome 트레이스를 내보냅니다. `--no-trace`로 건너뜁니다 |
| `--project` | `runs/profile` | 출력 디렉터리 루트 |
| `--json` | `false` | stdout으로 JSON 출력 |

p50, p90, p99 지연 시간과 초당 이미지 단위의 처리량, 그리고 전처리와 순전파,
후처리로 나뉜 단계별 비중을 보고합니다. 임계값 인자 세 개가 여기에 있는 이유는
그것들이 후처리 수치를 움직이기 때문입니다.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `trace` | | 위치 인자. `profile.json` 또는 `profile_trace.json` 경로. 필수 |
| `--json` | `false` | stdout으로 JSON 출력 |

가장 높은 층위의 읽기입니다: 스텝 시간, 처리량, GPU 사용률, Tensor Core 비중,
최대 VRAM, 호스트 오버헤드, 스텝당 커널 실행 횟수, 이유가 붙은 병목 판정,
카테고리별 커널 구성, 그리고 스텝당 상위 커널을 보여 줍니다. 추론 프로파일에서는
지연 시간 백분위수와 단계별 비중도 함께 출력합니다.

VRAM 스래싱 상태에서 채취한 프로파일에는 표시가 붙는데, 그런 상태에서 측정한
사용률과 처리량은 믿을 수 없기 때문입니다.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `trace` | | 위치 인자. 프로파일 경로. 필수 |
| `field` | | 위치 인자. 지표 이름. 생략하면 사용 가능한 지표를 나열합니다 |
| `--json` | `false` | stdout으로 JSON 출력 |

스크립트 루프에서 쓰도록 지표 하나만 출력하고 그 외에는 아무것도 출력하지
않습니다. 알 수 없는 필드는 코드 `2`로 종료하면서 목록 형태를 안내합니다.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `trace` | | 위치 인자. 프로파일 경로. 필수 |
| `--json` | `false` | stdout으로 JSON 출력 |

단계별 GPU 밀리초, 실제 경과 밀리초, 커널 수와 op 수입니다: forward, backward,
dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `trace` | | 위치 인자. 프로파일 경로. 필수 |
| `--top` | `20` | GPU 시간 기준 상위 N개 표시 |
| `--category` | | 카테고리 부분 문자열로 필터링: `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | 커널 이름 정규 표현식으로 필터링 |
| `--tensorcore` | `false` | Tensor Core 커널만 |
| `--sort` | `time` | `time`, `count` 또는 `name` |
| `--phase` | | 한 단계로 제한: `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | stdout으로 JSON 출력 |

분석의 가장 아래층입니다: 개별 GPU 커널과 그 커널의 GPU 시간 비중, 스텝당
밀리초, 스텝당 호출 횟수, 카테고리를 보여 줍니다. 알 수 없는 `--phase` 값은 코드
`2`로 종료하면서 그 프로파일에 있는 단계를 나열합니다.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `trace` | | 위치 인자. 프로파일 경로. 필수 |
| `--top` | `20` | CPU 시간 기준 상위 N개 표시 |
| `--phase` | | 한 단계로 제한 |
| `--json` | `false` | stdout으로 JSON 출력 |

장치 관점이 아니라 프레임워크 관점입니다: CPU 시간 기준으로 순위를 매긴 `aten`과
autograd op이며, 호스트 실행 비용이 드러나는 곳이 바로 여기입니다.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `before` | | 위치 인자. 베이스라인 프로파일. 필수 |
| `after` | | 위치 인자. 새 프로파일. 필수 |
| `--json` | `false` | stdout으로 JSON 출력 |

처리량, 이미지당 밀리초, GPU 사용률, 호스트 오버헤드, 스텝당 커널 실행 횟수,
병목 판정의 차이를 보여 줍니다.

유의성 판정에는 양쪽 모두 `--repeat` 값을 최소 2로 두고 측정한 결과가 필요합니다.
그 조건이 갖춰지면 차이가 결합 표준오차의 두 배를 넘을 때 유의한 것으로 보고,
출력에는 어떤 비교를 했는지가 함께 표시됩니다. 그 조건이 없으면 단일 실행으로는
판정을 뒷받침할 수 없다는 문구가 나옵니다.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `trace` | | 위치 인자. 프로파일 경로. 필수 |
| `--remove-category` | | 커널 카테고리를 제거했을 때를 추정: `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | 스텝당 커널 실행을 N회 줄였을 때를 추정, 예를 들어 op 융합으로 얻는 이득 |
| `--json` | `false` | stdout으로 JSON 출력 |

변경을 실제로 작성하기 전에 그 변경이 무엇을 벌어 줄지 추정합니다. 두 옵션 중
하나는 필수이며, 둘 다 없으면 코드 `2`로 종료합니다.

이 추정은 프로파일 자체의 판정을 따릅니다. GPU 사용률이 80% 미만이면 절감량을
줄어든 실행 횟수에 측정된 실행 1회당 호스트 비용을 곱한 값으로 모델링하고, 그
위에서는 줄어든 GPU 작업량으로 모델링합니다. 결과에는 주의 사항 필드가 함께
담기는데, 실행 1회당 비용은 근삿값이고 유일한 증명은 두 번째 측정이기
때문입니다.

## 예제

<code-tabs name="examples" />

## 참고

프로파일러는 측정하고 보고합니다. 아무것도 바꾸지 않습니다: 판정을 읽고, 설정이나
코드를 고치고, 다시 실행하고, 비교하는 것이 이 도구가 상정한 반복 과정입니다.

`--device`의 기본값은 `0`, 즉 CUDA 장치 0입니다. `--device cpu`를 주면 CPU에서
측정하며, GPU 커널 세부 정보는 빠지지만 읽기 쪽 하위 명령이 그대로 받아들이는
프로파일을 만듭니다.

모든 하위 명령이 `--json`을 지원하고 읽기 쪽 명령은 stdout으로만 출력하는데, 이
점이 이 그룹을 스크립트에서 쓸 수 있게 만듭니다.

여기서 쓰는 종료 코드는 이 그룹 고유의 것입니다: 존재하지 않는 파일이나 해석되지
않는 인자에는 `2`, `run`이 프로파일을 만들지 못했을 때는 `3`, 트레이스를 분석할
수 없을 때는 `1`입니다.

관련 문서: 학습 프로파일은 보통 [`libreyolo train`](/docs/cli/train)의 인자를
조정하려고 채취합니다.
