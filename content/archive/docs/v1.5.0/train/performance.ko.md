---
title: 학습 성능
seo_title: '더 빠른 학습: CUDA 그래프, AMP, 프로파일러'
description: >-
  학습 실행을 더 빠르게 하려면: 단계를 CUDA 그래프로 캡처하고, AMP 데이터 유형을 선택하며, 내장 프로파일러를 사용하여 시간이 실제로
  어디에 쓰이는지 확인하십시오.
lead: >-
  세 가지 요소가 학습 단계의 실행 속도를 바꿉니다: 혼합 정밀도, 네트워크 순전파와 역전파의 CUDA 그래프 캡처, 프로파일러가 실제로
  단계를 지연시킨다고 보고하는 항목.
keywords:
  - 쿠다 그래프 학습
  - 학습 속도
  - 혼합 정밀도 학습
  - bfloat16 학습
  - 파이토치 프로파일러
  - 데이터로더 바운드
  - 커널 실행 오버헤드
  - GPU 사용률
last_verified: 1.5.0
snippets:
  profile:
    - label: 프로필을 살펴보고 계속 학습하십시오
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 실제 단계의 짧은 창을 프로파일링하고, 판정을 출력한 다음
        # 갈고리를 제거한 상태로 실행을 계속합니다.
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: '측정만 하고, 그만두십시오'
      language: bash
      code: |
        # no_aug_epochs=0으로 설정하고 창을 채우기에 충분한 에포크만 실행합니다.
        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: 결과를 파고들다
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## 변경 전 측정

아래의 세 가지 레버는 서로 다른 문제를 해결하며, 잘못된 것을 적용하면 아무것도 바뀌지 않습니다. 프로파일러가 어떤 문제가 있는지 알려줍니다.

<code-tabs name="profile" />

`profile=True`는 실제 학습 단계의 창을 측정하며, 기본적으로 다섯 단계는 버리고 그 다음 스무 단계를 측정한 후 리포트를 출력하고 아티팩트를 기록한 다음 훅을 제거하고 학습을 계속합니다. 꺼져 있을 때는 비용이 들지 않으며, 분산 학습에서는 무시됩니다.

보고서는 네 가지 판결 중 하나로 끝납니다:

| 평결 | 의미 | 지렛대 |
|---|---|---|
| `dataloader` | GPU가 입력 데이터를 기다린다 | 더 많은 `workers`, `cache="ram"` 또는 `"disk"`, 더 가벼운 증강, 더 큰 배치 |
| `host / launch` | GPU에 공급되는 속도가 너무 느리고, 작은 커널이 많습니다 | 더 큰 배치, CUDA 그래프, 단계별 호스트 동기화 감소 |
| `compute` | GPU가 포화 상태입니다 | AMP 또는 bfloat16, 아니면 받아들인다 |
| `memory-pressure` | 할당자 과부하, VRAM이 거의 다 찼음 | 하위 배치; 여기의 활용 수치는 신뢰할 수 없습니다 |

이용률 수치는 커널 바쁜 시간(kernal busy time)을 비동기 단계 시간(unsynchronized step time)으로 나눈 것입니다. 윈도우는 의도적으로 나뉘어 있습니다: 첫 번째 절반은 추가 동기화 없이 실행되어 평가가 실제 겹침을 반영하고, 두 번째 절반만 각 단계를 동기화(sync)로 감싸 GPU 시간을 할당합니다. 모든 단계를 동기화하면 데이터로더 워커가 여유(slack)를 가지게 되고 기아(starvation)가 숨겨지므로, 조합 수치(composition numbers)는 평가(verdict)를 선택하는 데 사용되지 않습니다.

네 개의 파일이 실행 디렉터리에 생깁니다: `timeline.html`는 스스로 브라우저에서 열리고, `profile_trace.json`는 Perfetto 또는 Nsight용이며, `profile_summary.json`와 `profile.json`는 독립형으로 복사해서 `libreyolo profile` 하위 명령어에 다시 제공할 수 있습니다.

`profile run`에 대해 알아둘 만한 두 가지가 있습니다. 이것은 `no_aug_epochs=0`를 설정하는데, 그 이유는 프로파일러가 0 에폭을 측정하고 기본 `no_aug_epochs`로 짧게 실행하면 실제로 학습에서 사용하는 것보다는 가벼운 데이터로더를 프로파일링하기 때문입니다. 그리고 `--repeat N`는 평균과 표준편차를 보고하는데, 이는 실행 경계 단계가 충분히 노이즈가 많아서 단일 실행만으로는 오해를 일으킬 수 있기 때문입니다; 각 시도 디렉터리 `prof_1`, `prof_2` 등을 기록하고, 또한 전체 집계 `profile_repeat.json`도 기록합니다.

## 혼합 정밀도

`amp=True`는 대부분의 계열에 대한 기본값이며 CUDA 자동 캐스트에서 순방향 패스를 실행합니다. `amp_dtype`는 `float16` 또는 `bfloat16`를 선택합니다.

<code-tabs name="amp" />

Float16는 동적 손실 스케일링이 필요하며 라이브 그래디언트 스케일러를 사용합니다; bfloat16은 더 넓은 지수 범위를 가지고 있어 필요하지 않으므로 스케일러가 비활성화됩니다. `amp=False`, D-FINE, DEIM, YOLO-NAS, FOMO와 함께 네 가지 계열이 제공되며, DEIM 설정은 상속을 통해 RT-DETRv4로 전달됩니다. D-FINE은 이유를 명시합니다: 디코더가 활성화를 65504에서 클램핑하는데, 이는 가장 큰 유한 float16 값입니다.

bfloat16 요청이 bfloat16을 지원하지 않는 하드웨어에서 수행하는 작업을 포함한 인수 의미론은 [Hyperparameters](/docs/train/hyperparameters)에 있습니다.

## CUDA 그래프

`cuda_graph=True`는 네트워크의 학습 전방향 및 역방향을 CUDA 그래프로 캡처하여 단계별 커널 실행 오버헤드를 제거합니다.

<code-tabs name="graph" />

깃발은 항상 안전하게 전달할 수 있습니다. 잡을 수 없는 계열, 작업 또는 구성은 한 줄을 기록하고 열심히, 변함없이 학습합니다.

네트워크만 캡처됩니다. 손실은 설계상 계속 즉시 계산(eager)됩니다. 왜냐하면 탐지 손실은 불리언 마스크로 선택하고, 헝가리안 매칭을 수행하며, 할당 결과에 따라 분기하기 때문이며, 이들 중 어느 것도 그래프가 기록할 수 없습니다. 옵티마이저 단계, 그래디언트 클리핑, EMA 업데이트 및 학습률 스케줄도 계속 즉시 계산됩니다.

이는 네트워크가 얼마나 많은 단계를 차지하는지에 따라 승리를 제한하며, 그 비율은 크게 다릅니다. 640px에서 RTX 5070 Ti로 측정한 배치 8 기준으로, YOLOv9-t 단계의 84%가 네트워크, YOLOv7-b 단계의 44%, YOLOX-t 단계의 31%, RTMDet-t 단계의 26%가 네트워크입니다. 마지막 두 모델은 단계의 대부분을 레이블 할당기 안에서 보내므로, 네트워크를 포착하는 것이 그들에게는 가장 적은 도움을 줍니다.

### 그 가치가 무엇인지

아래 모든 그림의 조건: RTX 5070 Ti, Windows, AMP, 공유된 저장 상태에서 한 팔당 하나의 프로세스, 데이터 로더를 생략하고 실제 배치 하나 재생, 워밍업 후 24단계 중 가장 빠른 속도. 탐지는 640 px, 분류는 224 px. 배치 크기는 행당 기준.

| 계열 | 크기 | 배치 | 열망하는 | 그래프로 나타낸 | 속도 향상 |
|---|---|---:|---:|---:|---:|
| 놓칠까 봐 두려움 | s | 16 | 7.0 밀리초 | 1.9 밀리초 | 3.63배 |
| 모바일넷V4 | s | 16 | 14.5 밀리초 | 5.3 밀리초 | 2.74배 |
| EfficientNetV2 | b0 | 16 | 29.0 밀리초 | 11.9 밀리초 | 2.44배 |
| YOLOv9 | t | 8 | 93.6 밀리초 | 47.0 밀리초 | 1.99배 |
| NAFNet | s | 8 | 132.5 밀리초 | 105.5 밀리초 | 1.26배 |
| 파이코데트 | s | 8 | 145.0 밀리초 | 118.7 밀리초 | 1.22배 |
| 디-파인 | n | 4 | 185.3 밀리초 | 159.2 밀리초 | 1.16배 |
| RF-DETR | n | 4 | 276.3 밀리초 | 239.8 밀리초 | 1.15배 |
| YOLOX | t | 8 | 102.2 밀리초 | 90.5 밀리초 | 1.13배 |
| RTMDet | t | 8 | 149.7 밀리초 | 136.2 밀리초 | 1.10배 |
| YOLOv7 | b | 4 | 102.5 밀리초 | 98.0 밀리초 | 1.05배 |

그것들은 GPU 단계를 분리합니다. 완전한 파인튜닝은 또한 데이터로더와 검증에도 비용이 듭니다. 동일한 머신에서 406 이미지 탐지 세트에 YOLOv9-t, 20 에포크, 배치 8, 640 px, 데이터로더 워커 4명,을 적용했을 때: 깨어 있는 상태(eager)의 벽시계 시간은 428.4초, 그래프 상태(graphed)는 367.7초로 1.16배의 성능 향상이 있으며, 두 경우 모두 mAP50-95는 0.6394입니다.

이 숫자들을 움직이는 세 가지가 있습니다. 작은 배치는 출시 준비와 관련이 있고 큰 배치는 계산량과 관련이 있어서, RT-DETR-r18은 배치 2에서 1.19배, 배치 8에서 1.04배를 얻습니다. 출시 오버헤드는 Windows에서 가장 높으며, Linux의 이득은 표의 약 3분의 1에서 절반 정도입니다. 그리고 데이터로더가 병목인 실행에서는 실제 시간에는 변화가 전혀 없기 때문에 프로파일러가 먼저 나오는 것입니다.

Capture는 `amp=False`에서도 동일하게 작동하지만, fp32 커널은 실행 시간이 더 길기 때문에 한 단계는 런치에 덜 제한되고 대부분의 계열는 이득이 적습니다. 동일한 하드웨어에서, 배치 16의 MobileNetV4-s는 AMP에서 2.74배였던 것이 fp32에서는 3.61배로 증가하며, 배치 8의 YOLOv9-t는 1.99배에서 1.69배로, 배치 4의 RT-DETR-r18은 1.12배에서 0.99배로 변합니다.

### 캡처가 적용되는 곳

| 작업 | 계열들 |
|---|---|
| 탐지하다 | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| 분류하다 | 레스넷, 콘브넥스트, 모바일넷v4, 이피션트넷v2 |
| 의미론의 | 세그포머, 링봇비전 |
| 점 | 놓칠까 봐 두려움 |
| 복원하다 | 나프넷 |

나머지 모든 것은 한 줄의 로그와 함께 eager로 되돌아갑니다: 해당 계열의 다른 작업들, 목록에 없는 계열들, 분산 실행과 증류 실행입니다. 런타임에서 캡처 실패가 발생해도 나머지 실행은 실패하지 않고 eager로 전환됩니다.

인코더-디코더 검출기인 D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 및 v4, 그리고 EC의 경우, 백본과 인코더만 캡처됩니다. 이들의 디코더는 대조적 잡음 제거 쿼리를 만들기 위해 실제 값을 읽으며, 이러한 쿼리의 수는 배치 내 가장 큰 실제 값 수를 따르므로, 토큰 수가 배치마다 달라집니다.

### 모양

그래프는 정확히 캡처된 입력 형태에 대해서만 유효합니다. 트레이너는 배치 형태를 세고, 하나의 형태가 세 번 반복되면 캡처합니다. 다른 형태의 배치는 즉시 실행되며: 다중 스케일 배치와 에포크의 마지막 부분 배치가 포함됩니다.

이것은 기본적으로 모든 배치를 리사이즈하는 DETR 계열을 위한 함정입니다. `multi_scale=True`를 사용하면 짧은 실행에서는 하나의 형태를 충분히 자주 볼 수 없어 전혀 캡처하지 못할 수 있습니다. 속도 향상이 목적일 때에는 `multi_scale=False`를 사용하십시오.

YOLOX는 실행 도중 캡처된 영역이 계산하는 내용을 변경하며, 모자이크가 `no_aug_epochs`에서 닫힐 때 L1 회귀 브랜치를 켭니다. 트레이너는 그 지점에서 캡처를 무효화하고 새로운 형태가 안정되면 다시 캡처합니다.

### 숫자와 메모리

대부분의 계열들은 AMP 하에서 그들의 열성적인 손실 경로를 조금씩 재현합니다. FOMO와 LingBot-Vision은 서로 다른 합산 순서 때문에 float32의 마지막 비트에서 차이가 납니다. 변형 가능한 주의 탐지기인 D-FINE, DEIM, DEIMv2, RT-DETR, RF-DETR 및 EC도 자신의 열성 실행을 재현하지 못하는데, 이는 백워드가 원자 연산으로 누적되고 TF32 합성곱이 실행마다 축소 순서를 선택하기 때문입니다; 그래프화된 실행은 그 분포 안에 머물러 있습니다. RTMDet는 피라미드 레벨 전체에서 헤드 합성곱을 공유하고, 두 개의 백워드 경로가 세 가지 기여를 서로 다른 순서로 합산하기 때문에 139개의 그래디언트 중 두 개에서 약 3e-4 정도의 상대적 차이를 보입니다. SegFormer는 캡처된 영역 안에 확률적 깊이를 가지고 있으므로, 재생된 그래프는 자체적인 랜덤 스트림을 그리고 통계적으로 동일하지 않고 즉시 실행(eager)과 동등합니다; 매니저는 캡처 시점에 한 번 이를 기록합니다.

`amp=False`에서 비트 동일성(bit-identical)은 캡처 여부와 관계없이 이 하드웨어의 어떤 것에서도 사용할 수 없습니다. 동일한 시드로 초기화된 두 개의 즉시 실행(eager) YOLOv9-t 실행은 20단계 동안 상대적으로 36퍼센트 차이가 나며, YOLOX-t는 2.6퍼센트 차이를 보입니다. 이는 cuDNN이 일부 fp32 컨볼루션 형태에 대해 비결정적(non-deterministic) 가중치-기울기 알고리즘을 선택하기 때문입니다.

캡처된 그래프는 정적 입력, 출력 및 워크스페이스 버퍼를 고정하므로, VRAM 피크가 대략 한 세트의 추가 활성화만큼 상승합니다. 위의 다양한 모델 계열에서, 피크 할당량은 -5%에서 +19% 사이로 이동했습니다. 상대적인 비용은 처음부터 활성화가 작은 소형 분류 모델에서 가장 큽니다: 224px의 ResNet-18, 배치 16, eager 모드에서 0.48GB였던 것이 graphed 모드에서 0.57GB로 증가했습니다. 실행이 한도를 초과한다면, 배치를 줄이거나 해당 플래그를 끄십시오.

## 관련된

- `batch`, `nbs`, `cache` 및 `workers`의 [하이퍼파라미터](/docs/train/hyperparameters).
- [멀티 GPU 학습](/docs/train/multi-gpu), CUDA 그래프와 프로파일러를 모두 사용할 수 없는 경우.
- [CUDA 그래프](/docs/reference/cuda-graphs)는 결합된 추론 및 학습 지원 행렬, 솔기 분할 및 수치 축소를 위해 사용됩니다.
