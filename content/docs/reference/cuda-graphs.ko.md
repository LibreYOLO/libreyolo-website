---
title: CUDA 그래프
seo_title: LibreYOLO CUDA 그래프 지원 매트릭스
description: >-
  예측 시점에 순방향을, 학습 시점에 순방향과 역방향을 포착하는 계열는 어떤 것이며, 숫자가 보장되는 경우, 포착이 나뉘는 위치, 그리고
  지원되지 않는 계열가 예외를 발생시키는 이유가 무엇인지.
lead: >-
  CUDA 그래프는 고정된 커널 시퀀스의 한 번의 실행을 기록하고 이를 단일 실행으로 재생합니다. LibreYOLO는 39개의 검증된 계열에서
  추론을, 24개의 계열에서 학습을 항상 계열별로 캡처하며, 항상 비트 단위 패리티 검사를 거친 후 수행하고, 절대 조용한 대체로 수행하지
  않습니다.
keywords:
  - 리브레욜로 쿠다 그래프
  - cuda_graph=True
  - CUDA 그래프 지원 매트릭스
  - 토치 CUDA 그래프 학습
  - capture_error_mode 스레드 로컬
  - cuda 그래프 비트 동일
last_verified: 1.5.0
verification: >-
  tests/e2e/test_cuda_graph_families.py의 CAPTURABLE 행렬에서 파생된 추론 계열 목록 v1.5.0에서.
  docs/training_cuda_graphs.md. API와 BaseModel._require_cuda_graph_support의
  NotImplementedError, libreyolo/models/base/model.py의 cuda_graph_scope 및
  capture_graph에서 학습 계열 목록, 패리티 클래스 및 타이밍, SUPPORTS_CUDA_GRAPH 클래스 변수와 함께.
  depth_anything3, birefnet, ppocr, sam 및 sensenova 계열에서 _get_graph_runner
  오버라이드와 libreyolo/models/base/cuda_graph.py 및
  libreyolo/training/cuda_graph.py.에서 libreyolo/models/base/detr_cuda_graph.py.
  capture_error_mode에서 읽은 심 분할. libreyolo/training/trainer.py에서의 학습 폴백과
  libreyolo/cli/commands/train.py.의 --cuda-graph 플래그.
meta:
  - label: 추론 계열
    value: '39'
  - label: 계열 학습
    value: '24'
  - label: 추론 플래그
    value: predict(cuda_graph=True)
    mono: true
  - label: 학습 플래그
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: 예측하다
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # 입력 형상당 첫 사용 시 실제 캡처.
        # "자동"은 형태가 반복될 때까지 캡처 비용을 지불하지 않고 기다립니다.
        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: 기차
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI에서 학습
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## 캡처된 것

그래프는 고정된 순서의 커널과 커널이 읽고 쓰는 메모리 주소를 기록합니다. 값, 형태 또는 제어 흐름은 기록하지 않습니다. 재실행은 수백 번이 아닌 단일 실행이기 때문에, 연산보다는 실행 오버헤드가 지배적인 작은 배치 크기에서 작은 네트워크에서 이득이 가장 큽니다.

두 개의 진입점은 서로 다른 작업량을 포착합니다.

| | 그래프 안 | 열망하는 |
|---|---|---|
| 추론 | 네트워크 전달, `model._forward(x)` | 전처리, NMS, 모든 후처리 |
| 학습 | 네트워크 순방향 및 역방향 | 손실, 옵티마이저 스텝, 그래디언트 클리핑, EMA, 학습률 스케줄 |

NMS도 탐지 손실도 후보가 될 수 없습니다. 둘 다 불리언 마스크로 선택하고, 헝가리안 매칭이나 할당기를 실행하며, 결과에 따라 분기하는데, 이것이 바로 그래프가 기록할 수 없는 것입니다. 이들을 제외하는 것이 캡처를 안전하게 만드는 것이지, 우회해야 할 제한이 아닙니다.

<code-tabs name="usage" />

`cuda_graph`는 예측 시 세 가지 값을 허용합니다. `False`가 기본값입니다. `True`는 각 입력 형태가 처음으로 나타나는 시점을 캡처합니다. `"auto"`는 형태가 반복될 때까지 기다리므로, 한 번 사용하거나 형태가 다양한 작업은 재사용하지 않을 캡처 비용을 지불하지 않습니다. `capture_graph(imgsz=None, batch=1, dtype=None)`는 비용을 첫 번째 요청에서 이동시키고, `graph_info()`는 캡처된 그래프와 재생 횟수를 보고하며, `release_graphs()`는 이를 해제합니다.

학습 시간에 플래그는 CLI상의 단순 불리언인 `--cuda-graph`입니다. 주변 컨트롤에 대해서는 [예측 성능](/docs/predict/performance)과 [학습 성능](/docs/train/performance)을 참조하십시오.

## 추론 지원

지원은 `SUPPORTS_CUDA_GRAPH` 클래스 변수로 선언된 대로 계열당 제공되며, 계열은 서로 다른 분포에서 추출된 두 개의 프로브 입력에 대해 비트까지 동일하게 캡처하고 재생한 후에만 표시됩니다. 그 공유된 패리티 행렬은 아홉 가지 작업에 걸쳐 39개의 계열을 포함합니다.

| 작업 | 계열들 |
|---|---|
| 탐지하다 | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| 세그먼트 | dfine, rtmdet, rfdetr, ec |
| 자세 | ec, yolonas, rfdetr |
| 점 | 놓칠까 봐 두려움 |
| 분류하다 | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| 의미론의 | eomt, dinov2, segformer, pidnet, lingbotvision |
| 깊이 | depth_anything, depth_anything3, zipdepth |
| 복원하다 | nafnet, realesrgan, swinir |
| 무광 | birefnet |

여러 계열이 여러 작업에 나타나므로 매트릭스는 고유한 계열 수보다 더 많은 행을 가집니다. 세 계열은 공유 매트릭스를 통해서가 아니라 자체 전용 테스트를 가진 계열별 코드 경로를 통해 캡처되며, 39개에는 포함되지 않습니다: PP-OCR, SAM 및 SenseNova.

검증은 근사값이 아니라 비트 단위입니다. 프로토콜의 이전 버전은 상대적 크기로 패리티를 판단했으며, 실제로 중요한 프로브에서 비트가 동일함에도 불구하고 eager-to-graph 차이가 약 1e-7인 세 가지 건강한 계열인 YOLOX, EfficientNetV2, YOLOv7을 잘못 하향 평가했습니다.

## 학습 지원

이번 릴리스에서는 학습 캡처가 두 계열에서 다섯 가지 작업에 걸쳐 24계열으로 확대되었습니다.

| 작업 | 계열들 |
|---|---|
| 탐지하다 | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| 분류하다 | 레스넷, 콘브넥스트, 모바일넷v4, 이피션트넷v2 |
| 의미론의 | 세그포머, 링봇비전 |
| 점 | 놓칠까 봐 두려움 |
| 복원하다 | 나프넷 |

나머지는 모두 열심히 학습합니다: 같은 계열의 다른 작업, 목록에 없는 계열, 분배 실행 및 증류 실행. 캡처는 또한 형상이 아직 새로울 때 건너뛰는데, 학습 경로가 입력 형상이 세 번 반복될 때까지 기다린 후에야 캡처하기 때문이며, 이는 `multi_scale=True`가 전혀 캡처되지 않을 수도 있음을 의미합니다.

## 지원되지 않는 계열에 대한 두 가지 다른 답변

추론 경로가 올라갑니다. 참여하지 않은 계열에 대해 `predict(cuda_graph=True)`는 계열의 이름을 지정하는 `NotImplementedError`를 발생시키며, 이는 열심히 실행하고 속도가 빨라졌다고 믿게 만드는 것 대신입니다. 그 이유는 잘못된 캡처가 명확히 실패하지 않기 때문입니다: 캡처할 수 없는 작업을 하는 앞으로의 재생은 조용히 잘못된 숫자를 반환하므로, 지원은 대체 시도의 시도가 아니라 계열별 명시적 주장이어야 합니다.

학습 경로 로그. `train(cuda_graph=True)`는 항상 안전하게 통과할 수 있으며, 캡처할 수 없는 계열, 작업 또는 구성은 한 줄을 작성하고 eager로 변경 없이 학습합니다. 실행 도중 일부가 실패한 캡처도 나머지 실행을 중단하지 않고 eager로 전환합니다. 비대칭성은 의도적입니다: 예측은 호출 지점에서 수정할 수 있는 호출인 반면, 학습 실행은 선택적 최적화 때문에 여섯 시간에 죽어서는 안 됩니다.

## 솔 갈라짐

어떤 계열은 하나의 단계가 그래프로 기록할 수 없는 일을 실제로 수행하기 때문에 전체를 포착할 수 없습니다. 계열을 포기하는 대신, 포착은 검증된 연결점에서 나뉩니다: 포착 가능한 부분은 재생되고, 나머지는 즉시 실행되며, 결합된 출력은 모든 것을 즉시 실행했을 때와 동일합니다.

| 계열 | 포착됨 | 열망하는, 그리고 왜 |
|---|---|---|
| Depth Anything 3 | 네트워크 | 앞으로 이동 후 호스트가 볼 수 있는 작업인 스카이 스텝 |
| 바이레프넷 | 인코더, `forward_enc` | 캡처 시 다른 결과를 재생하는 `deform_conv2d`를 가진 디코더 |
| PP-OCR | 탐지 단계, `forward_det` | 인식, 줄마다 자르기 폭이 다르기 때문에 |
| 샘 | 이미지 인코더 | 인코딩마다 여러 번 실행되는 프롬프트 경로 |
| 센스노바 | 비전 타워 | 자동회귀 생성, 매 단계마다 커지는 KV 캐시와 함께 |
| 인코더-디코더 검출기 | 백본 및 인코더 | 디코더와 헝가리안 기준 |

BiRefNet 분할은 두 번 읽을 만합니다: 캡처 중에 문제가 발생한 `deform_conv2d`는 어떤 모델 외부의 빈 호출에서도 재현됩니다. 이를 순수 PyTorch 동등물로 교체하는 것은 즉시 실행 예측(eager predictions)도 바뀌게 되므로 거부되었습니다. 즉시 실행 수치가 계약이기 때문입니다.

인코더-디코더 사례에는 D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4 및 EC가 포함됩니다. 그들의 디코더는 실제값으로부터 대비-노이즈 제거 쿼리를 구성하며, 이러한 쿼리의 수는 배치에서 가장 큰 실제값 수에서 나오므로 디코더의 토큰 수는 배치마다 달라집니다. 이것이 그래프가 견딜 수 없는 한 가지입니다. 백본과 인코더를 합친 것은 이러한 계열에서 단계의 약 5분의 1에서 4분의 1 정도이므로, 이들이 속도 향상 표에서 하위에 위치하는 이유입니다.

PP-OCR은 러너의 캐시 한도에 의해 제한된 각 탐지 입력 형태마다 하나의 그래프를 캡처하며, 캡처 범위가 활성 상태가 아닐 때 즉시 결과를 반환합니다.

## 숫자

대부분의 계열은 비트 단위로 동일하며, 동일하지 않은 경우에는 손을 흔드는 대신 원인이 명확히 지적됩니다. 학습의 초기 단계에서 모든 24개 계열의 손실은 비트 단위로 동일하며, BatchNorm 버퍼도 다르지 않습니다. 카테고리를 구분하는 것은 그래디언트 비교입니다.

| 수업 | 계열들 | 의미 |
|---|---|---|
| 정확한 | 24 중 대부분 | 모든 그래디언트가 바이트 단위로 동일함 |
| 1 ULP | 포모, 링봇비전 | 다른 합산 순서에서 오는 float32의 마지막 조각, 약 1e-7 상대적 |
| 열성적인 소음 | DETR 계열 | 그래프는 열성(eager)과 서로 다른 두 개의 열성 실행이 서로 다른 것과 마찬가지로 차이가 난다 |
| 부동 소수점 반올림 | rtmdet | 139개의 그래디언트 중 137개는 비트 단위로 동일하며, 두 개는 약 3e-4 정도 차이가 납니다 |
| 자체 RNG 스트림 | 세그포머 | 확률적 깊이는 캡처된 영역 안에 있습니다 |

열망-노이즈 클래스는 올바르게 읽는 것이 중요한 클래스입니다. 그 계열들의 경우, 이미 두 개의 시드된 열망 실행이 서로 다르기 때문에, 비트 동일성이 그래프 실행이 실패한 기준이 아니며; 아무것도 통과하지 못하는 기준을 의미합니다. 이 현상은 `amp=False`에서는 더 넓게 나타나는데, fp32 가중치 그래디언트에서 측정된 3.2e-7 상대 비결정성이 누적되기 때문입니다: 시드된 두 개의 열망 YOLOv9-t 실행은 20단계에서 36퍼센트까지 달라지며, TF32를 끄더라도 해결되지 않습니다.

## 핀 메모리

`capture_error_mode="thread_local"`로 실행을 캡처합니다. PyTorch의 기본 `"global"` 모드에서는, 다음 배치를 준비하는 DataLoader 핀 메모리 스레드가 `cudaHostAlloc`를 호출하며, 이는 진행 중인 캡처를 모두 무효화하고 그로 인해 오염되기 때문에, 다음 배치를 가져올 때 핀 메모리 스레드 내부에서 발생한 오류로 실행이 종료됩니다. 이 조합은 진단되기 전에 실제 학습 캠페인에서 두 번 관찰되었습니다.

스레드 로컬 모드는 캡처하는 스레드만 제한합니다. 핀 스레드는 캡처 스트림을 절대 건드리지 않기 때문에, 그것이 하는 어떤 것도 처음부터 그래프에 속하지 않습니다. 학습은 더 나아가 일시적으로 모드를 강제하는 `torch.cuda.CUDAGraph` 하위 클래스를 대체합니다. 왜냐하면 `make_graphed_callables`는 이를 위한 인자를 제공하지 않기 때문이며, 두 개의 동시 캡처가 대체를 남기지 않도록 잠금 아래에서 수행됩니다.

## 그 가치가 무엇인지

RTX 5070 Ti에서 AMP 하에 측정했으며, 각 팔에 한 프로세스씩, 데이터 로더를 우회하여 한 실제 배치를 재생합니다. 워밍업 후 24단계 중 가장 빠릅니다. 탐지는 640 px, 분류는 224 px에서 수행됩니다.

| 계열 | 배치 | 속도 향상 |
|---|---:|---:|
| 놓칠까 봐 두려움 | 16 | 3.63배 |
| MobileNetV4 s | 16 | 2.74배 |
| EfficientNetV2 b0 | 16 | 2.44배 |
| YOLOv9-t | 8 | 1.99배 |
| YOLOv9 e2e | 8 | 1.76배 |
| YOLOv9 p2 | 8 | 1.49배 |
| 그 밖의 모든 것 | 다르다 | 1.04배에서 1.26배 |

전체 실행은 더 적은 이익을 얻습니다. 그래프는 데이터 로더나 검증을 가속할 수 없기 때문입니다. 406개의 이미지에 대해 20에포크 YOLOv9-t 파인튜닝을 진행한 결과, 시간은 428.4초에서 367.7초로 줄어들어 1.16배의 엔드 투 엔드 향상을 보였으며, 두 가지 방법 모두 mAP50-95가 0.6394로 동일했고 에포크별 손실도 동일했습니다.

천장 성능은 네트워크가 한 단계에서 차지하는 비율에 의해 결정됩니다. 동일한 하드웨어에서 640px과 배치 8일 경우, YOLOv9-t는 84퍼센트에 달하지만 RTMDet-t는 단지 26퍼센트에 불과합니다. RTMDet-t는 대부분의 단계에서 레이블 할당기를 사용하기 때문입니다. 실행 오버헤드는 Windows에서 가장 높으며, Linux에서는 이 표의 대략 3분의 1에서 절반 수준의 성능을 보여줍니다. 데이터로더가 병목인 경우 실제 실행 시간에는 변화가 전혀 없습니다. 최대 메모리는 5퍼센트 낮아질 수도 있고, 19퍼센트 높아질 수도 있습니다.

## 주의 사항

그래프는 값을 기록하는 것이 아니라 주소를 기록하므로, 매개변수를 재배치하는 모든 것은 그래프를 손상시킵니다. `predict(device=...)`를 통해 장치를 변경하고, 양자화 및 역양자화를 수행하면 캡처된 모든 그래프가 무효화됩니다.

배치 크기는 계열보다 더 중요합니다: RT-DETR-r18은 배치 2에서 1.19배, 배치 8에서 1.04배의 향상을 얻습니다. 큰 배치는 계산 제약을 더 많이 받고 제거할 런치 오버헤드가 적기 때문입니다.

추론 패리티 스위트는 선택적 `kernels` 패키지가 설치되지 않은 상태에서 실행되었으므로, 컴파일된 Hub 커널이 활성화된 상태에서의 캡처 안전성은 이 스위트에 포함되지 않습니다. 캡처 문제를 격리하는 동안 그것들을 제거하려면 `LIBREYOLO_HUB_KERNELS=0`를 설정하십시오. [커널](/docs/reference/kernels)을 참조하십시오.
