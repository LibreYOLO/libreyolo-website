---
title: CUDA 그래프
seo_title: LibreYOLO CUDA 그래프 지원 매트릭스
description: >-
  어떤 패밀리가 예측 시에 forward를 캡처하고 학습 시에 forward와 backward를 캡처하는지, 숫자가 보장되는 값, 캡처가
  분리되는 위치, 지원되지 않는 패밀리가 오류를 발생시키는 이유
lead: >-
  CUDA 그래프는 고정된 커널 시퀀스의 한 실행을 기록하고 이를 단일 실행으로 재생합니다. LibreYOLO는 39개의 검증된 패밀리에서
  추론을, 24개의 패밀리에서 학습을 각 패밀리마다 항상, 비트 단위 패리티 검사 후에만 캡처하며, 절대 무음 폴백으로 하지 않습니다.
keywords:
  - libreyolo cuda 그래프
  - cuda_graph=True
  - cuda 그래프 지원 매트릭스
  - torch cuda 그래프 학습
  - capture_error_mode thread_local
  - cuda 그래프 비트 동일
last_verified: 1.5.0
verification: >-
  tests/e2e/test_cuda_graph_families.py의 CAPTURABLE 매트릭스에서 파생된 추론 계열 목록 v1.5.0.
  docs/training_cuda_graphs.md. API의 학습 계열 목록, 패리티 클래스 및 타이밍,
  BaseModel._require_cuda_graph_support에서의 NotImplementedError, zxqp10q2xzp의
  cuda_graph_scope 및 capture_graph, SUPPORTS_CUDA_GRAPH 클래스 변수. depth_anything3,
  birefnet, ppocr, sam 및 sensenova 계열의 _get_graph_runner 오버라이드와
  libreyolo/models/base/detr_cuda_graph.py., libreyolo/models/base/cuda_graph.py
  및 libreyolo/training/cuda_graph.py.의 capture_error_mode에서 읽은 솔기 분할.
  libreyolo/training/trainer.py의 학습 폴백 및 libreyolo/cli/commands/train.py.의
  --cuda-graph 플래그.
meta:
  - label: 추론 계열
    value: '39'
  - label: 학습 계열
    value: '24'
  - label: 추론 플래그
    value: predict(cuda_graph=True)
    mono: true
  - label: 학습 플래그
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: 예측
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # 입력 형상별 최초 사용 시 실제 캡처
        # "auto"는 캡처 비용을 지불하기 전에 형상이 반복될 때까지 대기함.
        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: 학습
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

## 캡처된 내용

그래프는 고정된 순서의 커널과 이들이 읽고 쓰는 메모리 주소를 기록합니다. 값, 형태 또는 제어 흐름은 기록하지 않습니다. 재실행은 수백 번 대신 단일 실행이기 때문에 이득이 가장 작은 네트워크와 작은 배치 크기에서 큰데, 여기서는 한 단계가 산술보다는 실행 오버헤드에 의해 지배됩니다.

두 진입점은 서로 다른 양의 작업을 캡처합니다.

| | 그래프 내부 | 즉시 실행(Eager) |
|---|---|---|
| 추론(Inference) | 네트워크 순전파, `model._forward(x)` | 전처리, NMS, 모든 후처리 |
| 학습 | 네트워크 순전파 및 역전파 | 손실, 옵티마이저 단계, 그래디언트 클리핑, EMA, 학습률 스케줄 |

NMS나 검출 손실은 후보가 될 수 없습니다. 둘 다 불린 마스크로 선택하고, 헝가리안 매칭이나 할당자를 실행하며, 결과에 따라 분기하는데, 이것은 그래프가 기록할 수 없는 정확한 동작입니다. 이들을 제외하는 것이 캡처를 안전하게 만드는 것이지, 해결해야 할 제한 사항이 아닙니다.

<code-tabs name="usage" />

`cuda_graph`는 예측 시 세 가지 값을 허용합니다. `False`가 기본값입니다. `True`는 각 입력 형상이 처음 나타날 때 캡처합니다. `"auto"`는 형상이 반복될 때까지 기다리므로, 일회성 및 형상 변화 작업은 재사용하지 않을 캡처에 대해 비용이 발생하지 않습니다. `capture_graph(imgsz=None, batch=1, dtype=None)`는 첫 요청에서 비용을 이전하며, `graph_info()`는 캡처된 그래프와 재생 횟수를 보고하고, `release_graphs()`는 이를 해제합니다.

학습 시 플래그는 단순한 불리언이며, CLI에서는 `--cuda-graph`입니다. 주변 제어에 대해서는 [예측 성능](/docs/predict/performance) 및 [학습 성능](/docs/train/performance)을 참조하십시오.

## 추론 지원

지원은 계열 단위로 제공되며, `SUPPORTS_CUDA_GRAPH` 클래스 변수를 통해 선언됩니다. 그리고 계열은 서로 다른 분포에서 가져온 두 개의 프로브 입력에 대해 비트가 동일하게 캡처되고 재생된 후에야 표시됩니다. 그 공유 패리티 행렬은 아홉 개 작업에서 39개 계열을 다룹니다.

| 작업 | 계열 |
|---|---|
| 감지 | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| 분할 | dfine, rtmdet, rfdetr, ec |
| 포즈 | ec, yolonas, rfdetr |
| 포인트 | fomo |
| 분류 | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| 의미 기반 | eomt, dinov2, segformer, pidnet, lingbotvision |
| 깊이 | depth_anything, depth_anything3, zipdepth |
| 복원 | nafnet, realesrgan, swinir |
| 매트 | birefnet |

여러 계열들이 여러 작업에 걸쳐 나타나므로, 매트릭스는 고유한 계열 수보다 더 많은 행을 실행합니다. 세 개의 추가 계열은 공유 매트릭스를 통해서가 아니라 자체 전용 테스트가 있는 계열별 코드 경로를 통해 캡처되며, 39개 계열에 포함되지 않습니다: PP-OCR, SAM 및 SenseNova.

검증은 근사치가 아닌 비트 단위입니다. 이전 버전의 프로토콜은 상대적 크기로 패리티를 판단하여 YOLOX, EfficientNetV2 및 YOLOv7 세 개의 건강한 계열을 잘못 하향 평가했는데, 이들 계열의 eager-to-graph 차이는 약 1e-7 정도이지만 여전히 중요한 프로브에서 비트 단위로 동일합니다.

## 학습 지원

이번 릴리스에서 학습 캡처가 두 계열에서 24계열으로 증가했으며, 다섯 가지 작업에 걸쳐 진행되었습니다.

| 작업 | 계열 |
|---|---|
| 감지 | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| 분류 | resnet, convnext, mobilenetv4, efficientnetv2 |
| 의미론적 | segformer, lingbotvision |
| 포인트 | fomo |
| 복원 | nafnet |

나머지 모든 것은 eager로 학습됩니다: 동일한 계열의 다른 작업, 목록에 없는 계열, 분산 실행 및 증류 실행. 또한 새로운 형태가 있는 동안에는 캡처가 건너뛰어지며, 학습 경로가 입력 형태가 세 번 반복될 때까지 기다린 후 캡처하기 때문에 `multi_scale=True`는 전혀 캡처하지 않을 수도 있습니다.

## 지원되지 않는 계열에 대한 두 가지 다른 답변

추론 경로가 발생합니다. 선택하지 않은 계열에서 `predict(cuda_graph=True)`가 발생하면, `NotImplementedError`가 계열의 이름을 지정하여, eager를 실행하고 실제로 얻지 못한 속도 향상을 얻었다고 믿게 만들기보다는 발생합니다. 그 이유는 잘못된 캡처가 명확하게 실패하지 않기 때문입니다: 캡처할 수 없는 작업을 수행하는 forward를 재실행하면 잘못된 수치를 조용히 반환하므로, 지원은 시도와 대체 방법 대신 계열별 명시적 주장이어야 합니다.

학습 경로 로그. `train(cuda_graph=True)`는 항상 안전하게 통과할 수 있으며, 캡처할 수 없는 계열, 작업 또는 구성은 한 줄을 작성하고 변경되지 않은 채로 eager로 학습합니다. 실행 중간에 캡처가 실패하면 나머지 실행도 중단하지 않고 eager로 실행됩니다. 비대칭성은 의도적입니다: 예측은 호출 위치에서 수정할 수 있는 호출이지만, 학습 실행은 선택적 최적화 때문에 여섯 시간에 죽어서는 안 됩니다.

## 솔기 분할

일부 계열은 한 단계가 그래프가 기록할 수 없는 작업을 수행하기 때문에 전체를 캡처할 수 없습니다. 계열을 버리는 대신, 캡처는 검증된 솔기에서 분할됩니다: 캡처 가능한 부분은 재생되고 나머지는 eager로 실행되며, 결합된 출력은 모든 것을 eager로 실행한 것과 동일합니다.

| 계열 | 캡처됨 | eager, 및 이유 |
|---|---|---|
| 깊이 아무거나 3 | 네트워크 | 포워드 후 호스트가 볼 수 있는 작업인 스카이 스텝 |
| BiRefNet | 인코더, `forward_enc` | 디코더, `deform_conv2d`이 캡처 시 다른 결과를 재생함 |
| PP-OCR | 검출 단계, `forward_det` | 인식, 줄마다 크롭 폭이 다르기 때문 |
| SAM | 이미지 인코더 | 여러 번 실행되는 프롬프트 경로 |
| SenseNova | 비전 타워 | 단계별로 증가하는 KV 캐시를 가진 오토리그레시브 생성 |
| 인코더-디코더 검출기 | 백본 및 인코더 | 디코더 및 헝가리안 기준 |

BiRefNet 분할은 두 번 읽을 가치가 있습니다: `deform_conv2d` 캡처 시 잘못 작동하는 것은 어떤 모델에서도 벗어난 순수 호출에서 재현됩니다. 이를 순수 PyTorch 등가물로 교체하는 것은 거부되었는데, 이는 eager 예측도 변경되었을 것이고, eager 수치는 계약 사항이기 때문입니다.

인코더-디코더 케이스는 D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4 및 EC를 다룹니다. 그들의 디코더는 실제 값을 기반으로 대조-잡음 제거 쿼리를 생성하며, 이러한 쿼리의 수는 배치에서 가장 큰 실제 값 개수에 따라 결정되므로 디코더의 토큰 수는 배치마다 달라집니다. 이것은 그래프가 용납할 수 없는 유일한 사항입니다. 백본과 인코더를 합하면 이 계열에서는 한 단계의 약 5분의 1에서 4분의 1 정도이며, 이것이 속도 향상 테이블 하단에 위치하는 이유입니다.

PP-OCR은 러너의 캐시 한도 내에서 감지 입력 형태마다 하나의 그래프를 캡처하며, 캡처 범위가 활성화되지 않은 경우 즉시 결과를 반환합니다.

## 수치

대부분의 계열은 비트가 동일하며, 동일하지 않은 경우에는 이유가 명명되어 있고 무작위로 처리되지 않습니다. 학습의 0단계에서는 24개 계열 모두에서 손실이 비트 동일하며, BatchNorm 버퍼도 다르지 않습니다. 범주를 구분하는 것은 그래디언트 비교입니다.

| 클래스 | 계열 | 의미 |
|---|---|---|
| 정확한 | 24개의 대부분 | 모든 그래디언트 비트 동일 |
| 1 ULP | fomo, lingbotvision | float32의 마지막 비트, 약 1e-7 상대, 다른 합산 순서에서 나옴 |
| 즉시 노이즈 | DETR 계보 | 그래프화된 것은 eager 간 두 번의 eager 실행이 서로 다른 것과 마찬가지로 차이가 없다 |
| 부동 소수점 반올림 | rtmdet | 139개의 그래디언트 중 137개는 비트 동일, 두 개는 약 3e-4 차이 |
| 자체 RNG 스트림 | segformer | 확률적 깊이는 캡처된 영역 내에 위치 |

eager-노이즈 클래스는 올바르게 읽는 것이 중요하다. 이 계열들의 경우, 두 개의 시드된 eager 실행조차도 불일치하므로, 비트 동일 여부는 그래프화된 실행이 실패했다는 기준이 아니라, 아무 것도 일치하지 않는다는 기준이다. 이는 `amp=False`에서도 더 넓게 적용되며, 여기서 fp32 가중치 그래디언트의 측정된 3.2e-7 상대 비결정성이 누적된다: 두 개의 시드된 eager YOLOv9-t 실행은 20단계 동안 36퍼센트까지 다르며, TF32를 끄는 것으로는 해결되지 않는다.

## 메모리 핀

캡처는 `capture_error_mode="thread_local"`와 함께 실행됩니다. PyTorch의 기본 `"global"` 모드에서, 다음 배치를 스테이징하는 DataLoader 핀 메모리 스레드는 `cudaHostAlloc`를 호출하는데, 이는 진행 중인 캡처를 모두 무효화하고 동시에 그로 인해 오염되므로, 다음 배치를 가져올 때 핀 메모리 스레드 내부에서 발생한 오류로 실행이 중단됩니다. 이 페어링은 진단되기 전에 실제 학습 캠페인에서 두 번 관찰되었습니다.

스레드 로컬 모드는 캡처 스레드만 제한합니다. 핀 스레드는 캡처 스트림에 절대 접근하지 않으므로, 그가 하는 일은 처음부터 그래프에 속하지 않습니다. 학습은 더 진행되며 일시적으로 `torch.cuda.CUDAGraph` 서브클래스를 대체하여 모드를 강제로 설정합니다. 이는 `make_graphed_callables`가 이를 위한 인자를 제공하지 않기 때문이며, 두 개의 동시 캡처가 대체를 남기지 않도록 잠금 상태에서 수행됩니다.

## 가치가 있는 것

RTX 5070 Ti에서 AMP 환경 하에 측정, 한 팔당 한 프로세스, 실제 배치 하나를 재생하여 데이터로더는 제외, 워밍업 후 24단계 중 가장 빠름. 탐지 640 px, 분류 224 px.

| 패밀리 | 배치 | 속도 증가 |
|---|---:|---:|
| FOMO s | 16 | 3.63배 |
| MobileNetV4 s | 16 | 2.74배 |
| EfficientNetV2 b0 | 16 | 2.44배 |
| YOLOv9-t | 8 | 1.99배 |
| YOLOv9 e2e | 8 | 1.76배 |
| YOLOv9 p2 | 8 | 1.49배 |
| 기타 모든 것 | 다양함 | 1.04배에서 1.26배 |

전체 실행은 덜 향상되는데, 그래프가 데이터로더나 검증 속도를 높일 수 없기 때문입니다. 406장의 이미지에서 20에폭 YOLOv9-t 파인튜닝을 실행한 경우, 428.4초에서 367.7초로 줄어들었고, 엔드투엔드 성능 향상은 1.16배였으며, 두 경우 모두 mAP50-95는 0.6394로 동일하고 에폭별 손실도 동일했습니다.

상한선은 네트워크가 단계(step)를 얼마나 수행하느냐에 의해 결정됩니다. 동일한 하드웨어에서 640 px 및 배치 8일 때, YOLOv9-t는 84퍼센트지만 RTMDet-t는 단계의 대부분을 레이블 할당기에서 보내므로 26퍼센트에 불과합니다. 런치 오버헤드는 Windows에서 가장 높으므로, Linux의 향상은 이 표의 약 3분의 1에서 절반 정도에 해당하며, 데이터로더에 의해 제한된 실행은 실제 시간 변경이 전혀 없습니다. 최대 메모리는 5퍼센트 낮은 것부터 19퍼센트 높은 것까지 변동합니다.

## 주의사항

그래프는 값을 기록하는 것이 아니라 주소를 기록하므로, 매개변수를 재배치하는 모든 동작은 그래프를 무효화합니다. `predict(device=...)`를 통해 장치를 변경하거나, 양자화 및 역양자화하면 캡처된 그래프가 모두 무효화됩니다.

배치 크기가 모델 계열보다 더 중요합니다: RT-DETR-r18은 배치 2에서 1.19배를, 배치 8에서 1.04배를 얻습니다. 이는 큰 배치가 계산 제약을 받으며 제거할 런치 오버헤드가 적기 때문입니다.

추론 동등성 스위트는 선택적 `kernels` 패키지를 설치하지 않고 실행되었으므로, 컴파일된 허브 커널이 활성화된 상태에서의 캡처 안전성은 다루지 않습니다. 캡처 문제를 분리하는 동안 그것들을 제거하려면 `LIBREYOLO_HUB_KERNELS=0`를 설정하십시오. [커널](/docs/reference/kernels)을 참조하십시오.
