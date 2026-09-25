---
title: "RF100-VL 벤치마크: LibreYOLO YOLO 모델의 일반화 성능은?"
description: RF100-VL 결과로 파인튜닝 후 YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter, RF-DETR이 실제 데이터셋 100개 전반에 얼마나 잘 일반화하는지 측정합니다.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL은 검출기가 COCO를 넘어 얼마나 잘 적응하는지 묻습니다. 각 모델을 적외선, X선, 현미경, 스포츠, 전파 스펙트럼 이미지, 작은 객체, 비디오 게임 등 매우 다양한 데이터셋 100개에서 각각 파인튜닝했습니다. LibreYOLO 설정 17개를 벤치마크했으며, 총 파인튜닝 횟수는 **1700회**입니다.

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## 결과

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

이번 비교에서는 RF-DETR-L이 **61.76**으로 가장 높은 점수를 기록했습니다. M이 61.13으로 뒤를 이었고, S는 60.41을 기록했습니다. RF-DETR 결과 네 개는 [Roboflow가 공개한 수치](https://rfdetr.roboflow.com/latest/learn/benchmarks/)와 비슷하며, 이는 LibreYOLO의 RF-DETR 학습을 검증하는 데 유용합니다.

모델 크기가 모든 결과를 예측하지는 않았습니다. YOLO-NAS-S와 M은 각각 58.00과 57.99로 사실상 동률입니다. EdgeCrafter-M은 57.93으로, S의 55.99와 L의 56.11보다 높습니다. 이러한 성능 역전은 조사할 계획입니다. 이 실험은 통제된 스케일링 실험이 아닙니다. 해상도, 사전 학습 가중치, 정밀도와 레시피가 서로 다릅니다.

백본 LoRA를 적용한 RF-DETR-S는 57.19를 기록했으며, 전체 파인튜닝은 60.41을 기록했습니다. 전체 파인튜닝은 데이터셋 95개에서 더 높은 성능을 보입니다. 기록된 중앙값 기준으로는 시간이 7분만 더 걸렸고, 작업 시간 총합은 거의 달라지지 않았습니다.

YOLOv9 행은 중요한 학습 수정 사항이 적용되기 전에 나온 결과입니다. 이례적인 M 결과를 조사하다가 PGI 누락, 다른 레터박스 규칙, 100개 레이블 학습 제한을 발견했습니다. 보관된 수치는 실제 실행된 코드의 결과를 나타냅니다. YOLOv9 아키텍처에 대한 판정은 아닙니다.

아래에서 모델을 선택해 데이터셋 100개 전체의 점수를 확인할 수 있습니다.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## 방법론

각 데이터셋에서 `train`으로 학습하고, 검증 AP50:95가 가장 높은 체크포인트를 선택한 뒤 `test`에서 한 번 평가했습니다. 대표 점수는 테스트 결과 100개의 비가중 평균입니다.

| 설정 | 실험 규칙 |
|---|---|
| 학습 | 100 에폭, 조기 종료 없음 |
| 유효 배치 크기 | 16 |
| 시드 | 0, 데이터셋마다 완료된 실행 1회 |
| 체크포인트 | EMA 가중치를 사용한 검증 AP50:95 최고값 |
| 테스트 점수 계산 | `maxDets=500`을 적용한 pycocotools |
| 튜닝 | 패밀리별 고정 레시피 1개, 데이터셋별 튜닝 없음 |

각 패밀리는 자체 학습 레시피를 사용했습니다.

| 패밀리 | 해상도 | 정밀도 | 증강 개요 |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | 모자이크, HSV, 플립, 마지막 15 에폭 동안 강한 증강 비활성화 |
| YOLOX Nano/Tiny | 416 | FP32 | 모자이크, mixup, HSV, affine, 플립, 마지막 15 에폭 구간 |
| YOLOX S/M | 640 | FP32 | 같은 패밀리 레시피에 크기별 학습률 적용 |
| YOLO-NAS S/M | 640 | FP32 | mixup, HSV, 플립, 모자이크 비활성화 |
| EdgeCrafter S/M/L | 640 | FP16 | 플립, LibreYOLO 패밀리 기본값 |
| RF-DETR N/S/M/L 및 S LoRA | 384/512/576/704, LoRA는 512 | BF16 | 다중 스케일, 크롭/리사이즈, 플립 |

이 결과는 단일 시드로 얻었습니다. 작은 차이가 개선을 입증하는 것은 아닙니다. RF-DETR M과 L은 사용 가능한 메모리에 맞추기 위해 그래디언트 누적을 사용했으며, 밀도가 높은 일부 데이터셋에서는 실제 배치 크기를 줄여야 했습니다. 공개 RF-DETR 레시피는 COCO 체크포인트에서 시작하지만, Roboflow는 과거 표를 만들 때 비공개 Objects365 체크포인트를 사용했습니다.

학습 시간은 통제된 속도 벤치마크가 아니라 캠페인 기록입니다. 작업이 GPU를 공유하거나 재시도 또는 재개되는 경우도 있었습니다. 선택 사항인 T4 단일 아티팩트 지연 시간 프로토콜은 실행하지 않았습니다. YOLO-NAS는 별도 라이선스가 적용되는 Deci의 [비상업용 사전 학습 가중치](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md)를 사용합니다.

## 워크로드를 통해 개선한 점

소규모 테스트로 학습이 시작되는지는 확인할 수 있습니다. 하지만 여러 아키텍처와 데이터셋에서 몇 주 동안 이어지는 파인튜닝을 재현하지는 못합니다. 이 규모에서는 느린 경로, 메모리 부담, 정확성 문제가 뚜렷하게 드러났습니다.

- **이미지 로딩과 검증.** 증강 전에 결정론적 리사이즈 결과를 캐시하고, 검증기 작업자와 버퍼를 재사용했으며, 지원되는 경우 검증 순전파를 그래프로 캡처했습니다. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **학습 CUDA 그래프.** 그래프 캡처 지원을 YOLOv9와 RF-DETR에서 패밀리 24개로 확대했습니다. DataLoader의 pin-memory 경쟁 상태와 학습 전환 시 그래프 무효화 문제도 수정했습니다. YOLOv9-T 테스트는 보고된 AP가 같으면서 428.4초에서 367.7초로 단축됐습니다. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **COCO 점수 계산.** 밀도가 높은 데이터셋에서는 평가 시간이 학습 시간보다 오래 걸리기도 했습니다. faster-coco-eval 통합으로 테스트 분할 100개에서 저장된 예측을 평가하는 시간이 131.4초에서 8.4초로 줄었습니다. 지표 값 1,400개 중 1,381개는 비트 단위로 동일했습니다. 가장 큰 차이는 2.22e-16이었으며 대표 AP는 바뀌지 않았습니다. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR 및 공유 학습 경로.** 더 빠른 L1 매칭, 장치 동기화 감소, 융합 AdamW, 매처 메모리 제한으로 기록된 테스트에서 RF-DETR-S 스텝이 266ms에서 234ms로 단축됐습니다. 같은 조사로 다른 매처 5개, YOLOv9 로깅, EdgeCrafter 텐서 재사용도 개선했습니다. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **어텐션.** 조건에 맞는 어텐션 연산을 PyTorch SDPA로 처리하고, Apache-2.0 CUDA 구현을 통합했으며, 혼합 정밀도 실행 문제를 수정했습니다. 추론을 위해 저장소 내 Triton 변형 어텐션 커널도 작성했습니다. 문서화된 테스트에서 이 커널은 단독 실행 시 약 4배 빨랐고, RF-DETR-N의 종단 간 실행은 약 7% 빨라졌습니다. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **학습 정확성.** YOLOX BatchNorm 재구성을 수정하고 YOLOv9 PGI, 레터박싱 메타데이터, 레이블 용량, 모멘텀 워밍업을 복구했습니다. 벤치마크에서 확보한 체크포인트와 실패 패턴을 통해 두 문제를 모두 발견할 수 있었습니다. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

이 수치는 서로 다른 워크로드의 개별 테스트에서 나온 것입니다. 하나의 대표 속도 향상 수치로 곱해 합산해서는 안 됩니다. 이번 캠페인 전보다 LibreYOLO가 실제 학습 워크로드에서 더 빠르고 안정적으로 작동합니다.

## 하네스와 아티팩트

공개 [학습 하네스](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness)는 데이터셋을 GPU에 분배하고, 중단된 작업을 재개하며, OOM 복구를 처리하고, 레시피, 데이터셋 버전, 로그, 체크포인트와 예측을 기록합니다.

[run-rf100vl-benchmark 스킬](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md)은 AI 코딩 에이전트에 프로토콜과 Vast.ai 운영 지침을 제공합니다. [결과 아카이브](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main)에는 공개된 모든 실행 결과가 있습니다.

## Roboflow에 감사드립니다

제가 COCO를 넘어 벤치마크하고 싶지만 실행 비용을 감당하기 어렵다고 쓴 뒤, Joseph Nelson이 GPU 지원을 제안했습니다. Matvei Popov는 참조 설정을 공유하고 평가 설정을 설명했으며, 결과가 나올 때마다 확인했습니다.

덕분에 LibreYOLO의 17개 설정에 걸친 학습을 검증하고, 모델 선택에 활용할 수 있는 근거를 공개하며, 하네스를 배포하고, 라이브러리의 약점이 드러날 때까지 부하를 줄 수 있었습니다.

컴퓨팅 자원과 시간, 안내를 제공해 준 Joseph, Matvei와 Roboflow 팀에 감사드립니다.
