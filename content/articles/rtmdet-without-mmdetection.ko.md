---
title: "mmdetection 없이 RTMDet 실행하기: 설치 없이 추론하는 방법"
description: "RTMDet은 빠르면서 정확도도 높은 객체 탐지 모델입니다. 공식 설치는 최근 2년간 출시된 PyTorch에서 작동하지 않습니다. 대안을 소개합니다."
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "최근 PyTorch에서 mmcv 설치가 실패하는 이유는 무엇입니까?"
    a: "최신 mmcv 휠인 2.2.0은 2024년 4월에 출시된 이후 업데이트되지 않았으며, torch 2.4 / CUDA 12.1까지의 사전 빌드 바이너리만 제공합니다. 그보다 최신 PyTorch에서는 mmcv의 C++/CUDA 연산을 소스에서 컴파일해야 합니다. 이 작업은 10분에서 30분이 걸리고, 일치하는 CUDA 툴킷과 nvcc, 호환되는 C++ 컴파일러가 필요합니다. mmcv._ext 모듈이 없다는 오류가 발생하는 이유도 여기에 있습니다."
  - q: "mmdetection을 설치하지 않고 RTMDet을 실행할 수 있습니까?"
    a: "가능합니다. LibreYOLO는 업스트림 OpenMMLab 체크포인트에서 변환한 RTMDet 가중치를 불러오며, 같은 체크포인트를 기준으로 하면 추론 결과는 mmdetection과 비트 단위로 동일합니다. 이름으로 LibreRTMDets.pt를 불러오면 자동으로 다운로드됩니다. Tiny부터 X까지 다섯 가지 크기를 모두 640 px에서 실행할 수 있습니다."
  - q: "RTMDet은 상업적 사용이 무료입니까?"
    a: "예. MMDetection과 RTMDet은 Apache 2.0 라이선스이며, 변환된 가중치에도 같은 Apache 2.0 조건이 적용됩니다. LibreYOLO 코드는 MIT 라이선스입니다. 상업적 사용 제한은 없습니다."
  - q: "mmdetection을 계속 사용하는 편이 나은 경우는 언제입니까?"
    a: "전체 MMDetection 데이터 증강 파이프라인으로 RTMDet을 학습하거나 파인튜닝해야 하는 경우, 또는 mmcv가 이미 작동하는 OpenMMLab 생태계를 깊이 사용 중인 경우입니다. 추론과 배포에는 LibreYOLO가 더 나은 방법입니다."
---

RTMDet은 OpenMMLab의 실시간 객체 탐지 모델로, COCO에서 높은 정확도를 내면서도 배포에 충분히 빠릅니다. 아키텍처는 깔끔합니다. 설치 과정은 그렇지 않습니다.

공식 소스에서 RTMDet을 실행하려면 OpenMMLab의 4개 계층 스택을 구성해야 합니다.

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

버전 범위가 좁습니다. mmdet 3.3.0은 `mmcv < 2.2.0`을 요구하지만, `mim install mmcv>=2.0.0`은 문제없이 2.2.0을 설치하고, 그러면 런타임 어설션이 실패합니다. 버전을 직접 고정해야 합니다.

더 큰 문제도 있습니다. 최신 mmcv 휠은 2024년 4월에 출시된 2.2.0이며 그 이후 한 번도 업데이트되지 않았습니다. 사전 빌드 바이너리는 **torch 2.4 / CUDA 12.1**까지만 제공됩니다. 오늘 `pip install torch`를 새로 실행하면 PyTorch 2.12가 설치됩니다. 이 차이 때문에 mmcv의 C++/CUDA 연산을 소스에서 컴파일해야 합니다. 10분에서 30분이 걸리고, 일치하는 CUDA 툴킷과 `nvcc`, 호환되는 C++ 컴파일러가 필요합니다. 문서에 보고된 오류도 이 과정에서 발생합니다.

* `ModuleNotFoundError: No module named 'mmcv._ext'` -- 컴파일된 연산이 빌드되지 않았거나 현재 환경과 맞지 않습니다.

* 모든 RTX 30 시리즈 카드에서 `nvcc fatal: Unsupported gpu architecture 'compute_86'`가 발생합니다.

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` -- 스택 때문에 Python 3.8로 되돌려야 합니다.

* GCC 버전이 맞지 않아 가져오는 중 세그멘테이션 오류가 발생합니다.

OpenMMLab의 자체 FAQ는 버전 충돌을 피하려면 mim 대신 pip으로 mmcv를 설치하라고 안내합니다. Get Started 페이지에서는 mim을 사용하라고 합니다. 두 문서 모두 공식 문서지만 서로 모순됩니다.

스택을 실행하고 나서도 추론을 하려면 학습 레시피를 이름에 담은 설정 파일을 선택하고, 해시가 붙은 파일 이름의 체크포인트를 별도로 다운로드한 다음, 먼저 익혀야 하는 레지스트리에 연결해야 합니다.

LibreYOLO는 이 과정을 모두 대체합니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

`mim`도, 패키지 네 개의 버전 조합도, 소스 컴파일도, 설정 파일도, 체크포인트 해시 검색도, Python 3.8 버전 고정도 필요하지 않습니다. 가중치는 업스트림 OpenMMLab 체크포인트에서 변환되며, 같은 체크포인트를 기준으로 하면 추론 결과는 mmdetection과 비트 단위로 동일합니다.

Tiny, Small, Medium, Large, X의 다섯 가지 크기를 사용할 수 있습니다. 모두 640 px에서 실행됩니다.

```python
print(results[0].boxes.xyxy)  # xyxy coordinates
print(results[0].boxes.conf)  # confidence scores
```

## 기존 구현이 여전히 적합한 경우

전체 MMDetection 데이터 증강 파이프라인으로 RTMDet을 학습하거나 파인튜닝해야 하는 경우, 또는 mmcv가 작동하는 OpenMMLab 생태계를 이미 깊이 사용 중인 경우에는 해당 환경을 계속 사용하면 됩니다. 추론과 배포에는 LibreYOLO가 더 나은 방법입니다.

## 라이선스 안내

MMDetection과 RTMDet은 Apache 2.0 라이선스입니다. LibreYOLO 코드는 MIT 라이선스입니다. 가중치에는 업스트림의 동일한 Apache 2.0 조건이 적용됩니다. 상업적 사용 제한은 없습니다.

## 사용해 보기

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO는 MIT 라이선스이며 Linux, Mac, Windows에서 실행됩니다. 코드 변경 없이 GPU, Apple Silicon, 일반 CPU에서도 작동합니다. 하나의 API로 RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, 분할, 자세 추정, 깊이 추정 등을 사용할 수 있습니다.

GitHub에서 Star를 눌러 주십시오: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 문서: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
