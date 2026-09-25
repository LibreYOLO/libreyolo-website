---
title: "LibreYOLO로 Depth Anything V2 실행하기: 두 줄이면 충분합니다"
description: "Depth Anything V2는 최첨단 단안 깊이 추정 결과를 제공합니다. LibreYOLO를 사용해 두 줄의 코드로 실행하는 방법을 알아봅니다."
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "Depth Anything V2를 가장 간단하게 실행하는 방법은 무엇입니까?"
    a: "LibreYOLO에서는 두 줄이면 됩니다. LibreDepthAnythingV2l-depth.pt를 이름으로 불러오고 save=True와 함께 predict를 호출합니다. 가중치는 처음 사용할 때 자동으로 내려받으며, 정규화, 컬러맵, 장치 배치를 CUDA, Apple Silicon, 일반 CPU에서 모두 자동으로 처리합니다."
  - q: "Depth Anything V2를 상업적으로 사용할 수 있습니까?"
    a: "Small 인코더만 Apache 2.0입니다. Base, Large, Giant는 CC-BY-NC-4.0이므로 가장 강력한 체크포인트는 비상업적 용도로만 사용할 수 있습니다. 가중치를 불러오는 라이브러리와 관계없이 업스트림 라이선스가 적용됩니다."
  - q: "LibreYOLO는 Depth Anything V2 학습을 지원합니까?"
    a: "아니요. 학습은 원본 저장소에서 진행하며, LibreYOLO는 깊이 추정 작업의 추론, 비디오, 검증을 지원합니다."
---

![구겐하임 빌바오와 그 깊이 맵](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2는 현재 사용할 수 있는 단안 깊이 맵 가운데 가장 뛰어난 결과를 제공합니다.

YOLO를 사용해 왔다면 공식 저장소를 시작하기가 그리 쉽지 않다는 점을 알 수 있습니다. 깊이 맵 하나를 얻으려 해도 여러 단계를 직접 거쳐야 합니다.

* 올바른 체크포인트를 직접 내려받아 지정된 폴더에 넣습니다.

* 인코더를 설정에 맞춥니다 (`encoder="vitl"`, `features=256`, `out_channels=...`).

* 정규화와 색상화 단계를 직접 작성합니다.

* CUDA뿐 아니라 Mac이나 CPU에서도 실행되도록 장치를 배치합니다.

* 나머지 기술 스택과 전혀 다른 추론 API를 익힙니다.

어려운 작업은 아닙니다. 그저 번거로운 절차일 뿐이며, 모두 건너뛸 수 있습니다.

LibreYOLO는 동일한 Depth Anything V2 가중치를 불러오고, `predict()` 한 번으로 깊이 추정 작업을 제공합니다. 모델 이름을 지정하면 처음 사용할 때 자동으로 내려받습니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # auto-downloads on first run
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

이것이 전부입니다. 표준 YOLO API를 사용해 봤다면 익숙한 형태입니다. 모델 이름으로 모델을 불러오고, `predict`를 호출하고, `save=True`로 시각화 결과를 저장합니다. 객체 탐지에서 사용하는 것과 정확히 같은 호출이며, 결과에 바운딩 박스 대신 깊이 맵이 담깁니다. 컬러맵, 정규화, 장치 배치는 자동으로 처리됩니다. CUDA를 사용하는 Linux, Apple Silicon이 탑재된 Mac, 일반 CPU에서 똑같이 실행됩니다. 코드를 바꿀 필요가 없습니다.

같은 호출로 더 많은 작업도 할 수 있습니다. 원시 깊이 값을 가져와 직접 활용할 수 있습니다. Depth Anything V2 자체의 학습은 원본 저장소에서 진행하며, LibreYOLO는 추론, 비디오, 검증을 지원합니다.

서로 매우 다른 장면에서도 정확히 같은 한 줄의 호출을 사용합니다.

![파쿠르 샘플 이미지와 깊이 맵: 앞쪽 점프하는 사람들이 뒤쪽 콘크리트 벽과 구분되어 두드러집니다.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![도노스티아 라 콘차 만의 항공 사진과 깊이 맵: 보트와 해안선은 가까이 보이고 탁 트인 바다는 멀어집니다.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![후원타스 데 게르니카의 기둥이 늘어선 안뜰과 깊이 맵: 건축물이 멀어지는 모습이 드러납니다.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![도노스티아 헌법 광장의 야간 축제 인파와 깊이 맵: 앞쪽 사람들의 줄이 뒤쪽 조명된 건물과 구분되어 두드러집니다.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

가중치에 관한 참고 사항입니다. LibreYOLO는 변환된 Depth Anything V2 체크포인트를 호스팅하고 처음 사용할 때 내려받으므로 직접 다운로드할 필요가 없습니다. 업스트림 라이선스는 계속 적용됩니다. Small 인코더는 Apache-2.0이고 Base, Large, Giant는 CC-BY-NC-4.0(비상업적)이므로 강력한 체크포인트는 비상업적 용도로만 사용할 수 있습니다. 오프라인으로 작업하거나 직접 변환하려면 일회성 변환 스크립트를 사용할 수 있습니다.

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## 사용해 보기

```bash
pip install libreyolo
```

LibreYOLO는 pip로 설치할 수 있는 가장 완성도 높은 컴퓨터 비전 라이브러리입니다. 하나의 익숙한 API에서 객체 탐지(RF-DETR, D-FINE, DEIM), 분할, 자세 추정, 회전 바운딩 박스, 그리고 이제 Depth Anything V2를 사용한 단안 깊이 추정까지 점점 늘어나는 최첨단 모델을 제공합니다. 해당 작업에 학습과 검증을 지원하는 모델도 포함됩니다. 모든 주요 OS에서 GPU 또는 CPU로 실행되며, MIT 라이선스를 따르므로 아무런 조건 없이 상업적으로 배포할 수 있습니다.

GitHub에서 Star를 눌러 주세요: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 문서: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
