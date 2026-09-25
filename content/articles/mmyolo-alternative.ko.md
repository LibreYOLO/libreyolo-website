---
title: "2026년 MMYOLO 대안: LibreYOLO로 RTMDet 실행하기"
description: "MMYOLO의 최신 릴리스는 2023년 버전입니다. LibreYOLO의 직접 API로 RTMDet 같은 지원되는 객체 탐지 모델을 실행할 수 있습니다."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "MMYOLO는 유지보수가 중단되었습니까?"
    a: "MMYOLO가 공식적으로 보관 처리된 것은 아니지만, 최신 GitHub 릴리스는 2023년 8월의 버전 0.6.0입니다."
  - q: "LibreYOLO는 모든 MMYOLO 체크포인트를 불러올 수 있습니까?"
    a: "아닙니다. LibreYOLO는 명시된 모델 계열을 지원하며, 임의의 MMYOLO 체크포인트를 바로 불러올 수 있다고 보장하지 않습니다."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo)는 보관 처리되지 않았지만, 최신 릴리스인 버전 0.6.0은 2023년 8월에 공개되었습니다. RTMDet 같은 지원되는 객체 탐지 모델만 필요하다면 전체 OpenMMLab 스택이 필요하지 않을 수 있습니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO는 MMYOLO를 완전히 대체하지 않습니다. 모든 config, recipe 또는 체크포인트를 재현하지는 않습니다. 대신 RTMDet 및 YOLOX 같은 지원 모델 계열에 예측, 학습, 검증, 내보내기를 위한 직접 API를 제공합니다.

`pip install libreyolo`부터 시작합니다. [RTMDet 문서](https://www.libreyolo.com/docs/models/rtmdet)에서 지원되는 작업을 확인할 수 있습니다. [간단한 RTMDet 가이드](/articles/rtmdet-without-mmdetection)에서 마이그레이션 방법을 다룹니다.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [문서](https://www.libreyolo.com/docs)
