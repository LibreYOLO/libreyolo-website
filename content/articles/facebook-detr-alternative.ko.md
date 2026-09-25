---
title: "Facebook DETR 대안 2026: LibreYOLO로 실행하기"
description: "기존 Facebook Research DETR 저장소는 보관 처리되었습니다. LibreYOLO를 통해 공식 DETR 변형 4가지를 실행할 수 있습니다."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Facebook DETR은 중단되었습니까?"
    a: "기존 facebookresearch/detr 저장소는 2024년 3월 12일에 보관 처리되었습니다."
  - q: "LibreYOLO로 기존 DETR을 학습할 수 있습니까?"
    a: "아니요. LibreYOLO는 DETR의 예측, 검증, 내보내기를 지원하지만 500 에폭 학습 레시피는 지원하지 않습니다."
---

기존 [Facebook Research DETR 저장소](https://github.com/facebookresearch/detr)는 2024년 3월 12일에 보관 처리되었습니다. LibreYOLO는 다른 모델 계열과 동일한 API로 기존 DETR을 계속 사용할 수 있게 합니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

ResNet-50 및 ResNet-101 변형 4가지를 예측, 검증, ONNX 및 TorchScript 내보내기에 사용할 수 있습니다. 학습은 구현되어 있지 않습니다. 또한 LibreYOLO는 원래의 종횡비 유지 평가 파이프라인을 재현하지 않고, 800 x 800 고정 입력을 사용합니다.

`pip install libreyolo`를 실행한 뒤 사용해 보십시오. [DETR 문서](https://www.libreyolo.com/docs/models/detr)에서 체크포인트 4가지를 모두 확인할 수 있습니다. CNN과 비교하려면 [EfficientDet 대안](/articles/efficientdet-alternative)을 참고하십시오.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [문서](https://www.libreyolo.com/docs)
