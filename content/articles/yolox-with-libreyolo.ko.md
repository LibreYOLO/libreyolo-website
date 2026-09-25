---
title: "YOLOX 대안 2026: LibreYOLO로 실행하기"
description: "최신 YOLOX 릴리스는 2022년 버전입니다. LibreYOLO API로 YOLOX를 실행하고, 학습하고, 검증하고, 내보내는 방법을 알아봅니다."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "YOLOX는 유지보수가 중단되었습니까?"
    a: "공식 저장소는 보관 처리되지 않았지만, 최신 릴리스는 2022년 4월의 YOLOX 0.3.0입니다. 공식적으로 유지보수가 중단되었다기보다는 업데이트가 뜸하다고 하는 편이 정확합니다."
  - q: "LibreYOLO로 YOLOX를 학습할 수 있습니까?"
    a: "예. LibreYOLO는 YOLOX 예측, 학습, 검증 및 내보내기를 지원합니다."
---

YOLOX는 여전히 유용한 Apache-2.0 객체 탐지 모델입니다. [공식 저장소](https://github.com/Megvii-BaseDetection/YOLOX)는 보관 처리되지 않았지만, 최신 릴리스인 0.3.0은 2022년 4월 버전입니다. 유지보수 문제는 실제로 있습니다. 모델은 여전히 작동합니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO는 Nano부터 X까지 YOLOX의 여섯 가지 크기를 지원하며, 하나의 API로 예측, 학습, 검증 및 내보내기를 제공합니다. 정확히 `Exp` 설정 워크플로가 필요하다면 원본 프로젝트를 사용하십시오.

`pip install libreyolo`로 시작합니다. [YOLOX 문서](https://www.libreyolo.com/docs/models/yolox)에서 전체 API를 확인할 수 있습니다. [YOLO-NAS는 계속 유지보수되고 있습니다](/articles/yolo-nas-with-libreyolo)라는 짧은 안내도 있습니다.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [문서](https://www.libreyolo.com/docs)
