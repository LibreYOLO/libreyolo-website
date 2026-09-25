---
title: "MiDaS 대안 2026: LibreYOLO로 깊이 추정"
description: "공식 MiDaS 저장소는 보관 처리되었습니다. LibreYOLO를 통해 Small 및 DPT-Large 깊이 모델을 실행합니다."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "MiDaS는 유지보수가 중단되었습니까?"
    a: "공식 isl-org/MiDaS 저장소는 2025년 8월 25일에 보관 처리되었습니다."
  - q: "MiDaS는 미터 단위 깊이를 반환합니까?"
    a: "아닙니다. MiDaS는 상대 역깊이를 반환하며, 미터 단위나 이미지 간 고정된 스케일은 없습니다."
---

[공식 MiDaS 저장소](https://github.com/isl-org/MiDaS)는 2025년 8월 25일에 보관 처리되었습니다. LibreYOLO를 사용하면 Small 및 DPT-Large 체크포인트를 유지보수되는 방식으로 실행할 수 있습니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

예측, 제로샷 검증, 고정 해상도 내보내기를 지원합니다. 학습은 지원하지 않습니다. 출력은 상대 역깊이입니다. 값이 높을수록 더 가까우며, 미터 단위나 이미지 간 고정된 스케일은 없습니다.

기본 설치만으로 충분합니다: `pip install libreyolo`. [MiDaS 문서](https://www.libreyolo.com/docs/models/midas) 또는 [Real-ESRGAN 대안](/articles/real-esrgan-alternative)을 이어서 살펴보십시오.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [문서](https://www.libreyolo.com/docs)
