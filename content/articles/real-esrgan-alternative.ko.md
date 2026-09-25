---
title: "Real-ESRGAN 대안 2026: LibreYOLO로 이미지 업스케일링"
description: "LibreYOLO에서 Real-ESRGAN 2x 및 4x 이미지 업스케일링을 실행하고, 대형 이미지에 타일 추론을 적용합니다."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Real-ESRGAN은 중단되었습니까?"
    a: "저장소는 보관 처리되지 않았지만, 최신 GitHub 릴리스는 2022년 9월의 버전 0.3.0입니다."
  - q: "LibreYOLO로 Real-ESRGAN을 학습할 수 있습니까?"
    a: "아닙니다. LibreYOLO는 Real-ESRGAN 예측, 검증 및 내보내기를 지원하며 학습은 지원하지 않습니다."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)은 공식적으로 보관 처리되지는 않았지만, 최신 GitHub 릴리스인 버전 0.3.0은 2022년 9월에 나왔습니다. LibreYOLO는 일반 예측 API를 통해 2x 및 4x 체크포인트를 실행합니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO는 타일 예측, PSNR/SSIM 검증 및 내보내기를 지원하며, 2x, 4x 및 fast 4x 체크포인트를 제공합니다. 학습은 구현되어 있지 않습니다. 업스트림의 노이즈 제거 강도 블렌딩 기능도 이 포트에는 포함되지 않습니다.

`pip install libreyolo`로 사용해 볼 수 있습니다. 전체 복원 옵션은 [Real-ESRGAN 문서](https://www.libreyolo.com/docs/models/real-esrgan)를 참고하십시오. 깊이 추정에 대해서는 [MiDaS 대안](/articles/midas-alternative)을 참고하십시오.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [문서](https://www.libreyolo.com/docs)
