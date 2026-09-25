---
title: "YOLO-NAS는 아직 유지보수 중입니다"
description: "LibreYOLO에서 YOLO-NAS를 실행하고 학습, 검증, 내보내기할 수 있습니다. SuperGradients는 2024년 4월 이후 새 릴리스를 내지 않았습니다. Deci 라이선스 조건이 적용되는 체크포인트를 사용하지 않도록 처음부터 새 가중치를 학습할 계획입니다."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "YOLO-NAS는 유지보수가 중단되었습니까?"
    a: "SuperGradients는 2024년 4월에 3.7.1을 마지막으로 릴리스했습니다. LibreYOLO는 최신 PyTorch에서 YOLO-NAS 예측, 학습, 검증, 내보내기를 지원합니다."
  - q: "Ultralytics는 YOLO-NAS를 유지보수합니까?"
    a: "Ultralytics는 추론, 검증, 내보내기를 지원합니다. 학습은 지원하지 않습니다."
  - q: "사전 학습된 YOLO-NAS 가중치를 상업적으로 사용할 수 있습니까?"
    a: "Deci의 사전 학습된 가중치에는 어떤 라이브러리로 불러오든 업스트림의 비상업적 사용 조건이 계속 적용됩니다. 무작위 초기화부터 학습하면 해당 체크포인트를 사용하지 않아도 됩니다. 처음부터 학습한 COCO 가중치도 공개할 계획입니다."
---

YOLO-NAS는 여전히 쓸 만한 객체 탐지 모델입니다. 원래 개발 환경인 [SuperGradients](https://github.com/Deci-AI/super-gradients)는 2024년 4월에 3.7.1을 마지막으로 릴리스했습니다. LibreYOLO는 YOLO-NAS의 예측, 학습, 검증, 내보내기를 계속 지원합니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO는 탐지용 S, M, L 크기와 자세 추정을 지원합니다. 체크포인트는 `LibreYOLONASs.pt` 같은 이름으로 Deci의 공개 CDN에서 다운로드됩니다. LibreYOLO는 이 체크포인트를 호스팅하지 않으며, Deci의 비상업적 사용 조건은 계속 적용됩니다. SuperGradients는 PyTorch를 1.9부터 1.13 사이 버전으로 고정하지만, LibreYOLO에는 2.4 이상이 필요합니다. 탐지 모델의 내보내기는 ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch, Core AI를 지원합니다. CoreML은 지원하지 않습니다.

Deci 체크포인트 없이 학습하려면 다음과 같이 합니다.

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

YOLO-NAS를 COCO 데이터셋으로 처음부터 학습하고 해당 가중치를 공개할 계획도 있습니다.

`pip install libreyolo` 명령으로 설치합니다. [YOLO-NAS 문서](https://www.libreyolo.com/docs/models/yolo-nas)에서 모델 세부 정보를 확인할 수 있습니다. 더 자세한 내용은 [SuperGradients 대안](/articles/supergradients-alternative) 문서를 참고하십시오.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [문서](https://www.libreyolo.com/docs)
