---
title: "LiteRT vs TensorFlow Lite: 이름만 바뀌었을까? 파일 형식은 그대로"
description: "Google은 2024년 9월 TensorFlow Lite의 이름을 LiteRT로 바꿨습니다. 런타임과 .tflite 파일은 그대로이며 아무것도 깨지지 않습니다. 이름을 바꾼 이유와 실제 변경 사항, LibreYOLO 모델을 내보내는 방법을 알아봅니다."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "TensorFlow Lite는 지원 중단됐습니까?"
    a: "기술은 계속 유지되며 TensorFlow Lite라는 이름만 더 이상 사용하지 않습니다. 런타임은 같은 API를 갖춘 LiteRT로 계속 제공되며, 기존 TFLite 모델과 코드도 계속 작동합니다. CompiledModel API와 NPU 가속 같은 새 개발 기능은 LiteRT라는 이름으로 제공됩니다."
  - q: "LiteRT를 사용하려면 모델을 다시 내보내야 합니까?"
    a: "아닙니다. TensorFlow Lite용으로 내보낸 .tflite 파일은 LiteRT 모델입니다. 파일은 바뀌지 않았으므로 다시 내보내거나 마이그레이션할 필요가 없습니다."
---

**LiteRT는 이름만 바뀐 TensorFlow Lite입니다. 새 런타임도 아니고 새로운 파일 형식도 아닙니다. 모델은 여전히 `.tflite` 파일이며, TensorFlow Lite에서 실행되던 모든 `.tflite` 파일은 LiteRT에서도 변경 없이 실행됩니다.**

"LiteRT vs TensorFlow Lite", "TensorFlow Lite는 지원 중단됐나", "LiteRT란 무엇인가"를 검색했다면, 위 문단이 답입니다. 이 페이지에서는 이름을 바꾼 이유와 LibreYOLO로 LiteRT용 모델을 내보내는 방법을 간략히 설명합니다.

## Google이 이름을 바꾼 이유

TensorFlow Lite는 2017년 휴대폰과 임베디드 보드에서 TensorFlow 모델을 실행하기 위한 도구로 출시됐습니다. 이후 범위가 크게 넓어졌습니다. Google은 PyTorch, JAX, Keras에서 변환하는 경로를 만들었고, 2024년에는 TensorFlow를 전혀 거치지 않은 모델도 상당수가 TensorFlow Lite에서 실행됐습니다.

그 시점에는 이름이 실제 용도를 오해하게 만들었습니다. PyTorch 사용자들은 TensorFlow 전용 도구처럼 들려서 사용을 피했습니다. 그래서 Google은 2024년 9월 "Lite Runtime"의 약자인 [LiteRT로 이름을 바꿨습니다](https://developers.googleblog.com/tensorflow-lite-is-now-litert/). 이것이 전부입니다. 팀과 코드는 그대로이며, 프레임워크에 종속되지 않는 이름으로 바뀌었습니다.

## 실제로 바뀐 점

바뀐 부분은 거의 없으며, 기존 기능을 깨뜨리는 변경은 없습니다.

* 코드 저장소가 새 위치로 이동했습니다: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* 문서가 [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)로 이동했습니다.
* 모델 실행용 Python 패키지는 이제 `ai-edge-litert`입니다. 이전 `tflite-runtime` 패키지는 오래된 상태이며, 새 패키지에도 같은 `Interpreter` 클래스가 있습니다.
* 이름 변경 이후의 새 기능은 LiteRT라는 이름으로 제공됩니다. 더 간단한 `CompiledModel` API와 GPU/NPU 가속이 포함되며, Google은 이를 [2026년 1월 프로덕션 준비 완료로 선언했습니다](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

기존 Interpreter API도 종전과 같이 계속 작동합니다. 실제로 새로 추가된 확장자는 `.litertlm` 하나지만, 온디바이스 LLM 전용 패키징 형식입니다. 컴퓨터 비전 모델에는 해당 형식이 없으므로 신경 쓸 필요가 없습니다. 따라서 어떤 문서에서 "TFLite"라고 하고 다른 문서에서 "LiteRT"라고 해도 같은 파일을 가리킵니다.

## LibreYOLO 모델을 LiteRT로 내보내기

LibreYOLO는 v1.3.0에서 TFLite/LiteRT 내보내기 경로를 추가했습니다. Python 3.12 이상과 추가 패키지 하나가 필요합니다.

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # 처음 실행할 때 자동으로 내려받습니다
model.export(format="tflite")        # .tflite 파일을 생성합니다
```

또는 CLI에서 실행합니다.

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

현재 검증된 경로는 YOLO9 탐지와 RF-DETR의 탐지, 분할, 자세 추정이며, 모두 아직 실험 단계로 표시되어 있습니다. 전체 지원표는 [문서](/docs/v1.3.0)에서 확인할 수 있습니다.

내보낸 파일을 실행하려면 대상 장치에 Google의 `ai-edge-litert` 패키지를 사용합니다.

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, 전처리된 이미지를 여기에 입력합니다
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

입력 형식은 NHWC(채널 마지막)입니다. ONNX에서 변환할 때 레이아웃이 전치되며, 이는 이 형식에서 정상적인 동작입니다.

## LibreYOLO에서 변경한 점

변경할 부분이 많지 않았습니다. 실제로 바꿔야 할 내용이 거의 없었기 때문입니다. 이제 문서에서 형식을 "TFLite (LiteRT)"로 표시해 두 이름을 모두 검색할 수 있으며, API의 형식 이름은 계속 `format="tflite"`입니다.

## FAQ

**TensorFlow Lite는 지원 중단됐습니까?**
기술은 계속 유지되며 TensorFlow Lite라는 이름만 더 이상 사용하지 않습니다. 런타임은 같은 API를 갖춘 LiteRT로 계속 제공되며, 기존 TFLite 모델과 코드도 계속 작동합니다. CompiledModel API와 NPU 가속 같은 새 개발 기능은 LiteRT라는 이름으로 제공됩니다.

**LiteRT를 사용하려면 모델을 다시 내보내야 합니까?**
아닙니다. TensorFlow Lite용으로 내보낸 `.tflite` 파일은 LiteRT 모델입니다. 파일은 바뀌지 않았으므로 다시 내보내거나 마이그레이션할 필요가 없습니다.
