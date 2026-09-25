---
title: "SuperGradients 대안: YOLO-NAS 추론, 학습, 내보내기를 지원하는 LibreYOLO"
description: "NVIDIA가 2024년 Deci를 인수한 뒤 SuperGradients는 새 버전을 내놓지 않았습니다. LibreYOLO는 동일한 YOLO-NAS 가중치를 불러오는 유지보수 중인 대안으로, MIT 라이선스 API 하나로 예측, 학습, 내보내기를 지원합니다."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "SuperGradients는 아직 유지보수되고 있습니까?"
    a: "아닙니다. 마지막 릴리스는 2024년 4월의 3.7.1이며, NVIDIA가 Deci를 인수하기 직전이었습니다. 그 뒤 저장소에는 릴리스가 없고, 이슈에는 답변이 달리지 않으며, 문서 사이트도 오프라인이 되었습니다. 공식적으로 보관 처리된 것은 아니지만, 사실상 관리자가 없습니다."
  - q: "SuperGradients의 가장 좋은 대안은 무엇입니까?"
    a: "YOLO-NAS를 실행하려면 LibreYOLO가 동일한 Deci 가중치를 유지보수 중인 MIT 라이선스 API로 불러오며, 학습, 검증, 내보내기도 지원합니다. 다른 아키텍처에 SuperGradients의 학습 레시피를 사용하고 있다면, 기존 저장소도 의존성을 고정하면 계속 실행됩니다."
  - q: "YOLO-NAS를 상업적으로 사용할 수 있습니까?"
    a: "Deci의 사전 학습된 가중치는 어떤 라이브러리로 불러오더라도 비상업적 용도로만 사용할 수 있습니다. 아키텍처 자체는 허용적 라이선스로 제공되므로, 자체 데이터로 YOLO-NAS를 처음부터 학습하면 이력이 Deci 체크포인트와 무관한 모델을 얻을 수 있습니다."
  - q: "LibreYOLO가 SuperGradients의 모든 기능을 대체합니까?"
    a: "모든 기능을 대체하지는 않습니다. LibreYOLO는 YOLO-NAS용 CoreML 내보내기를 지원하지 않고, 이 모델 계열의 TensorRT 경로는 검증되지 않았으며, SuperGradients의 양자화 인식 학습 레시피와 직접 대응하는 기능도 없습니다. 이런 기능이 필요하다면 기존 저장소를 계속 사용하십시오."
---

SuperGradients는 Deci의 오픈 소스 학습 라이브러리이자, 역대 가장 정확한 실시간 탐지 모델 중 하나인 YOLO-NAS의 기반이었습니다. NVIDIA가 2024년 5월 Deci를 인수한 뒤 SuperGradients는 조용해졌습니다. 마지막 릴리스 3.7.1은 2024년 4월 8일에 나왔습니다. supergradients.com의 문서도 오프라인이 되었습니다. 약 120개의 이슈가 열린 채 남아 있으며, 그중 ["이 프로젝트는 끝난 건가요?"](https://github.com/Deci-AI/super-gradients/issues/2062)라는 제목의 이슈에도 관리자의 답변이 없습니다. 그 침묵 자체가 답입니다.

저장소가 보관 처리되지는 않아 얼핏 보면 살아 있는 것처럼 보입니다. 하지만 설치를 시도하는 순간 방치된 현실을 체감하게 됩니다. `super-gradients`는 현재 PyTorch 스택과 충돌하는 `torchmetrics==0.8`을 고정하고, 그 과정에서 hydra, omegaconf, boto3, tensorboard도 함께 설치합니다. 달이 지날수록 이 버전 고정은 나머지 환경과 더 심하게 충돌하지만, 이를 해결할 수정은 나오지 않을 것입니다.

그렇다고 YOLO-NAS의 성능이 나빠진 것은 아닙니다. 대형 변형은 여전히 실시간 속도로 COCO에서 52.2 mAP를 기록합니다. 모델은 괜찮지만, 모델이 있던 집은 불타 사라졌습니다.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS accuracy vs parameters - visionanalysis.org">
</iframe>

## LibreYOLO에서 동일한 가중치 실행

LibreYOLO는 Deci CDN에서 YOLO-NAS 체크포인트를 직접 불러오며, 다른 모든 모델 계열에 사용하는 것과 동일한 API를 제공합니다. 작성해야 할 hydra 설정 파일도 없고, 다시 풀어야 할 예측 래퍼도 없습니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

S, M, L 탐지 변형을 모두 사용할 수 있고, 자세 추정도 지원합니다. `LibreYOLONASs-pose.pt`로 바꾸면 COCO 키포인트를 얻습니다. 모든 계열이 동일한 `Results` 객체를 반환하므로, 자체 데이터에서 YOLO-NAS와 RF-DETR 또는 D-FINE을 비교할 때 별도의 코드베이스를 만드는 대신 한 줄만 바꾸면 됩니다.

## 추론뿐 아니라 학습과 내보내기

SuperGradients는 본래 학습 라이브러리였으므로, 실질적인 대안이라면 학습도 지원해야 합니다. LibreYOLO는 다른 곳에서 쓰는 것과 같은 호출로 데이터셋에 맞춰 YOLO-NAS를 파인튜닝합니다.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

무작위 초기화된 모델에서 학습을 시작할 수도 있으며, 이는 라이선스 측면에서 중요합니다(아래에서 자세히 설명합니다). 어떤 형식의 데이터셋이든 검증을 통해 mAP 지표를 확인할 수 있고, ONNX, TorchScript, OpenVINO, NCNN, TFLite로 내보낼 수 있습니다. 이는 YOLO-NAS용으로 원래 라이브러리가 제공했던 것보다 폭넓은 내보내기 지원입니다. 자세한 내용은 [YOLO-NAS 문서 페이지](https://www.libreyolo.com/docs/models/yolo-nas)를 참고하고, [YOLO-NAS가 아직 유지보수되고 있다는 짧은 안내](/articles/yolo-nas-with-libreyolo)도 확인할 수 있습니다.

## 기존 저장소가 여전히 적합한 경우

솔직히 말씀드리면, SuperGradients를 설치된 상태로 유지해야 하는 경우는 세 가지입니다.

**CoreML.** LibreYOLO는 YOLO-NAS를 CoreML로 내보내지 않습니다. Apple 기기용으로 배포하고 `.mlpackage`가 필요하다면, SuperGradients에는 여전히 작동하는 경로가 있습니다.

**TensorRT.** SuperGradients는 배치 크기 관련 특이 사항까지 포함해 YOLO-NAS용 TensorRT 절차를 문서화하고 테스트했습니다. 이 모델 계열에 대한 LibreYOLO의 TensorRT 지원은 검증되지 않았습니다.

**양자화 인식 학습 레시피.** YOLO-NAS는 INT8을 고려해 설계되었고, SuperGradients는 이 성능을 끌어내는 QAT 레시피를 제공했습니다. LibreYOLO는 INT8 및 FP16 양자화를 적용해 내보낼 수 있지만, 이에 해당하는 학습 레시피는 없습니다.

기존 저장소도 2024년 당시의 환경으로 의존성을 고정하면 계속 실행됩니다. 다만 아무도 유지보수하지 않는 기반 위에서 새로 개발할 이유는 없습니다.

## 가중치에 관한 안내

사전 학습된 YOLO-NAS 가중치는 Deci가 비상업적 라이선스로 배포하며, 어떤 라이브러리로 불러오든 그 라이선스는 가중치에 적용됩니다. LibreYOLO는 가중치를 호스팅하거나 미러링하지 않습니다. 다운로드는 Deci의 공개 CDN에서 이뤄지며, 시작 전에 Deci의 약관이 출력됩니다. 연구와 비상업적 작업에는 어느 쪽을 사용해도 괜찮습니다.

상업용 YOLO-NAS가 필요하다면 이제 명확한 방법이 있습니다. 아키텍처 자체는 허용적 라이선스로 제공되므로, 자체 데이터로 처음부터 학습하면 Deci 체크포인트에서 유래하지 않은 모델이 만들어집니다. LibreYOLO는 `LibreYOLONAS(None, size="s")`로 바로 이 방식을 지원합니다.

## 사용해 보기

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO는 MIT 라이선스로 제공되고 Linux, Mac, Windows에서 실행되며, 코드 변경 없이 GPU, Apple Silicon, 일반 CPU에서 동작합니다. 하나의 API로 YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet 등을 사용할 수 있으며, 탐지, 분할, 자세 추정, 분류, 깊이 추정, 추적 등을 지원합니다.

GitHub에서 Star를 눌러 주세요: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 문서: [libreyolo.com/docs](https://www.libreyolo.com/docs)
