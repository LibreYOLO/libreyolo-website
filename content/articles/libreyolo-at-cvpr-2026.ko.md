---
title: "CVPR 2026 엣지 AI 데모: Hailo-8L과 Snapdragon에서 실행한 LibreYOLO"
description: "덴버에서 열린 CVPR 2026에서 Jabra와 코펜하겐 IT 대학교 팀은 'Edge AI in Action' 튜토리얼의 예제 모델로 LibreYOLOXs를 사용해 Hailo-8L과 Snapdragon에서 실행했습니다."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "엣지 하드웨어에서 LibreYOLOXs는 얼마나 빠릅니까?"
    a: "CVPR 2026 튜토리얼에서 LibreYOLOXs는 Hailo-8L을 탑재한 Raspberry Pi 5에서 프레임당 19.0 ms(52.6 FPS), INT8 양자화 후 Qualcomm QCS6490의 HTP/DSP에서 6.69 ms로 실행됐습니다."
  - q: "LibreYOLO 모델을 엣지 가속기에 배포하려면 어떻게 해야 합니까?"
    a: "LibreYOLO의 내보내기 호출로 ONNX로 내보낸 다음 대상 하드웨어용으로 컴파일합니다. Hailo Dataflow Compiler는 Hailo 칩용 HEF를 생성하고, SNPE / QAIRT / AI Hub 스택은 Qualcomm Snapdragon을 지원합니다. 이것이 CVPR 튜토리얼에서 처음부터 끝까지 소개한 정확한 파이프라인입니다."
---

![CVPR 2026, 덴버, 콜로라도, 6월 3일부터 7일까지](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO가 CVPR 2026에 소개됐습니다.

CVPR은 세계에서 가장 권위 있는 컴퓨터 비전 학회로 널리 인정받고 있습니다. 올해는 덴버에서 열렸으며, Jabra와 코펜하겐 IT 대학교 팀이 "Edge AI in Action: Mastering On-Device Inference"라는 튜토리얼을 진행했습니다. 이 팀은 엣지 칩에서 객체 탐지를 실행하는 예제 모델로 LibreYOLOXs를 선택했습니다.

## 구현 내용

튜토리얼에서는 엣지 파이프라인 전체를 처음부터 끝까지 다뤘습니다. LibreYOLOXs 모델에서 시작해 ONNX로 내보낸 다음, 서로 다른 두 하드웨어인 Hailo-8L 가속기와 Qualcomm Snapdragon용으로 ONNX를 컴파일했습니다.

## Hailo-8L에서 실시간 실행

ONNX를 Hailo Dataflow Compiler로 HEF로 컴파일한 다음, Hailo-8L AI HAT을 장착한 Raspberry Pi 5에서 실행했습니다.

![Hailo-8L을 탑재한 Raspberry Pi 5에서 LibreYOLOXs가 실행되며 번화한 거리의 사람과 핸드백을 탐지하는 모습](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Raspberry Pi에서 프레임당 19.0 ms, 52.6 FPS로 실행됐습니다.

## Snapdragon에서 실시간 실행

Qualcomm 쪽에서는 LibreYOLOXs를 INT8로 양자화하고 SNPE, QAIRT, AI Hub 스택을 통해 실행했으며, QCS6490에서 벤치마크했습니다.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="Snapdragon 휴대폰의 EdgeVision AI 앱에서 LibreYOLOXs가 TV, 노트북, 키보드, 마우스, 휴대전화를 실시간으로 탐지하는 모습" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

HTP/DSP에서 6.69 ms를 기록했습니다.

## 감사 인사

프로젝트를 소개해 주신 Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera, Shan Ahmed Shaffi께 진심으로 감사드립니다.

- 튜토리얼 및 슬라이드: [Edge AI in Action: 온디바이스 추론 마스터하기](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Qualcomm 양자화 LibreYOLOXs 모델: [Hugging Face에서 확인](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## 사용해 보기

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO는 MIT 라이선스를 따르며 Linux, Mac, Windows에서 실행되고, 코드 변경 없이 GPU, Apple Silicon, 일반 CPU에서 작동합니다. 하나의 API로 YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, 분할, 자세 추정, 깊이 추정 등을 사용할 수 있습니다.

GitHub에서 Star를 눌러 주세요: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 문서: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
