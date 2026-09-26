---
title: Hailo
seo_title: Hailo 가속기에서 LibreYOLO 모델 실행
description: >-
  LibreYOLO 모델을 Hailo-8 또는 Hailo-8L에 배포합니다. 정적 ONNX 내보내기, 직접 실행하는 Dataflow
  Compiler 단계, 컴파일 가능한 아키텍처를 설명합니다.
lead: >-
  Hailo 가속기는 Hailo Developer Zone을 통해 배포되는 독점 SDK인 Hailo Dataflow Compiler로
  컴파일합니다. 이 흐름에서 LibreYOLO의 역할은 단순한 정적 ONNX 내보내기이며, 이후 DFC에서 파싱, 양자화, HEF 컴파일을
  수행합니다.
keywords:
  - libreyolo hailo 배포
  - hailo-8 모델 컴파일
  - hailo-8l
  - raspberry pi ai kit yolo
  - ai hat+
  - hailo dataflow compiler
  - hef 컴파일
  - hailortcli 사용법
last_verified: 1.5.0
meta:
  - label: LibreYOLO 단계
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: 형식 아님
    value: format="hef"는 없습니다. DFC는 pip 의존성이 될 수 없습니다.
  - label: 추가 설치
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: 컴파일 호스트
    value: WSL2 Ubuntu 22.04를 포함한 Linux x86_64입니다. ARM에서는 컴파일할 수 없습니다.
  - label: 컴파일 가능
    value: '고정 형상 그래프를 사용하는 순수 CNN입니다. 어텐션, 동적 형상, LayerNorm 중심 설계는 컴파일되지 않습니다.'
  - label: 상태
    value: 아직 어떤 LibreYOLO 계열도 DFC 전체 과정을 거쳐 실행되는 HEF로 완성되지 않았습니다.
verification: >-
  dev 브랜치의 skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py,
  libreyolo/cli/commands/export.py를 확인했습니다. DFC 제약 조건은 해당 스킬에 기록된 내용이며, 컴파일 및
  측정이 완료된 LibreYOLO HEF는 없습니다.
snippets:
  install:
    - label: LibreYOLO 측
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 직접 설치하는 Hailo 측
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hailo에는 배치 1, 고정 해상도, 동적 축 없음이 필요합니다.
        # Python API의 기본값은 dynamic=True이므로 명시적으로 끕니다.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # CLI는 이미 정적 형상을 기본값으로 사용합니다.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: 컴파일 전 정적 그래프 확인
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: '파싱, 양자화, 컴파일'
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # YOLOX는 end_node_names 없이 한 번 변환하면 DFC 로그에

        # 권장 종료 노드가 표시됩니다. 해당 노드로 다시 실행합니다.

        runner.translate_onnx_model(ONNX)


        # 정규화는 LibreYOLO 전처리와 일치해야 합니다. YOLOX와 YOLO9에는

        # 평균이나 표준 편차 없이 0-255에서 0-1로의 스케일만 필요합니다.

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # 선택 사항: Hailo가 NMS를 담당하게 합니다. 구성은 클래스 수와 입력

        # 크기에 모두 종속되므로 COCO-80 구성은 파인튜닝한 3클래스 모델에

        # 맞지 않습니다. 이 줄이 없으면 HEF가 원시 헤드 텐서를 출력하고

        # 애플리케이션이 이를 디코딩합니다.

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # 보정 이미지는 배포 데이터를 대표해야 합니다.

        # 무작위 이미지는 컴파일되지만 정확도를 소리 없이 훼손합니다.

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: YOLO9 종료 노드
      language: python
      code: |
        # LibreYOLO 그래프는 다른 내보내기용 구성에 나타나는 "model.N"
        # 접두사가 아니라 "/head/..." 접두사를 사용합니다. 복사한 구성은
        # 일치하지 않습니다. 파싱에 실패하면 자체 그래프에서 이름을 확인합니다.
        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]
        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: AI Kit 또는 AI HAT+를 장착한 Raspberry Pi 5
      language: bash
      code: |
        sudo apt install dkms hailo-all
        hailortcli fw-control identify       # 디바이스를 확인하고 아키텍처 이름을 표시합니다
        hailortcli run libreyoloxs.hef       # 스모크 테스트 및 처리량
source_hash: 33b077f1c23d5535
---

## 설치

LibreYOLO에는 `format="hef"`가 없으며 앞으로도 추가되지 않습니다. Hailo
Dataflow Compiler는 Developer Zone 등록 뒤 비공개 휠로 배포되는 독점 SDK이므로
의존성이나 extra로 포함할 수 없습니다. 배포는 두 단계로 진행됩니다. LibreYOLO가
정적 ONNX 파일을 작성한 다음 DFC로 처리합니다.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## 내보내기

<code-tabs name="export" />

`half=True`를 전달하지 마십시오. DFC는 FP32 ONNX를 입력받아 자체적으로 INT8
양자화를 수행합니다. `nms=True`도 전달하지 마십시오. Hailo가
`nms_postprocess`를 통해 NMS를 담당하거나 애플리케이션이 담당하므로 종료 노드
뒤의 NMS 서브그래프는 불필요합니다. 기본 opset을 사용할 수 있으며, DFC 파서가
거부하면 `opset=11`로 다시 내보냅니다.

DFC는 지정한 종료 노드에서 그래프를 자르고 이후의 모든 요소를 버리며, 종료
노드는 탐지 헤드 컨볼루션입니다. 따라서 LibreYOLO의 일반적인 디코딩된 ONNX도
입력으로 사용할 수 있습니다. 파서가 디코딩 꼬리 부분을 단순히 무시합니다.

## 컴파일

<code-tabs name="compile" />

대상에 맞는 `hw_arch`를 선택합니다. Hailo-8, 26 TOPS AI HAT+, M.2 및 PCIe
모듈에는 `hailo8`을 사용하고, Hailo-8L, Raspberry Pi AI Kit, 13 TOPS AI
HAT+에는 `hailo8l`을 사용합니다. Hailo-10H에는 일치하는 최신 DFC와 Model Zoo가
필요하며 `hailo10h`를 사용합니다. 확실하지 않으면 디바이스에서
`hailortcli fw-control identify`를 실행해 확인할 수 있습니다.

두 계열은 HailoRT NMS 메타 아키텍처에 매핑되므로 Hailo가 컴파일된 파이프라인
내부에서 억제를 담당할 수 있습니다. YOLOX는 `meta_arch=yolox`를 사용하며,
YOLO9은 헤드 레이아웃이 동일한 Hailo의 분리형 헤드 메타 아키텍처를 사용합니다.
Hailo Model Zoo에서 일치하는 `nms_postprocess` 구성을 가져와 클래스 수와 입력
크기에 맞게 조정합니다. 그 외 컨볼루션 탐지기는 모두 일치하는 메타 아키텍처가
없는 그래프로 컴파일됩니다. HEF는 원시 헤드 텐서를 출력하고 애플리케이션이
CPU에서 디코딩과 NMS를 실행합니다.

문제가 발생하면 컴파일 로그를 보관합니다. 모든 수정은 정확히 실패한 레이어나
연산자 이름에 달려 있습니다.

## 아티팩트 실행

<code-tabs name="device" />

애플리케이션 추론은 `hailo_platform` Python API를 사용합니다.
`nms_postprocess`를 컴파일해 넣으면 출력은 모델 좌표의
`[y1, x1, y2, x2, score]`를 담은 `(batch, num_classes, max_dets, 5)`이며,
애플리케이션에서 원본 이미지 크기로 직접 조정합니다. 실행 시 LibreYOLO의
`Results` 파이프라인은 관여하지 않습니다. HEF는 독립 실행형 아티팩트이며
전처리와 후처리는 애플리케이션이 담당합니다.

## 제약 조건

모델이 Hailo-8 또는 Hailo-8L을 대상으로 할 수 있는지는 이름이 아니라
아키텍처의 속성이므로, 아래 규칙은 이 페이지가 작성된 뒤 추가된 계열에도
적용됩니다.

다음 중 하나라도 포함된 모델은 컴파일되지 않습니다.

- 자체, 교차, 변형 가능, 윈도우 방식 등 모든 종류의 어텐션입니다. 따라서 모든
  DETR 계열 탐지기, 모든 오픈 보캐뷸러리 또는 텍스트 조건부 탐지기, 모든 ViT
  백본, 모든 언어 또는 비전 언어 타워가 제외됩니다. Hailo 자체 Zoo에는 수작업으로
  튜닝한 트랜스포머 HEF가 몇 개 포함되지만 이는 공급업체가 맞춤 제작한 결과이며,
  임의의 어텐션 그래프를 컴파일할 수 있다는 근거가 아닙니다.
- 동적 형상 또는 데이터 종속 제어 흐름입니다. DFC는 하나의 고정 입력 형상과 정적
  그래프를 컴파일하므로 가변 쿼리 수, 텍스트 프롬프트, 동적 top-k, 동적 인덱스를
  사용하는 `NonZero`, `Gather`, `TopK`, 그리고 `grid_sample`은 모두 제외됩니다.
- LayerNorm 또는 GELU 중심 설계입니다. BatchNorm은 컨볼루션에 깔끔하게
  접히지만 LayerNorm 지원은 부족하고 GELU는 네이티브 활성화가 아니므로, 명목상
  컨볼루션 방식이어도 ConvNeXt 계열 스택은 적합하지 않습니다.
- 네이티브 해상도의 이미지 간 변환 작업입니다. 복원 모델은 전체 입력 해상도로
  실행되며 실용적인 Hailo SRAM 용량을 초과합니다.

컨볼루션만 사용하고 ReLU 또는 SiLU와 함께 BatchNorm을 사용하며 입력 크기가
고정된 계열이 후보입니다. 이 라이브러리에서는 CNN 단일 단계 탐지기가 이에
해당하며 YOLOX와 YOLO9이 주요 대상입니다. PicoDet, YOLO-NAS, RTMDet 같은 다른
컨볼루션 탐지기는 애플리케이션 측에서 디코딩합니다. CNN 분류기인 ResNet,
MobileNetV4-conv, EfficientNetV2도 해당하며, Hailo Model Zoo가 ResNet용 레시피를
제공하므로 ResNet 지원이 가장 좋습니다. ResNet 백본의 FOMO 포인트 탐지와 L2CS
시선 추정 같은 소형 컨볼루션 작업 헤드도 원칙적으로 컴파일할 수 있지만 Hailo
레시피가 없습니다.

이 페이지의 어떤 내용도 지원되는 것으로 제시하지 않는 이유가 되는 상태상 주의
사항이 하나 있습니다. 어떤 LibreYOLO 계열도 DFC 전체 과정을 거쳐 실행되는 HEF로
완성되지 않았습니다. 위 규칙은 아키텍처를 바탕으로 컴파일 가능성을 예측합니다.
HEF를 컴파일하고 측정하기 전까지 파서 동작, 양자화, 정확도는 입증되지 않았으므로
모든 후보에는 각각 기록된 증거가 필요합니다. 정확한 체크포인트에서 컴파일한
HEF와 기록된 DFC, Model Zoo, HailoRT 버전, 문서화된 보정, 그리고 처리량 수치가
아닌 FP32 베이스라인 대비 온디바이스 정확도 비교가 필요합니다.

모델이 부적격이라면 동등성이 기록된 런타임을 대안으로 사용합니다.
[ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt),
[OpenVINO](/docs/export/openvino)가 있습니다.
