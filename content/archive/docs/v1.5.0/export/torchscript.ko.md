---
title: TorchScript
seo_title: LibreYOLO에서 TorchScript로 내보내기
description: >-
  LibreYOLO 모델을 TorchScript로 내보냅니다. 내부에 LibreYOLO 메타데이터가 있고 Python 또는 libtorch에서
  불러올 수 있는 추적된 .torchscript 아카이브를 생성합니다.
lead: >-
  TorchScript는 PyTorch의 자체 직렬화 그래프 형식입니다. LibreYOLO는 torch.jit.trace로 모델을 추적하고
  libreyolo_metadata.json 추가 파일과 함께 결과를 저장하므로 아카이브에 계열, 작업, 클래스 이름, 입력 크기가
  포함됩니다.
keywords:
  - yolo torchscript 변환
  - torch.jit.trace 사용법
  - torch.jit.load 모델 로드
  - libtorch 배포
  - torchscript 메타데이터
  - extra_files
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="torchscript")
    mono: true
  - label: 출력
    value: libreyolo_metadata.json 추가 파일이 포함된 .torchscript 아카이브 하나
  - label: 추가 설치
    value: 없습니다. TorchScript는 PyTorch와 함께 제공됩니다.
  - label: 다시 불러오기
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: 형상
    value: 고정입니다. 그래프는 하나의 입력 형상에서 추적됩니다.
  - label: 정밀도
    value: 'FP32, FP16(half=True)을 지원합니다. INT8은 지원하지 않습니다.'
verification: >-
  dev 브랜치의 libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/torchscript.py를 확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.torchscript를 작성합니다
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: 인수
      language: python
      code: |
        model.export(
            format="torchscript",
            imgsz=640,        # int 또는 (height, width)
            batch=1,
            half=False,       # FP16 가중치와 활성화
            device=None,      # 이 형식에서 None은 CPU로 추적합니다
            output_path=None, # None이면 weights/<stem>.torchscript를 작성합니다
        )

        # dynamic은 허용되지만 아카이브는 항상 고정 형상 추적이며,
        # 내장 메타데이터는 어느 경우든 dynamic=False를 기록합니다.
  run:
    - label: LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: PyTorch 직접 사용
      language: python
      code: |
        import json

        import torch

        extra_files = {"libreyolo_metadata.json": ""}
        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )
        module.eval()

        metadata = json.loads(extra_files["libreyolo_metadata.json"])
        print(metadata["model_family"], metadata["task"], metadata["imgsz"])

        # 이 경로에서는 전처리와 후처리를 직접 담당합니다.
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: 내보내기 전 계열 및 작업 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## 설치

<code-tabs name="install" />

`torch.jit`가 PyTorch와 함께 제공되므로 TorchScript에는 기본 설치 외에 필요한
것이 없습니다. 선택적 의존성과 외부 변환기가 모두 없는 유일한 내보내기 대상이므로
더 긴 도구 체인이 실패할 때 유용한 첫 확인 수단입니다.

## 내보내기

<code-tabs name="export" />

디바이스를 지정하지 않으면 CPU에서 추적하며, `output_path`를 생략하면 체크포인트
이름을 사용하여 `weights/`에 아카이브를 작성합니다.

`torch.jit.trace`가 일반적으로 수행하는 재추적 검사는 비활성화됩니다. 일부
내보내기 래퍼는 첫 순전파 중에 형상 종속 앵커를 캐시하므로 기록된 고정 형상
그래프가 올바른 경우에도 두 번째 추적은 다른 Python 경로를 관찰합니다. 대신
동등성 테스트가 저장된 모듈을 직접 검증합니다.

메타데이터는 사이드카에 있지 않습니다. `torch.jit.save`가
`libreyolo_metadata.json`을 아카이브 내부에 저장하고, `torch.jit.load`는
`_extra_files`를 통해 이를 반환합니다.

## 아티팩트 실행

<code-tabs name="run" />

`LibreYOLO()`는 `.torchscript` 접미사를 기준으로 라우팅하고 원본 체크포인트와
동일한 `Results` 객체를 반환합니다. `device="auto"`를 사용하면 가능할 때 CUDA,
그다음 MPS, 마지막으로 CPU에 모듈을 매핑합니다.

두 번째 스니펫은 LibreYOLO가 설치되지 않은 경우와 `torch::jit::load`로 같은
아카이브를 불러오는 libtorch C++ 배포를 위한 경로입니다. 이 경로에서는 전처리,
디코딩, NMS, 좌표 크기 조정을 직접 담당합니다. 메타데이터 추가 파일은 여전히
읽을 수 있으며 클래스 이름이 존재하는 유일한 위치입니다.

## 제약 조건

그래프는 하나의 입력 형상에서 생성된 추적입니다. 인터페이스 대칭성을 위해
`dynamic=True`를 허용하지만 아무것도 바뀌지 않으며, 백엔드가 사용할 수 없는
축을 가정하지 않도록 내장 메타데이터는 `dynamic=False`를 보고합니다. 다른
해상도에는 별도의 아카이브를 내보냅니다.

`half=True`는 모델과 추적 입력을 FP16으로 변환합니다. INT8 경로는 없습니다.
`int8=True`는 검증 중 `NotImplementedError`를 일으킵니다.

직사각형 `imgsz`는 YOLO9 계열, HRNet, NAFNet, Real-ESRGAN에서 작동하며, 고정
정사각형 계약이 있는 계열에서는 거부됩니다.

다섯 가지 조합은 추적 전에 거부됩니다. LibreYOLO의 YOLO9은 탐지만 지원하므로
YOLO9 분할이 거부됩니다. 동적 커널 마스크 디코딩에 내보낸 런타임 계약이 없는
RTMDet-Ins 분할도 거부됩니다. 가변 길이 또는 동적 앵커 그래프의 동등성 근거가
ONNX Runtime 계약에만 있는 SSD, Faster R-CNN, RetinaNet 탐지도 거부됩니다.

전체 계열 및 작업 표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참조합니다. 조합 하나를 확인하려면 다음을 실행합니다.

<code-tabs name="support" />
