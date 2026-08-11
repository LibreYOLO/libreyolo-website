---
title: Core ML
seo_title: LibreYOLO에서 Core ML로 내보내기
description: >-
  LibreYOLO 탐지기를 Core ML의 .mlpackage로 내보내기: ImageType 입력 계약, FP16, 컴퓨트 유닛, 내장
  NMS, 그리고 지원되는 네 가지 계열.
lead: >-
  Core ML은 Apple의 온디바이스 모델 형식입니다. LibreYOLO는 계열별 전처리 래퍼 뒤에서 탐지기를 트레이스하므로 변환된
  그래프는 항상 표준 RGB 이미지 입력을 받으며, 그런 다음 모델 메타데이터를 첨부한 ML Program 형식의 .mlpackage를
  기록합니다.
keywords:
  - yolo coreml 변환
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml nms 파이프라인
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="coreml")
    mono: true
  - label: 출력
    value: ML Program 형식의 .mlpackage 번들(디렉터리) 하나
  - label: 추가 의존성
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: 다시 불러오기
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") on macOS
    mono: true
  - label: 형상
    value: 고정입니다. 입력은 형상이 고정된 ct.ImageType입니다.
  - label: 정밀도
    value: 'FP32, FP16(half=True). INT8은 없습니다.'
  - label: 계열
    value: '탐지 전용이며 yolox, yolo9, rtdetr, rfdetr 계열에 적용됩니다'
verification: >-
  dev 브랜치의 libreyolo/export/coreml.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/coreml.py, pyproject.toml에서
  확인했습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.mlpackage 번들을 기록합니다
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: 인자
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True로 두면 FLOAT16 연산 정밀도로 변환합니다
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None이면 weights/<stem>.mlpackage에 기록합니다
        )

        # dynamic 인자는 받아들여지지만 입력은 형상이 고정된 ct.ImageType이며,
        # 내장 메타데이터는 어느 쪽이든 dynamic=False로 기록합니다.
  nms:
    - label: Apple의 NMS 레이어 내장
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOX와 YOLO9 탐지 전용, 배치 1.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: macOS에서 LibreYOLO로 실행
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # 또는 cpu_and_ne로 Neural Engine에 고정
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: coremltools 단독 사용
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # 입력은 내보내기 시 고정된 크기의 "image"라는 이름의 이미지입니다.
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # 이 경로에서는 레터박싱과 후처리를 직접 해야 합니다.
  support:
    - label: 내보내기 전에 계열과 작업 조합 확인
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## 설치

<code-tabs name="install" />

예측에는 macOS가 필요합니다. `LibreYOLO()`는 다른 플랫폼에서 `.mlpackage`를 거부하면서
현재 플랫폼 이름을 담은 메시지를 표시하며, 런타임 동등성 확인에는 macOS 러너가 필요하다는
이유로 지원 매트릭스는 이 조합들을 사용 가능으로 기록합니다.

## 내보내기

<code-tabs name="export" />

번들은 체크포인트의 파일 이름을 따서 `weights/` 아래에 기록되며, `half=True`일 때는
`_fp16`이 덧붙습니다. `.mlpackage`는 디렉터리이므로 트리 전체를 복사해야 합니다.

모든 계열은 전처리 래퍼 뒤에서 트레이스되므로, 변환된 그래프는 하나의 표준 입력만 받습니다:
RGB, `scale=1/255`, 바이어스 없음, `ct.ImageType`으로 선언. 래퍼는 계열 고유의 규약을
흡수하는데, YOLOX는 0에서 255 범위의 BGR, RF-DETR 계열은 ImageNet 평균과 표준편차,
YOLO9과 RT-DETR 계열은 항등 변환을 씁니다. 그래서 Core ML 모델을 사용하는 쪽은 계열별
텐서가 아니라 평범한 이미지를 넣습니다.

변환은 최소 배포 대상 iOS 15로 ML Program을 목표로 합니다. `compute_units`는 변환된
모델에 저장되며, 아티팩트를 불러올 때 다시 덮어쓸 수 있습니다.

모델 메타데이터는 문자열로 `user_defined_metadata`에 들어가며, 백엔드는 여기에서 계열,
작업, 클래스 이름, 입력 크기, 자세 스키마를 읽습니다.

### 내장 NMS

<code-tabs name="nms" />

`nms=True`는 모델을 Apple의 `NonMaximumSuppression` 레이어로 끝나는 Core ML
파이프라인으로 감쌉니다. 결과에는 출력이 두 개 있습니다: 형상이 `N` x 클래스 개수인
`confidence`와, 정규화된 `xywh`로 형상이 `N` x 4인 `coordinates`입니다.

YOLOX와 YOLO9 탐지에만 적용되며, 배치 1이 필요합니다. DETR 계열 모델은 이름으로
거부되는데, 집합 예측은 쿼리와 클래스에 대해 IoU 단계 없이 top-k를 취하므로 그 레이어를
쓸 수 없기 때문입니다. `max_det`도 여기에서는 노출되지 않으므로, 탐지 개수 상한이
중요하다면 [ONNX 내장 NMS](/docs/export/onnx)를 대신 사용하십시오.

## 아티팩트 실행

<code-tabs name="run" />

`LibreYOLO()`는 `.mlpackage` 접미사가 붙은 디렉터리를 인식하고, 체크포인트와 동일한
`Results` 객체를 반환합니다. `compute_units`는 이 형식에서 팩토리가 그대로 전달하는 유일한
인자이며, `all`, `cpu_and_gpu`, `cpu_and_ne`, `cpu_only`를 받습니다. `device` 인자는
무시되는데, Core ML은 대신 컴퓨트 유닛으로 경로를 정하기 때문입니다.

두 번째 스니펫은 런타임을 직접 다루는 경로입니다. 여기에서는 레터박싱, 디코딩, NMS,
좌표 재조정을 직접 처리해야 하며, 클래스 이름은 `user_defined_metadata`에 들어 있습니다.

## 제약

네 개 계열, 탐지 전용입니다: `yolox`, `yolo9`, `rtdetr`, `rfdetr`. 그 외에는 사전 점검에서
거부되는데, 고정된 이미지 입력 계약이 올바르게 성립하는 것은 계열을 인식하는 전처리 래퍼
덕분이며, 여기에 없는 계열은 잘못된 정규화로 변환되기 때문입니다. 오류 메시지는 대안으로
ONNX와 TorchScript를 알려 줍니다.

입력 형상은 `ct.ImageType`으로 고정되어 있어서 `dynamic=True`는 아무것도 바꾸지 않으며,
메타데이터에는 `dynamic=False`가 기록됩니다. 다른 해상도가 필요하면 번들을 하나 더
내보내야 합니다.

`half=True`는 FP16 연산 정밀도로 변환합니다. 이 내보내기 도구에는 INT8 경로가 없습니다.

전체 계열과 작업 조합표는 [내보내기 매트릭스](/docs/reference/export-matrix)를
참고하십시오. Apple의 더 새로운 온디바이스 형식은 [Core AI](/docs/export/coreai)를
참고하십시오. 특정 조합 하나만 확인하려면:

<code-tabs name="support" />
