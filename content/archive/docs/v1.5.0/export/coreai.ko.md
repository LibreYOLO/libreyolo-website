---
title: Core AI
seo_title: LibreYOLO에서 Apple Core AI로 내보내기
description: >-
  LibreYOLO 모델을 Apple Core AI의 .aimodel 자산으로 내보내기: macOS 전용, 고정 캔버스, FP32, 그리고
  사용하는 쪽이 지켜야 하는 명명된 출력 순서 규약.
lead: >-
  Core AI는 Apple의 온디바이스 추론 스택입니다. LibreYOLO는 torch.export로 모델을 캡처하고 Core AI 변환기를
  거쳐 로우어링한 뒤, 모델 메타데이터와 내보낸 출력 이름을 담은 .aimodel 자산을 씁니다.
keywords:
  - libreyolo core ai 내보내기
  - aimodel
  - coreai-torch
  - torch.export apple
  - apple 온디바이스 추론
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: 플래그
    value: export(format="coreai")
    mono: true
  - label: 생성물
    value: 메타데이터가 붙은 .aimodel 자산 하나
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: 다시 불러오기
    value: LibreYOLO로는 불가능합니다. 사용하는 쪽에서 Core AI 런타임을 직접 씁니다.
  - label: 형태
    value: 고정 캔버스. dynamic=True는 NotImplementedError를 발생시킵니다.
  - label: 정밀도
    value: FP32만 지원합니다. half=True와 int8=True는 거부됩니다.
  - label: 요구 사항
    value: 'macOS. 툴체인은 다른 곳에서 변환도 실행도 되지 않으며, coreai-torch는 torch를 2.11.x에 고정합니다.'
verification: >-
  dev 브랜치의 libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py, pyproject.toml에서
  확인했습니다.
snippets:
  install:
    - label: '설치, macOS에서'
      language: bash
      code: |
        # 모든 통합 extra에서 의도적으로 제외했습니다: coreai-torch가 torch를
        # 2.11.x에 고정해 환경 전체를 그 버전으로 끌고 가기 때문입니다.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.aimodel을 씁니다
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: 인자
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int 또는 (height, width); 실행 캔버스입니다
            batch=1,
            output_path=None, # None이면 weights/<stem>.aimodel에 씁니다
        )

        # dynamic=True는 NotImplementedError를 발생시킵니다.
        # half=True와 int8=True는 검증 단계에서 거부됩니다.
  outputs:
    - label: 연동하기 전에 출력 순서 확인하기
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="coreai", imgsz=640)

        # 자산 메타데이터는 내보낸 출력 이름을 그래프 순서대로
        # "coreai_output_names" 아래에 기록합니다. 그 목록으로 Core AI가 반환한
        # 딕셔너리를 이름으로 매핑하고, eager 튜플과 위치로 짝지어서는 안 됩니다.
  support:
    - label: 내보내기 전에 계열과 작업 하나 확인하기
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## 설치

이 형식은 macOS 전용입니다. `coreai-torch` 요구 사항에는
`sys_platform == 'darwin'` 마커가 붙어 있고, 툴체인은 다른 어디에서도 변환하거나
실행되지 않습니다.

<code-tabs name="install" />

이 extra는 `coreai-torch`가 torch를 2.11 계열에 고정하기 때문에 `libreyolo[all]`을
포함한 모든 통합 extra 바깥에 있습니다. 그 조합으로 제약해도 괜찮은 환경에
설치하십시오.

## 내보내기

<code-tabs name="export" />

캡처는 단일 기록 트레이스가 아니라 가드를 갖춘 실제 그래프 캡처인
`torch.export`입니다. 이는 Core ML 경로보다 엄격합니다: 호스트 스칼라 읽기와 데이터
의존 제어 흐름을 조용히 고정해 넣지 않고 거부하며, 그래서 일부 모델 계열은 캡처
실패가 기록된 채 여기서 차단됩니다.

세 가지 준비 단계는 내보내기가 성공하든 실패하든 호출자의 살아 있는 모델을
복원하는 스코프 안에서 실행됩니다. Darknet 계열 모델은 Core AI 0.4.1이 Darknet의
제곱근 뒤 엡실론 공식을 보존하지 않기 때문에 추론용 배치 정규화가 앞선 합성곱에
정확히 접혀 들어갑니다. 그리드와 앵커 계열 모델은 고정 캔버스에 맞춰 앵커가
고정됩니다. RF-DETR 모델은 변환기에 `aten._upsample_bicubic2d_aa`에 대한 로우어링이
없기 때문에, 모델 자체의 베이킹 경로를 다시 실행해 요청한 캔버스에 맞게 위치
임베딩을 다시 굽습니다.

로우어링 단계는 Core AI 변환기에 DETR 계열이 사용하는 디포머블 어텐션 샘플러에
대한 로우어링이 없으므로, `aten.grid_sampler_2d`에 대한 PyTorch 참조 분해를 분해
테이블에 넣습니다.

자산은 최소 OS를 v27로 선언하며, 이는 툴체인이 제공하는 유일한 값입니다. 이는
변환이 아니라 배포를 제한합니다: 변환과 Python 쪽 실행은 휠 안의 런타임으로 이전
macOS에서도 동작하지만, 수치는 OS 버전마다 다르므로 기록된 수치 일치는 macOS 27에서
측정합니다.

## 아티팩트 실행

`libreyolo/backends`에는 Core AI 항목이 없으므로 `LibreYOLO()`는 `.aimodel`을
불러오지 않습니다. 사용하는 쪽에서 Core AI 런타임을 직접 쓰며, 전처리와 디코딩,
NMS, 좌표 재조정은 그쪽 몫입니다. 지원 매트릭스에서 검증됨으로 표시된 행은 내보낸
그래프가 참조와 같은 수치를 계산한다는 뜻이지, `predict`가 그것을 실행한다는 뜻은
아닙니다.

사용하는 쪽에서 스스로 다시 도출할 수 없는 유일한 것은 출력 순서입니다:

<code-tabs name="outputs" />

Core AI는 이름이 붙은 딕셔너리를 반환하는데, 그 키 순서는 eager forward의 튜플
순서와도, 추측 가능한 어떤 것과도 일치하지 않습니다. 내보낸 이름이 바로 이 이유로
자산 메타데이터에 `coreai_output_names`로 기록됩니다. 이름으로 매핑하십시오.

## 제약 사항

고정 캔버스, FP32, 내보낸 그대로의 배치. `dynamic=True`는
`NotImplementedError`를 발생시키고, `half=True`와 `int8=True`는 검증 단계에서
거부됩니다.

변환 쪽 커버리지는 넓습니다. 검증된 조합에는 YOLO9 계열, YOLOX, YOLO7, Darknet
시대의 탐지기 네 종, YOLO-NAS, PicoDet, RTMDet, RT-DETR, RT-DETRv2, RT-DETRv4,
D-FINE, DEIM, DEIMv2, EC, RF-DETR 탐지, CNN 분류 계열 네 종과 클래스가 고정된 CLIP
및 SigLIP2, Depth Anything V2와 ZipDepth, NAFNet과 Real-ESRGAN 복원, PIDNet과
LingBotVision 시맨틱 분할, FOMO 점 탐지가 포함됩니다. 각각 자체적으로 기록된
컨텍스트가 있으며, `libreyolo formats`가 이를 출력합니다.

차단된 조합과 조합별로 기록된 이유:

| 조합 | 이유 |
|---|---|
| EoMT 시맨틱 분할 | 엄격한 캡처가 `GuardOnDataDependentSymNode`로 실패합니다: 마스크 경로의 어딘가에서 텐서의 값을 읽어 그 값으로 분기합니다 |
| SegFormer 시맨틱 분할 | 캡처 경로가 아직 평가되지 않았고, 공개된 가중치는 형식과 무관하게 비상업용입니다 |
| L2CS 시선 추정 | 모델 자체가 ONNX, TorchScript, ExecuTorch, TensorRT, OpenVINO만 지원하며, 이는 모델 쪽의 결정입니다 |
| Depth Anything 3 깊이 추정 | 이 계열은 모든 형식에 대해 내보내기를 거부합니다 |

RF-DETR 모델에는 아티팩트를 비교하기 전에 읽어 둘 만한 주의 사항이 하나 있습니다.
이 모델의 수치 일치는 ONNX가 아니라 Core AI 내보내기 도구가 직접 준비한 그래프를
기준으로 기록되며, 640 캔버스에서는 RF-DETR ONNX 아티팩트가 그 준비된 그래프와
일치하지 않습니다. Core AI의 재베이킹은 eager 모델이 수행하는 안티에일리어싱
리사이즈를 보존하지만, ONNX 경로는 안티에일리어싱을 비활성화합니다. 따라서 ONNX는
네이티브가 아닌 캔버스에서 그 계열의 유효한 기준이 되지 못합니다.

Apple의 이전 형식은 [Core ML](/docs/export/coreml)을 참고하십시오. 전체 계열과 작업
표는 [내보내기 매트릭스](/docs/reference/export-matrix)를 참고하십시오. 조합 하나는:

<code-tabs name="support" />
