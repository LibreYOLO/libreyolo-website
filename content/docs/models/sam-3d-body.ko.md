---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: 'SAM 3D Body: LibreYOLO의 전신 메시 복원'
description: >-
  LibreYOLO에서 SAM 3D Body로 전신 인체 메시를 복원합니다. 설치 후 예측할 수 있으며 체크포인트에는 Meta의 SAM
  License 접근 제한과 CUDA가 필요합니다.
lead: >-
  SAM 3D Body는 단일 이미지와 사람 박스에서 손과 발을 포함한 전신 3D 메시를 복원하는 Meta의 프롬프트 가능 모델입니다.
  LibreYOLO는 이를 이식하는 대신 업스트림 패키지를 래핑합니다.
keywords:
  - SAM 3D Body 사용법
  - 인체 메시 복원
  - 바디 메시
  - MHR
  - Momentum Human Rig
  - 3D 자세 추정
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # 이 계열은 LibreYOLO() 팩토리에 등록되지 않아 직접 생성합니다.
        # model_path=None이면 접근 제한이 있는 Hugging Face 다운로드가 실행됩니다.
        # 문자열은 기존 로컬 체크포인트 경로로 처리되며 자동으로 가져오지 않습니다.
        # 추론에는 CUDA 장치가 필요하며 CPU 경로는 없습니다.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.vertices.shape)    # (N, V, 3), 카메라 좌표계, 미터
        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: 사람 탐지기 사용
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # 이름 문자열 단축형은 없습니다. 생성된 LibreYOLO 탐지기나 일반
        # callable 또는 PersonDetector 인스턴스를 전달합니다.
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## 설치

```bash
pip install libreyolo
```

이 명령은 LibreYOLO 어댑터만 설치합니다. SAM 3D Body 자체는 번들로 제공되지 않습니다. 해당 라이선스로부터 LibreYOLO의 자체 코드를 파생할 수 없기 때문입니다. 업스트림 저장소를 복제하고 종속성을 직접 설치한 뒤 LibreYOLO가 복제본을 가리키도록 설정합니다.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

호출할 때마다 `sam_3d_body_path`를 전달하는 대신 `SAM_3D_BODY_PATH` 환경 변수를 설정할 수도 있습니다. 이 계열을 생성하지 않으면 가져오기가 실행되지 않으며 SAM License와도 마주치지 않습니다. 이 계열은 `LibreYOLO()` 팩토리나 `libreyolo predict` CLI 명령에 연결되어 있지 않습니다. 유일한 진입점은 `LibreSAM3DBody`입니다.

## 예측

<code-tabs name="predict" />

체크포인트 다운로드에는 접근 제한이 있습니다. Hugging Face 모델 페이지에서 Meta의 라이선스에 동의하고 첫 다운로드 전에 `hf auth login`으로 인증해야 합니다. 추론 자체에는 조건 없이 CUDA 장치가 필요합니다. 업스트림 추정기가 확인 없이 배치를 GPU로 이동하므로 CPU 전용 시스템에서는 대체 경로로 전환하지 않고 예외가 발생합니다. `result.meshes`는 `result.boxes`와 행이 정렬된 `Meshes` 페이로드이며 탐지된 사람마다 행 하나가 대응합니다. `vertices`와 `joints3d`는 미터 단위이고 추정된 카메라 이동을 이미 포함합니다. `joints2d`는 원본 이미지의 픽셀 좌표이며 회전은 축각이 아닌 오일러 각을 사용하는 MHR 규칙을 따릅니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

## 변형

동일한 MHR 바디 모델에 두 백본을 사용할 수 있습니다. `d3`는 DINOv3 ViT-H/16+ 인코더를 사용하고 `h`는 원래의 ViT-H 인코더를 사용합니다.

## 내보내기

<export-matrix />

바디 메시 내보내기는 구현되지 않았습니다. LibreYOLO는 MHR 매개변수 레이아웃을 PyTorch 외부에서 표현하는 방법을 포함해 메시 작업용 내보낸 그래프 계약을 아직 정의하지 않았습니다.

## 체크포인트

이 계열에 공개된 모든 가중치 파일입니다.

<checkpoint-table />

## 라이선스

<provenance-box>

체크포인트가 구동하는 바디 모델인 MHR(Momentum Human Rig)은 Apache-2.0으로 공개된 별도의 Meta 릴리스입니다. LibreYOLO는 런타임에 MHR의 자체 공개 릴리스에서 TorchScript 자산을 가져와 로컬에 캐시합니다. 이 파일은 LibreYOLO에서 미러링하지 않으며 SAM License가 아니라 자체 Apache-2.0 약관이 적용됩니다.

</provenance-box>

## 인용

<citation-block />
