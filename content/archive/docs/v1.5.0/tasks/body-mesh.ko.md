---
title: 바디 메시
seo_title: LibreYOLO에서 신체 메시 복구
description: >-
  LibreYOLO에서 사람마다 매개변수 3D 신체 메시를 복구합니다. 사람 박스 또는 탐지기에서 예측하고, 정점, 관절 및 카메라 이동을
  읽습니다.
lead: >-
  신체 메시 복원은 단일 이미지와 사람 박스 세트를 각 사람에 대한 파라메트릭 3D 신체로 변환합니다: 형태와 자세 매개변수, 포즈된 정점,
  3D 관절, 그리고 이를 렌즈 앞에 배치하는 카메라 이동.
keywords:
  - 휴먼 메시 복원 파이썬
  - 바디 메시
  - 3D 신체 자세
  - SAM 3D 몸
  - MHR
  - 파라메트릭 신체 모델
  - libreyolo 메시 작업
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # 이 계열은 LibreYOLO() 팩토리에 등록되어 있지 않으므로
        # 직접 구성됩니다. model_path=None은 게이트를 활성화합니다
        # Hugging Face 다운로드; 문자열은 기존 로컬로 취급됩니다
        # 체크포인트이며 절대 가져오지 않습니다. 추론에는 CUDA가 필요합니다.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.body_model)      # 이 텐서들이 사용하는 매개변수화
        print(meshes.vertices.shape)  # (N, V, 3), 카메라 프레임, 미터
        print(meshes.joints3d.shape)  # (N, J, 3)
        print(meshes.joints2d.shape)  # (N, J, 2), 원본 이미지上的 픽셀
    - label: 사람 탐지기와 함께
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # person_detector는 구성된 LibreYOLO 탐지기와 일반 탐지기를 허용합니다
        # 호출 가능(callable)하거나 PersonDetector 인스턴스입니다. 이름 단축키는 없습니다.
        detector = LibreYOLO("LibreYOLO9s.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## 정의

바디 메시 복구는 이미지당 `Meshes` 페이로드를 반환하며, `result.boxes`와 행 정렬됩니다: 행 `i`는 박스 `i`에 있는 사람을 설명하며, 포즈 작업이 키포인트에 사용하는 동일한 계약입니다.

모든 것은 원본 이미지의 카메라 프레임에서 표현됩니다. `transl`는 미터 단위의 측정값이며, +z는 카메라에서 멀어지는 방향을 가리킵니다. `vertices`와 `joints3d`는 미터 단위이며 이미 `transl`를 포함하고 있으므로 추가 합성이 필요 없습니다. `joints2d`는 네트워크가 본 크롭이 아니라 원본 이미지 캔버스에서의 픽셀 단위입니다. `faces`는 모든 사람에게 공유되므로 각각의 행마다가 아니라 이미지 전체에 한 번만 메시 토폴로지를 저장합니다. 이 버전에는 월드 프레임이나 중력 프레임이 없으며, 어떤 필드도 이를 대신하지 않습니다.

매개변수 레이아웃은 신체 모델마다 다르므로 형태에 대해 고정된 것은 없습니다: `body_model`는 매개변수화를 나타내며, 개수는 텐서에서 읽어옵니다. `"mhr"`, 즉 Momentum Human Rig의 경우 회전은 축-각이 아니라 라디안 단위의 오일러 각입니다. `body_pose`는 각 관절마다 한 쌍이 아닌 평면화된 관절별 매개변수 벡터이고, `betas`는 정체성 블렌드셰이프 계수입니다. 골격 크기, 손 자세 및 얼굴 표정은 `extras`에 있습니다.

표준 작업 키는 `mesh`입니다. `body-mesh`, `hmr` 및 `human-mesh-recovery`는 이것으로 정규화됩니다.

## 모델들

[SAM 3D Body](/docs/models/sam-3d-body)은 이 작업을 수행하는 유일한 계열이며, 포트가 아니라 래퍼입니다. Meta의 `sam-3d-body` 패키지는 SAM 라이선스 하에 공개되어 있으며, LibreYOLO의 자체 코드는 이를 기반으로 할 수 없으므로 어느 것도 벤더링되지 않았습니다. 두 개의 백본은 동일한 MHR 바디 모델을 공유하며, `d3`는 DINOv3 ViT-H/16+ 인코더에서, `h`는 원래 ViT-H에서 사용됩니다.

첫 번째 예측 전에 세 가지 요구 사항이 적용되며, 그 중 어느 것도 선택 사항이 아닙니다.

상위 패키지는 LibreYOLO가 아니라 사용자가 설치합니다:

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

라이브러리를 `sam_3d_body_path=` 또는 `SAM_3D_BODY_PATH` 환경 변수가 설정된 클론으로 지정하십시오. 이 계열를 한 번도 구성하지 않은 사용자는 가져오기를 절대 실행하지 않습니다.

체크포인트 미러는 게이트가 설정되어 있습니다. Hugging Face 모델 페이지에서 라이선스를 수락하고 `hf auth login`로 인증하지 않으면 첫 번째 다운로드가 실패합니다. MHR 본체 모델 자체는 별도의 Apache-2.0 릴리스로, 자체 공개 위치에서 가져와 로컬에 캐시됩니다.

추론에는 CUDA 장치가 필요합니다. 상위 추정기는 확인 없이 배치를 GPU로 이동시키므로 돌아갈 CPU 경로가 없으며 `device="cpu"`가 발생합니다.

## 예측

<code-tabs name="predict" />

사람들은 모델에 두 가지 방법 중 하나로 도달합니다. `person_boxes`는 이미 보유하고 있는 박스를 전달하며, 단일 이미지에만 해당됩니다: 고정된 박스 집합은 비디오 프레임을 따라 이동할 수 없으므로, 비디오 소스와 함께 전달하면 첫 번째 프레임의 박스를 조용히 재사용하는 대신 오류가 발생합니다. `person_detector`는 구성된 LibreYOLO 탐지기, 호출 가능 객체 또는 `PersonDetector`를 받아들이며, 비디오의 경로에 사용됩니다. `focal_length`는 알려진 카메라 내재 매개변수를 제공합니다; 설정하지 않으면 모델이 자체 추정을 사용하며, 이는 `meshes.focal_length`에서 보고하는 것입니다.

이 계열는 `LibreYOLO()` 팩토리가나 `libreyolo predict` CLI 명령어와 연결되어 있지 않습니다. `LibreSAM3DBody`가 유일한 진입점입니다. 소스, 스트리밍 및 결과 처리는 [prediction](/docs/predict)를 참조하십시오.

## 학습

이 작업에서 어떤 계열도 LibreYOLO 안에서 학습하지 않습니다. `LibreSAM3DBody.train()`는 다음과 같이 말합니다: 업스트림 프로젝트에서 학습하고 그 결과 체크포인트를 여기로 로드하십시오.

## 검증

메시 검증기가 없으며, `val()`가 발생합니다. 일반적인 벤치마크는 연구용 라이선스 전용이므로 포함되어 있지 않으며 제공할 수도 없습니다.

해당 메트릭 자체는 `libreyolo.validation.mesh_metrics`로 제공되며, 이미 보유하고 있는 데이터셋에 대해 평가할 수 있습니다. 이는 예측된 관절과 목표 관절, 선택적으로 예측된 정점과 목표 정점을 받아서, 검증기와 정확히 동일하게 키가 지정된 딕셔너리를 반환합니다:

`metrics/mpjpe`는 루트 관절을 정렬한 후 관절별 평균 위치 오차를 의미하며, 사람이 장면에서 어디에 서 있는지 무시하고 자세를 평가합니다. `metrics/pa_mpjpe`는 전체 Procrustes 정렬, 회전, 균일 스케일 및 이동 후의 동일한 수치로, 전역 방향과 신체 크기 오차를 제거하고 관절 자세만 남깁니다. `metrics/pve`는 정점 중심에 맞춘 후 메시 표면에서의 정점별 평균 오차이며, 관절 지표와 달리 신체 형태에 민감하며, 두 정점 배열이 모두 제공될 때만 나타납니다. 세 가지 모두 값이 낮을수록 좋습니다. 입력값은 미터 단위의 미터법으로 가정되며, `scale_to_mm`는 결과를 문헌에서 보고된 밀리미터로 변환합니다.

## 내보내기

메시 내보내기는 구현되지 않았습니다. LibreYOLO는 이 작업을 위해 내보낸 그래프 메타데이터 계약을 정의하지 않았으며, PyTorch 외부에서 MHR 매개변수 레이아웃을 전달하는 방법도 정의하지 않았기 때문에, `export()`는 출력이 해석될 수 없는 그래프를 내보내기보다는 오류를 발생시킵니다.
