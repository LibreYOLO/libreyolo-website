---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: LibreYOLO의 초소형 객체 탐지'
description: >-
  LibreYOLO에서 Dome-DETR로 항공 및 드론 이미지의 초소형 객체를 탐지합니다. 업스트림 가중치를 변환하고 MIT 라이선스 코드로
  예측, 파인튜닝, 검증합니다.
lead: >-
  D-FINE 기반 초소형 객체 전문 모델입니다. 밀도 헤드가 객체 위치를 결정하고 인코더 어텐션은 객체가 있는 창으로 제한되며 쿼리 수는
  고정되지 않고 해당 밀도에서 결정됩니다. LibreYOLO는 객체 탐지를 지원합니다.
keywords:
  - Dome-DETR 사용법
  - 초소형 객체 탐지
  - 소형 객체 탐지
  - 항공 이미지
  - 드론 탐지
  - 원격 탐사
  - VisDrone
  - AI-TOD
  - DETR
  - 밀도 적응 쿼리
last_verified: 1.5.0
snippets:
  predict:
    - label: 변환 후 예측
      language: bash
      code: |
        # LibreYOLO는 Dome-DETR 가중치를 호스팅하지 않으므로 업스트림 저장소에서
        # 체크포인트를 가져와 한 번 변환합니다.
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 로컬 경로이며 단순 이름이 아닙니다. 이 계열은 아무것도 다운로드하지 않습니다.
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        result = model("drone-frame.jpg", save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt
        source=drone-frame.jpg save=True
    - label: 클래스 이름
      language: python
      code: |
        from libreyolo import LibreYOLO

        # COCO 체크포인트가 없으므로 클래스는 가중치를 학습한 데이터셋에서 가져오고
        # 체크포인트 메타데이터에서 읽습니다.
        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")
        print(aitod.model.names)     # AI-TOD-V2 클래스 9개

        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        print(visdrone.model.names)  # VisDrone 클래스 12개
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: 다중 GPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
source_hash: 381f01d769e7c420
---

## 설치

Dome-DETR에는 선택적 extra가 필요하지 않습니다. 가져오는 모든 항목이 기본 설치에 포함됩니다.

```bash
pip install libreyolo
```

## 예측

자동으로 다운로드할 항목은 없습니다. LibreYOLO는 이러한 가중치를 호스팅하지 않으므로 업스트림 체크포인트를 가져와 한 번 변환한 다음 변환된 파일을 경로로 불러옵니다. 그 이유는 [라이선스](#licensing)에 나와 있습니다.

<code-tabs name="predict" />

반환되는 `Results` 객체는 모든 계열이 반환하는 것과 같으므로 탐지기를 바꾸려면 한 줄만 변경하면 됩니다. `conf`와 `max_det`은 쿼리 선택을 필터링합니다. 디코더가 NMS 단계 없는 집합 예측기이므로 `iou`는 API 일관성을 위해 허용되지만 효과가 없습니다. 소스, 스트리밍, 결과 처리는 [예측](/docs/predict)을 참조합니다.

이 계열에서는 두 기능을 끕니다. PAQI의 쿼리 수가 데이터에 따라 달라져 순전파 형태도 이미지마다 달라지며 그래프 캡처가 이를 수용할 수 없으므로 CUDA 그래프 캡처를 비활성화합니다. 테스트 시간 증강은 하나의 고정 정사각형 크기에서 실행되므로 다중 스케일 TTA 요청은 아무 작업도 하지 않습니다.

## 변형

s, m, l의 세 크기는 모두 800x800에서 실행됩니다. 크기는 백본을 선택하고 가중치가 나온 데이터셋은 디코더 깊이와 쿼리 예산을 선택하므로 크기 코드만으로 그래프를 식별할 수 없습니다. AI-TOD-V2 가중치는 이미지마다 300~1500개 쿼리를 선택하고 VisDrone 가중치는 250~500개를 선택합니다. large 모델은 AI-TOD-V2에서 디코더 계층 네 개, VisDrone에서 여섯 개를 실행합니다.

Dome-DETR은 D-FINE에 세 기능을 추가합니다. DeFE는 밀도 맵을 예측합니다. MWAS는 어디든 어텐션하는 대신 해당 맵을 사용해 실제로 객체가 있는 창으로 인코더 어텐션을 제한합니다. PAQI는 고정된 300개를 디코딩하는 대신 같은 밀도에서 쿼리 집합 크기를 정합니다. 이점은 객체가 가장 작을 때 집중되고 크기가 커지면 줄어듭니다. 업스트림 자체 ablation에서 매우 작은 객체의 AP는 14.0에서 17.8로 높아지지만 중간 객체 AP는 45.4에서 46.4로만 높아집니다. 이를 D-FINE의 대체품이 아니라 항공, 드론, 원격 탐사 이미지용 [D-FINE](/docs/models/d-fine) 동반 모델로 취급합니다.

LibreYOLO는 이 계열의 체크포인트를 게시하지 않아 벤치마크할 체크포인트도 없으므로 벤치마크 행을 게시하지 않습니다.

## 학습

Dome-DETR은 학습할 수 있습니다. 학습은 D-FINE 손실에 DeFE 밀도 및 개수 지도를 더한 업스트림 전체 목적 함수를 실행합니다. 패딩된 쿼리는 분류 항에서 마스킹하고 이미지별 노이즈 제거 어텐션 마스크를 사용하여 한 이미지의 패딩이 다른 이미지로 누출되지 않게 합니다.

<code-tabs name="train" />

구성은 D-FINE 레시피를 상속하고 MWAS 요구 사항에 맞게 변경합니다. `imgsz`는 800, `lr0`는 `2e-4`, 백본 매개변수 그룹은 `backbone_lr_mult=0.1`로 조정합니다. MWAS 창에서 입력이 stride 8로 계속 나누어떨어져야 하므로 `multi_scale`을 강제로 끕니다. `batch` 기본값은 D-FINE의 16이 아니라 4입니다. PAQI가 각 배치를 가장 넓은 구성원에 맞춰 패딩하므로 메모리는 평균 이미지가 아니라 배치에서 가장 복잡한 이미지를 따릅니다.

정확도에 관한 중요한 제한이 하나 있습니다. 업스트림은 `MultiStepLR(milestones=[80, 120], gamma=0.8)`로 160 epoch 동안 학습하지만 여기의 기본값은 같은 160 epoch에 D-FINE의 flat-cosine 일정을 실행합니다. 해당 일정은 여기서 재현되지 않았고 논문의 AP 수치도 재현되지 않았으므로 이 레시피가 그 수치에 도달한다는 보장이 아니라 업스트림 저자의 결과로 읽어야 합니다. 논문과 일치시키려면 업스트림 일정을 제공합니다.

데이터셋, 증강, 다중 GPU, 로거는 [학습](/docs/train)을 참조합니다.

## 검증

`val()`은 지표 이름으로 키가 지정된 사전을 반환하며 `verbose`가 활성화되어 있으면 클래스별 결과를 출력합니다.

<code-tabs name="val" />

이 계열에는 측정할 COCO 체크포인트가 없으므로 라이브러리의 COCO 검증 게이트가 적용되지 않습니다. 학습에 사용한 형식의 자체 데이터셋으로 검증합니다.

## 내보내기

어떤 형식도 내보내기를 지원하지 않으며 요청하면 파일을 만들지 않고 예외를 발생시킵니다.

이유는 PAQI입니다. 밀도로 필터링한 제안과 탐욕적 밀도 적응 억제 루프에서 이미지별 쿼리 수를 정하므로 디코더 출력 길이는 그래프가 아닌 입력의 속성입니다. 추적하면 추적 이미지가 우연히 생성한 개수를 고정하여 다른 모든 이미지에 잘못된 결과를 조용히 반환하는 아티팩트가 만들어집니다. 정적 구현은 250~1500개 후보 전체에 억제를 펼쳐야 하고 고정 top-k로 축소하면 이 계열의 존재 이유인 초소형 객체 재현율이 사라집니다. 내보낼 수 있는 detection transformer가 필요하면 [D-FINE](/docs/models/d-fine)을 사용합니다.

## 체크포인트

나열할 체크포인트가 없습니다. LibreYOLO는 Dome-DETR 가중치를 게시하지 않으며 `LibreDOMEDETR<size>-<dataset>.pt` 형식의 어떤 이름도 다운로드로 해석되지 않습니다.

업스트림은 AI-TOD-V2와 VisDrone의 두 데이터셋마다 s, m, l의 체크포인트를 제공하여 총 여섯 개입니다. AI-TOD-V2는 클래스 9개, VisDrone은 12개입니다. COCO 체크포인트가 없으므로 표준 파일명에는 항상 데이터셋 접미사가 들어가고 클래스 이름은 계열 상수가 아니라 체크포인트 메타데이터에 담깁니다. 단순 `LibreDOMEDETRs.pt`를 요청하면 404가 발생할 다운로드를 시도하지 않고 실제 두 파일명과 변환 명령을 알려주는 메시지와 함께 즉시 예외를 발생시킵니다.

`weights/convert_domedetr_weights.py`가 변환을 수행합니다. LibreYOLO 그래프를 다시 구축하고 업스트림 텐서를 불러온 뒤 누락되거나 예상 밖이거나 형태가 잘못된 키가 하나라도 있으면 파일 쓰기를 거부하므로 변환 파일은 정확히 일치하거나 아예 존재하지 않습니다. 업스트림 `.pth`와 크기 및 변형을 전달합니다.

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

수치 충실도의 경우 `weights/parity_domedetr.py`는 여섯 체크포인트 모두에서 이 이식과 업스트림 구현을 비교합니다. 먼저 MWAS 창 마스크를 비트 단위로 검사한 뒤 `pred_logits`와 `pred_boxes`에서 모두 `max_abs_diff == 0.0`을 보고하고 모든 손실 항을 업스트림 criterion과 별도로 비교합니다. 이는 업스트림 체크아웃과 공개 체크포인트가 디스크에 있어야 하고 수동으로 실행하는 스크립트입니다. 지속적 통합의 일부가 아니며 어떤 CI 작업도 이를 재현하지 않습니다.

## 라이선스

<provenance-box>

이 계열을 미러링하지 않는 이유는 가중치입니다. 업스트림 모델 카드 메타데이터에는 라이선스 필드가 없고 본문은 프로젝트가 Apache-2.0이라고 하면서 자료를 학술 연구 용도로만 제한합니다. 두 해석은 일치하지 않으며 더 엄격한 쪽은 재배포 허가가 아닙니다. 따라서 LibreYOLO는 명확해질 때까지 파일을 복사하지 않고 업스트림 저장소를 연결합니다. 여기서 [YOLO-NAS](/docs/models/yolo-nas)에 적용하는 것과 같은 논리입니다.

코드는 별도이며 더 명확합니다. 업스트림 저장소는 Apache-2.0이고 LibreYOLO 이식은 MIT이며 자체 데이터로 직접 학습한 가중치는 소유자에게 귀속됩니다.

</provenance-box>

## 인용

Dome-DETR은 ACM Multimedia 2025에서 "Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection"이라는 제목으로 발표되었습니다. 프리프린트는 [arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741)에 있습니다. 저자는 저장소에 BibTeX 블록을 게시하지 않았으므로 직접 조합하지 않고 여기에도 재현하지 않습니다.

<citation-block />
