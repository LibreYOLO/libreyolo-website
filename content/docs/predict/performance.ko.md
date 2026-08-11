---
title: 추론 성능
seo_title: LibreYOLO에서 더 빠른 추론
description: 'CUDA 그래프, 반정밀도, 배치, 타일드 추론 및 예측 시 테스트 시간 증강, 실제 기본값과 각 기능을 지원하는 계열과 함께.'
lead: >-
  예측 시점의 다섯 가지 제어는 처리량 또는 정확도를 변경합니다: CUDA 그래프 재실행, 정밀도, 배치 처리, 타일링, 테스트 시
  증강입니다. 각각은 특정 계열 집합에 적용되며, 그 중 두 가지는 성능을 향상시키기보다는 정확도나 지연 시간을 희생합니다.
keywords:
  - 쿠다 그래프 파이토치 추론
  - YOLO 배치 추론 파이썬
  - fp16 추론
  - 타일 추론 작은 객체
  - 큰 이미지 슬라이스 추론
  - 테스트 시간 증강 탐지
  - 그래프 캡처
  - 배치 예측 디렉토리
last_verified: 1.5.0
verification: >-
  libreyolo/models/base/inference.py. CUDA 그래프 API의 BaseModel.capture_graph,
  graph_info, release_graphs 및 cuda_graph_scope에서 InferenceRunner.__call__의 인수
  기본값; SUPPORTS_CUDA_GRAPH 클래스 변수에서의 계열 옵트인. libreyolo/utils/predict_args.py의
  NOOP_PREDICT_KWARGS, libreyolo/cli/commands/predict.py의 CLI 경고 및
  libreyolo/quant/api.py.의 CAST_RECIPES와 SUPPORTED_FAMILIES에서의 하프 프리시전 동작.
  InferenceRunner._process_in_batches 및 _predict_batch의 배칭 조건. _predict_tiled 및
  _merge_tile_detections에서의 타일링. BaseModel._predict_augment 및 _merge_tta에서 테스트 시
  증강, TTA_ENABLED, TTA_SCALES 및 TTA_FIXED_SIZE가 libreyolo/models/. 전반에서 읽힘
snippets:
  batch:
    - label: 폴더에 대한 배치 추론
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("batch_demo")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # 이를 지원하는 계열마다 4개 단위로 하나씩 앞으로 쌓습니다.
        results = model(str(folder), batch=4)
        print(len(results), "results")
    - label: '스트리밍, 그래서 목록이 결코 나타나지 않는다'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: '먼저 캡처하고, 그 다음 재생 (CUDA 필요)'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # 첫 번째 요청 시 한 번만 워밍업 비용을 지불하고 캡처하십시오.
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: 모양이 반복될 때 한 번만 캡처(쿠다 필요)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "자동"은 모양이 두 번 보일 때까지 기다리므로, 한 번만 작업하면 됩니다
        # 포획 비용을 절대 지불하지 않습니다.
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: 내보내기 추가 기능을 설치하십시오
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 기본 정밀도로 내보내고 다시 불러오기
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16 내보내기 (CUDA 머신에서 빌드하고 실행하십시오)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: 'PyTorch에서 FP16, 캐스트 레시피를 통해 (CUDA 필요)'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # 캐스트 레시피가 보정 데이터를 읽지 않습니다.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: 큰 이미지에서 타일 추론
      language: python
      code: |
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 타일링은 이미지가 입력 크기보다 클 때만 작동합니다.
        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))
        large.save("large.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model("large.jpg", tiling=True, overlap_ratio=0.2)
        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: 테스트 시 증강
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## 컨트롤과 기본값

이것들 각각은 `predict`에 대한 인수이며, 모든 기본값은 꺼져 있습니다.

| 논쟁 | 기본값 | 효과 |
|---|---|---|
| `batch` | `1` | 폴더 및 목록 소스의 순방향 처리당 이미지 수 |
| `cuda_graph` | `False` | 캡처된 CUDA 그래프에서 순방향을 다시 실행하기 |
| `tiling` | `False` | 큰 이미지를 겹치는 타일로 나누기 |
| `overlap_ratio` | `0.2` | `tiling`가 켜져 있을 때 타일 겹침 |
| `augment` | `False` | 뒤집힌 뷰를 실행하고 병합하십시오 |
| `half` | | 수용되고, 경고받고, 무시됨 |
| `device` | `None` | 예측하기 전에 모델을 이동하십시오 |

`imgsz`는 모델이 실행되는 해상도를 설정하기 때문에 비용에도 영향을 미치지만, 이는 우선 정확도에 관한 인자이며 여기보다는 모델과 함께 있어야 합니다.

## 배치 처리

<code-tabs name="batch" />

`batch`는 폴더와 목록 소스에 적용됩니다. `batch=1`를 사용하면 이미지가 각각 한 번의 순방향 패스로 실행됩니다. `1` 이상에서는 각 청크가 전처리된 후 하나의 텐서로 쌓여 한 번 실행되고, 그런 다음 다시 잘라내어 각 계열의 기존 단일 이미지 후처리가 기대하는 것을 보게 됩니다.

스택 경로는 다음 모든 조건이 충족될 때만 선택됩니다:

- `batch`는 `1`보다 큽니다
- `tiling`가 꺼져 있습니다
- 테스트 시 증강이 활성화되어 있지 않습니다
- 계열이 `SUPPORTS_BATCHED_PREDICT`를 설정합니다
- 기저 네트워크가 학습 모드에 있지 않습니다

마지막 조건은 형식적인 문제가 아닙니다. 학습 모드에 있는 네트워크는 쌓인 청크를 교차 이미지 배치 통계로 정규화하여, 같은 청크에 있는 이미지들이 서로의 예측을 바꾸게 하고, 그래서 그 실행들은 순차적으로 유지됩니다.

`SUPPORTS_BATCHED_PREDICT`는 기본값이 true입니다. 이러한 계열는 옵트아웃하며 `batch`와 관계없이 한 번의 forward당 하나의 이미지만 실행합니다: Depth Anything V2, Depth Anything 3, EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net, LibreMODUS, MiDaS, MoGe-2, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SwinIR, YOLOv1, ZipDepth, 모든 오픈 보캐뷸러리 탐지기 및 모든 비전 언어 모델.

한 가지 더 대체 방법이 있습니다. 전처리가 청크 전체에서 동일한 모양, dtype 및 장치의 균일한 `(1, C, H, W)` 텐서를 반환하지 않으면, 청크는 쌓이는 대신 순차적으로 실행되므로 정확성이 이미지가 우연히 같은 크기인지에 의존하지 않습니다.

`batch`와 `stream=True`를 큰 폴더에서 결합하여 모든 결과를 메모리에 보관하지 않고 배치 전방 전달을 얻으십시오.

## CUDA 그래프

<code-tabs name="graphs" />

CUDA 그래프는 순방향 패스를 한 번 기록하고 단일 실행으로 재생합니다. 작은 검출기는 배치 1 시간의 많은 부분을 커널 실행에 소비하므로, 이러한 실행을 통합하는 것은 처리량 증대에 도움이 되며, 재생 출력은 즉시 실행과 비트 단위로 동일합니다.

`cuda_graph`는 세 가지 값을 가집니다. `False`는 기본값이며 아무 작업도 하지 않습니다. `True`는 각 입력 모양에 대해 처음 사용할 때 캡처합니다. `"auto"`는 모양이 반복될 때까지 기다린 후 캡처하므로, 한 번만 사용되거나 모양이 변하는 작업은 캡처 비용을 부담하지 않습니다.

`capture_graph(imgsz=None, batch=1, dtype=None)`는 첫 번째 요청에서 비용이 발생하는 동작입니다. 그래프는 캡처한 정확한 형상에 대해서만 유효하므로, `batch`는 나중에 `predict`가 호출되는 방식과 일치해야 합니다.

`graph_info()`는 캡처된 그래프, 재생 횟수, 실행이 eager로 되돌아간 이유를 보고합니다. `release_graphs()`는 그것들과 그들의 정적 버퍼를 해제합니다.

캡처는 CUDA와 `SUPPORTS_CUDA_GRAPH`를 통해 참여를 선택한 계열가 필요합니다. 이는 호스트가 볼 수 없는 작업 없이 포워드를 수행해야 하고 계열별로 검증되어야 하기 때문입니다. 참여를 선택하지 않은 계열에서 이를 요청하면 조용히 즉시 실행되는 대신 `NotImplementedError`가 발생합니다.

그래프는 값을 기록하는 것이 아니라 메모리 주소를 기록하므로, 매개변수를 이동시키는 모든 것은 그것을 무효화합니다. `predict(device=...)`를 통해 장치를 변경하고, 양자화 및 역양자화를 수행하면 캡처된 모든 그래프가 무효화됩니다.

전체 가정별 지원 매트릭스, 솔기 분할 및 수치 계약은 [CUDA 그래프](/docs/reference/cuda-graphs)에 있습니다.

## 정확성

<code-tabs name="precision" />

예측 시 `half=True`는 아무 것도 하지 않습니다. 명령줄 호환성을 위해 허용되며, 아무 작업도 수행하지 않는다는 경고를 발생시키고, 어떤 계열에도 도달하기 전에 폐기됩니다. CLI의 `--half` 플래그는 `.pt` 모델에 대해 동일한 경고를 출력합니다.

정밀도를 낮추는 실제 경로는 두 가지가 있습니다.

내보내기된 아티팩트의 경우, 정밀도는 `export(format=..., half=True)`를 사용하여 내보낼 때 선택되며, 생성된 파일은 `LibreYOLO()`를 통해 변경 없이 다시 로드됩니다.

PyTorch 실행을 위해, `model.quantize(recipe="fp16")`는 모델을 float16으로 변환하고 모델의 입력과 출력에서 float32를 유지하는 훅을 설치합니다. `"bf16"`도 bfloat16으로 동일하게 수행합니다. 두 변환 모두 보정 데이터를 읽지 않으므로 `calib`는 이들에게 무시됩니다. 양자화는 현재 네 가지 계열을 다룹니다: YOLOv9, RF-DETR, BiRefNet 및 FeyNobg. CPU 장치에서의 변환은 느릴 것이라는 경고를 기록하므로, 이 레시피들은 GPU용으로 설계되었습니다.

두 경로 모두 숫자를 변경합니다. 어느 쪽도 동일한 탐지를 보장하지 않으므로 배포하기 전에 검증하십시오.

## 타일드 추론

<code-tabs name="tiling" />

타일링은 큰 이미지를 겹치는 정사각형 타일로 나누고, 각 타일에 대해 예측을 수행한 후 결과를 병합하는 방법입니다. 이는 전체 이미지를 리사이즈하면 모델이 인식할 수 있는 수준 이하로 목표가 축소되는 고해상도 이미지의 작은 객체에 적합한 옵션입니다.

타일 크기는 모델의 입력 크기이며, 지정된 경우 `imgsz`이고 정사각형이어야 합니다. `overlap_ratio`는 기본값이 `0.2`입니다. 겹치는 타일은 `iou` 임계값에서 클래스별 비최대 억제(NMS)로 조정되며, 병합된 목록은 `max_det`로 잘립니다. 이는 `iou`가 자체 NMS를 실행하지 않는 계열의 경우에도 타일 예측에 영향을 미친다는 것을 의미합니다.

이미지가 이미 맞을 때 타일링은 단순히 저렴해서가 아니라 건너뜁니다: 두 차원이 모두 입력 크기 이하라면 대신 일반 순방향 연산이 한 번 수행됩니다. 분류, 의미 분할 및 `embed` 작업에서도 타일링은 의미가 없기 때문에 단일 패스로 대신 건너뜁니다.

이는 페이로드를 다시 합칠 수 없는 작업에 대해 발생합니다: 인스턴스 분할 마스크, 회전 바운딩 박스, 점, 깊이, 엣지 및 노멀. `augment`와 결합할 수 없습니다.

결과에는 `result.tiled` 및 `result.num_tiles`가 포함됩니다. `save=True`를 사용하면 타일 실행이 모든 타일, 주석 이미지, 그리드 시각화 및 타일 크기, 겹침 및 임계값을 기록하는 `metadata.json`를 보관하는 `runs/tiled_detections` 아래에 디렉토리를 작성하며, `result.tiles_path` 및 `result.grid_path`가 이를 가리킵니다.

## 테스트 시 증강

<code-tabs name="tta" />

`augment=True`는 이미지를 한 번 이상 실행하고 `iou` 임계값에서 클래스별 비최대 억제와 함께 검출 결과를 병합합니다. 타일링과 마찬가지로, 이것은 그렇지 않으면 이를 무시하는 계열들에게 `iou`를 하중 지지로 만듭니다.

실제 사용에서는 이것이 수평 뒤집기입니다. 스케일 목록 `TTA_SCALES`는 기본적으로 `1.0` 단일 스케일로 설정되어 있으며, 배송된 어떤 계열도 이를 무시하지 않으므로 모든 계열은 두 번의 처리 과정을 거칩니다: 원본 이미지와 그 거울 이미지. `TTA_FIXED_SIZE`로 표시된 계열은 고정된 정사각형으로 크기를 변경하므로, 어떤 경우에도 다중 스케일은 이들에게는 아무 효과가 없습니다.

시맨틱 및 파노프틱 세그멘테이션은 다른 병합 방식을 취합니다. 그들의 뒤집힌 뷰는 다시 뒤집히고 두 소프트맥스 분포는 박스로 병합되는 대신 argmax 전에 평균됩니다.

테스트 시 증강은 모든 작업에 사용할 수 있는 것은 아닙니다. 이는 회전 바운딩 박스, 포즈, 포인트, 깊이, 노말, 엣지, 복원, OCR 및 임베딩 모델에 대해 발생하며 타일링과 결합할 수 없습니다.

이 계열들은 이를 아예 비활성화하기 때문에 `augment=True`는 단일 일반 과정을 실행합니다: BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet, L2CS-Net, LibreMODUS, NAFNet, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SigLIP2, SwinIR, TEED, 모든 SAM 변형, 모든 오픈-어휘 탐지기, 그리고 모든 비전 언어 모델.

## 측정

이 페이지의 어떤 내용도 지연 시간 수치를 포함하고 있지 않습니다. 하드웨어, 런타임, 정밀도 및 배치 크기 없는 1밀리초는 사실이 아니기 때문입니다. 하드웨어와 런타임을 가로지른 측정 수치는 [visionanalysis.org](https://www.visionanalysis.org)에 게시되어 있으며, `libreyolo profile`는 바로 앞에 있는 기계에서 특정 모델을 측정합니다.
