---
title: 추론 성능
seo_title: LibreYOLO에서 더 빠른 추론
description: '예측 시 CUDA 그래프, 반정밀도, 배치 처리, 타일드 추론 및 테스트 시간 증강, 실제 기본값과 각 기능을 지원하는 패밀리'
lead: >-
  다섯 가지 예측 시간 제어가 처리량 또는 정확도를 변경: CUDA 그래프 재생, 정밀도, 배치, 타일링 및 테스트 시간 증강. 각 기능은
  특정 패밀리에 적용되며, 그 중 두 가지는 성능을 절약하기보다는 정확도나 지연 시간에 영향을 줌
keywords:
  - CUDA 그래프 PyTorch 추론
  - YOLO 배치 추론 Python
  - FP16 추론
  - 소형 객체 타일드 추론
  - 대형 이미지 슬라이스 추론
  - 테스트 시간 증강 검출
  - 캡처 그래프
  - 배치 디렉토리 예측
last_verified: 1.5.0
verification: >-
  libreyolo/models/base/inference.py.의 BaseModel.capture_graph, graph_info,
  release_graphs 및 cuda_graph_scope에서 InferenceRunner.__call__의 인수 기본값을 CUDA 그래프
  API에서; SUPPORTS_CUDA_GRAPH 클래스 변수의 패밀리 옵트인. libreyolo/utils/predict_args.py의
  NOOP_PREDICT_KWARGS, libreyolo/cli/commands/predict.py의 CLI 경고,
  libreyolo/quant/api.py.의 CAST_RECIPES와 SUPPORTED_FAMILIES에서의 반정밀도 동작
  InferenceRunner._process_in_batches와 _predict_batch에서의 배칭 조건. _predict_tiled와
  _merge_tile_detections에서의 타일링. BaseModel._predict_augment와 _merge_tta에서의 테스트
  시간 증강, libreyolo/models/. 전반에서 읽는 TTA_ENABLED, TTA_SCALES 및 TTA_FIXED_SIZE
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

        # 이를 지원하는 패밀리에서 4개의 청크마다 한 번씩 스택된 포워드
        results = model(str(folder), batch=4)
        print(len(results), "results")
    - label: '스트리밍, 그래서 리스트는 절대 실체화되지 않음'
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
    - label: 먼저 캡처한 후 재생( CUDA 필요)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # 워밍업 지불 후 한 번 캡처, 첫 번째 요청에서는 제외
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: 형태가 반복될 때만 캡처( CUDA 필요)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "자동"은 형태가 두 번 보일 때까지 대기하므로 단일 작업 수행
        # 캡처 비용은 절대 지불하지 않음
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: 내보내기 추가 기능 설치
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 내보낸 후 기본 정밀도로 다시 로드
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16 내보내기( CUDA 머신에서 빌드 및 실행)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: 'CUDA 필요, 캐스트 레시피를 통한 PyTorch에서의 FP16'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # 캐스트 레시피는 보정 데이터 읽지 않음
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: 큰 이미지에서 타일 추론
      language: python
      code: |
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 이미지가 입력 크기보다 클 때만 타일링 작동
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

## 제어 및 기본값

이 모든 것은 `predict`의 인수이며, 모든 기본값은 꺼짐.

| 인수 | 기본값 | 효과 |
|---|---|---|
| `batch` | `1` | 순방향 패스당 이미지 수, 폴더 및 목록 소스용 |
| `cuda_graph` | `False` | 캡처된 CUDA 그래프에서 순방향 재생 |
| `tiling` | `False` | 큰 이미지를 겹치는 타일로 분할 |
| `overlap_ratio` | `0.2` | `tiling` 사용 시 타일 겹침 |
| `augment` | `False` | 뒤집힌 뷰를 실행하고 병합 |
| `half` | | 수락됨, 경고됨, 무시됨 |
| `device` | `None` | 예측 전에 모델 이동 |

`imgsz` 또한 비용에 영향을 주며, 모델이 실행되는 해상도를 설정하지만, 이 인수는 주로 정확도 관련이며 여기보다는 모델과 함께 속하는 것이 맞습니다.

## 배칭

<code-tabs name="batch" />

`batch`는 폴더 및 목록 소스에 적용됩니다. `batch=1`와 함께, 이미지는 각기 한 번의 순방향 패스를 실행합니다. 그 이상 `1`에서는, 각 청크가 전처리되고, 하나의 텐서로 쌓이고, 한 번 실행된 후 다시 분할되어 각 패밀리의 기존 단일 이미지 후처리가 예상하는 것을 볼 수 있습니다.

모든 조건이 충족될 때만 쌓인 경로가 사용됩니다:

- `batch`가 `1`보다 큽니다
- `tiling`가 꺼져 있습니다
- 테스트 시 증강이 활성화되어 있지 않습니다
- 패밀리가 `SUPPORTS_BATCHED_PREDICT`를 설정합니다
- 기반 네트워크가 학습 모드가 아닙니다

마지막 조건은 기술적인 사항이 아닙니다. 학습 모드인 네트워크는 쌓인 청크를 이미지 간 배치 통계로 정규화하여, 같은 청크의 이미지들이 서로의 예측을 변경할 수 있으므로, 이러한 실행은 순차적으로 유지됩니다.

`SUPPORTS_BATCHED_PREDICT` 기본값은 true입니다. 다음 계열들은 기본 설정에서 제외되며, `batch`와 관계없이 전달마다 하나의 이미지씩 실행됩니다: Depth Anything V2, Depth Anything 3, EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net, LibreMODUS, MiDaS, MoGe-2, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SwinIR, YOLOv1, ZipDepth, 모든 오픈 보캐뷸러리 탐지기, 모든 비전 언어 모델.

한 가지 더 대체 방법이 있습니다. 전처리가 덩어리 전체에서 동일한 형태, 데이터 유형 및 장치를 가진 균일한 `(1, C, H, W)` 텐서를 반환하지 않으면, 덩어리는 쌓는 대신 순차적으로 실행되므로 정확성은 이미지가 우연히 동일한 크기인지에 의존하지 않습니다.

`batch`를 `stream=True`와 큰 폴더에서 결합하면, 모든 결과를 메모리에 보관하지 않고 배치 포워드를 얻을 수 있습니다.

## CUDA 그래프

<code-tabs name="graphs" />

CUDA 그래프는 순방향 패스를 한 번 기록하고 이를 단일 실행으로 재생합니다. 작은 탐지기는 배치-1 시간의 대부분을 커널을 실행하는 데 사용하므로, 이러한 실행을 병합하면 처리량이 증가하며, 재생 출력은 즉시 실행과 비트 단위로 동일합니다.

`cuda_graph`는 세 가지 값을 가집니다. `False`는 기본값이며 아무 동작도 하지 않습니다. `True`는 각 입력 형태에 대해 처음 사용 시 캡처합니다. `"auto"`는 형태가 반복될 때까지 캡처를 기다리므로 한 번 실행되거나 형태가 다양한 작업은 캡처 비용을 지불하지 않습니다.

`capture_graph(imgsz=None, batch=1, dtype=None)`는 해당 비용을 첫 번째 요청에서 제거합니다. 그래프는 캡처된 정확한 형태에서만 유효하므로, `batch`는 이후 `predict`가 호출되는 방식과 일치해야 합니다.

`graph_info()`는 캡처된 그래프, 재생 횟수, 그리고 실행이 즉시 실행으로 돌아간 이유를 보고합니다. `release_graphs()`는 이들 및 그 정적 버퍼를 해제합니다.

캡처는 CUDA와 `SUPPORTS_CUDA_GRAPH`를 통해 옵트인한 패밀리가 필요합니다. 이는 호스트에서 볼 수 없는 작업 없이 순방향이 필요하고 패밀리별로 검증되기 때문입니다. 옵트인하지 않은 패밀리에서 요청하면, 조용히 즉시 실행되는 대신 `NotImplementedError`를 발생시킵니다.

그래프는 값을 기록하지 않고 메모리 주소를 기록하므로, 매개변수를 재배치하는 모든 작업은 이를 무효화합니다. `predict(device=...)`를 통해 장치를 변경하거나, 양자화 및 역양자화를 수행하면 캡처된 그래프가 모두 무효화됩니다.

전체 패밀리별 지원 매트릭스, 솔기 분할 및 수치 계약은 [CUDA 그래프](/docs/reference/cuda-graphs)에서 확인할 수 있습니다.

## 정밀도

<code-tabs name="precision" />

`half=True`는 예측 시 아무 작업도 하지 않습니다. 명령행 호환성을 위해 허용되며, 실행하지 않는다는 경고를 발생시키고, 패밀리에 도달하기 전에 폐기됩니다. CLI의 `--half` 플래그는 `.pt` 모델에 대해 동일한 경고를 출력합니다.

정밀도를 낮추는 두 가지 실제 경로가 있습니다.

내보낸 아티팩트의 경우, 정밀도는 `export(format=..., half=True)`를 사용하여 내보낼 때 선택되며, 결과 파일은 `LibreYOLO()`를 통해 변경 없이 다시 로드됩니다.

PyTorch 실행의 경우, `model.quantize(recipe="fp16")`는 모델을 float16으로 캐스팅하고 모델의 입력과 출력에서 float32를 유지하는 훅을 설치합니다. `"bf16"`는 bfloat16으로 동일하게 수행합니다. 어느 캐스트도 보정 데이터를 읽지 않으므로 `calib`는 무시됩니다. 현재 양자화는 네 가지 계열을 다룹니다: YOLOv9, RF-DETR, BiRefNet 및 FeyNobg. CPU 장치에서의 캐스트는 느릴 것이라는 경고를 기록하므로, 이러한 방법은 GPU용으로 설계되었습니다.

두 경로 모두 수치를 변경합니다. 어느 것도 동일한 탐지를 보장하지 않으므로 배포 전에 검증해야 합니다.

## 타일드 추론

<code-tabs name="tiling" />

타일링은 큰 이미지를 겹치는 정사각형 타일로 자르고, 각 타일에 대해 예측을 수행한 후 결과를 병합합니다. 이는 전체 이미지를 리사이즈하면 모델이 해결할 수 있는 수준보다 목표물이 작아지는 고해상도 이미지에서 작은 객체를 위한 옵션입니다.

타일 크기는 모델의 입력 크기이거나 `imgsz`가 주어진 경우 해당 크기이며, 정사각형이어야 합니다. `overlap_ratio`는 기본값으로 `0.2`를 사용합니다. 겹치는 타일은 `iou` 임계값에서 클래스별 비최대 억제(NMS)로 조정되며, 병합된 목록은 `max_det`로 잘립니다. 이는 `iou`가 자체적으로 NMS를 실행하지 않는 클래스의 경우에도 타일 예측에 영향을 미침을 의미합니다.

이미 이미지가 맞는 경우에는 단순히 값이 싸서가 아니라 타일링을 건너뜁니다. 만약 두 차원이 모두 입력 크기 이하이면, 대신 일반적인 한 번의 순방향 실행을 합니다. 또한 분류, 시맨틱 분할 및 `embed` 작업에서도 타일링이 의미가 없기 때문에 단일 패스로 대신합니다.

payload를 다시 합칠 수 없는 작업에서는 수행됩니다: 인스턴스 분할 마스크, 방향 박스, 포인트, 깊이, 에지, 노멀. `augment`와 결합할 수 없습니다.

결과에는 `result.tiled`와 `result.num_tiles`가 포함됩니다. `save=True`의 경우, 타일 실행은 `runs/tiled_detections` 하위에 모든 타일, 주석 이미지, 그리드 시각화, 타일 크기, 겹침 및 임계값을 기록하는 `metadata.json`를 포함한 디렉토리를 작성하며, `result.tiles_path`와 `result.grid_path`가 이를 가리킵니다.

## 테스트 시 증강

<code-tabs name="tta" />

`augment=True`은 이미지를 한 번 이상 실행하고, 클래스별 비최대 억제(per-class non-maximum suppression)를 `iou` 임계값에서 검출 결과와 병합합니다. 타일링과 마찬가지로, 이는 그렇지 않으면 무시되는 패밀리에 대해 `iou`를 지지하게 만듭니다.

실제로 이는 수평 뒤집기(horizontal flipping)입니다. 스케일 리스트 `TTA_SCALES`는 기본적으로 `1.0` 단일 스케일로 설정되며, 제공된 패밀리 중 어느 것도 이를 override하지 않으므로 모든 패밀리는 두 번의 실행을 합니다: 원본 이미지와 그 미러 이미지. `TTA_FIXED_SIZE`로 표시된 패밀리는 고정 정사각형으로 크기를 조정하므로, 어떤 경우에도 멀티 스케일은 그들에게는 무의미합니다.

의미적(semanctic) 및 전체적(panoptic) 세분화(segmentation)는 다른 병합 방식을 사용합니다. 그들의 뒤집힌 뷰는 다시 뒤집히며, 두 소프트맥스 분포는 argmax 전에 평균화되고, 박스로 병합되는 방식은 아닙니다.

테스트 시 증강은 모든 작업에서 사용할 수 없습니다. 이는 방향이 지정된 상자, 포즈, 점, 깊이, 법선, 가장자리, 복원, OCR 및 임베딩 모델에서 발생하며, 타일링과 결합할 수 없습니다.

이러한 계열은 이를 완전히 비활성화하므로 `augment=True`는 단일 일반 패스만 실행합니다: BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet, L2CS-Net, LibreMODUS, NAFNet, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SigLIP2, SwinIR, TEED, 모든 SAM 변형, 모든 오픈보캐뷸러리 디텍터 및 모든 비전 언어 모델.

## 측정

이 페이지에는 지연 시간이 표시되지 않습니다. 하드웨어, 런타임, 정밀도 및 배치 크기 없이의 1밀리초는 사실이 아니기 때문입니다. 하드웨어와 런타임 전반에서 측정된 수치는 [visionanalysis.org](https://www.visionanalysis.org)에 게시되어 있으며, `libreyolo profile`는 사용자가 있는 기기에서 특정 모델을 측정합니다.
