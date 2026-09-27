---
title: 모델 API
seo_title: LibreYOLO 모델 객체 메서드 및 시그니처
description: >-
  로드된 LibreYOLO 모델의 모든 메서드: predict, embed, track, val, train, export, save,
  quantize, info 및 CUDA 그래프 제어, 실제 기본값 포함.
lead: >-
  로드된 LibreYOLO 모델은 BaseModel의 인스턴스입니다. 이 페이지는 libreyolo/models/base/model.py.에서
  읽은 시그니처와 기본값과 함께 해당 인스턴스가 가지는 메서드를 나열합니다.
keywords:
  - libreyolo 모델 메서드
  - libreyolo 예측 인자
  - libreyolo val 인수
  - libreyolo 내보내기 인수
  - 모델.트랙
  - 모델.양자화
  - 그래프 캡처
last_verified: 1.5.0
verification: >-
  서명과 기본값은 v1.5.0에서 libreyolo/models/base/model.py와
  libreyolo/models/base/inference.py에서 읽어옵니다. 계열 클래스는 이를 좁히거나 확장할 수 있습니다;
  train()은 계열별로 정의되며 여기에는 공유 cfg= 래퍼만 문서화되어 있습니다.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True는 프레임이나 이미지마다 하나의 Results를 반환하는 제너레이터를 반환합니다.
        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## 건설

팩토리는 계열 클래스 인스턴스를 반환합니다. 해당 클래스를 직접 생성할 때는 같은 인자를 사용하지만, `size`가 필요합니다:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"`는 사용 가능한 경우 CUDA를 선택하고, 그 다음 MPS를 선택하며, 그 다음 CPU를 선택합니다. 정수나 숫자 문자열은 CUDA 순서 인덱스로 읽히므로, `device=0`와 `device="0"`는 모두 `cuda:0`를 의미합니다. `task`는 해당 계열의 `SUPPORTED_TASKS`에 대해 검증됩니다. `model_path=None`를 전달하면 아키텍처가 구축되고 학습 모드로 남게 됩니다; `dict`를 전달하면 해당 상태 사전을 직접 로드합니다.

## 예측하고 \_\_call\_\_

`predict`는 `__call__`의 별칭입니다.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| 논쟁 | 기본값 | 의미 |
|---|---|---|
| `source` | `None` | 메모리 내 이미지의 이미지, 리스트 또는 튜플, 디렉토리, 비디오 파일 또는 `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"`와 같은 화면 소스 |
| `conf` | `0.25` | 신뢰도 임계값 |
| `iou` | `0.45` | NMS를 위한 IoU 임계값 |
| `imgsz` | `None` | 입력 크기 재정의; `None`는 모델의 기본 크기를 사용합니다 |
| `device` | `None` | 이 통화를 위한 장치 무시 |
| `classes` | `None` | 이 클래스 ID만 유지하십시오 |
| `max_det` | `300` | 이미지당 최대 탐지 수 |
| `augment` | `False` | 테스트 시 증강 |
| `save` | `False` | 주석이 달린 이미지나 비디오를 작성하십시오 |
| `batch` | `1` | 디렉토리 및 목록 소스에 대한 전달당 이미지 수 |
| `stream` | `False` | 실제화된 목록 대신 제너레이터를 반환하십시오 |
| `stream_buffer` | `False` | 최신 것만이 아니라 캡처된 모든 라이브 프레임을 저장하십시오 |
| `vid_stride` | `1` | 매 N번째 비디오 또는 화면 프레임을 처리합니다 |
| `show` | `False` | 주석이 달린 프레임을 창에 표시 |
| `output_path` | `None` | `save=True`일 때 출력 경로 |
| `color_format` | `"auto"` | 메모리 내 배열을 위한 색상 형식 힌트 |
| `tiling` | `False` | 대형 이미지에 대한 타일 추론 |
| `overlap_ratio` | `0.2` | 타일 겹침 비율 |
| `output_file_format` | `None` | `"jpg"`, `"png"` 또는 `"webp"` |
| `cuda_graph` | `False` | `True`는 입력 형태마다 처음 사용 시 캡처하고, `"auto"`는 형태가 반복될 때까지 기다립니다 |

단일 이미지 소스는 하나의 `Results`를 반환합니다. 리스트, 튜플 또는 디렉토리는 그것들의 리스트를 반환하며, `stream=True`는 모든 경우에 제너레이터를 반환합니다.

라이브 스트림 소스는 무한하며 `stream=True`가 필요합니다. `tiling`와 `augment`는 결합할 수 없습니다. 테스트 시 증강은 `embed`, `point` 및 `edge` 작업에 대해 증가시킵니다.

<code-tabs name="usage" />

`batch > 1`을 사용하면 `SUPPORTS_BATCHED_PREDICT`이 참인 계열은 각 청크마다 한 번 앞으로 쌓아서 실행합니다; `batch=1`은 이미지당 한 번 앞으로 실행합니다.

<code-tabs name="stream" />

## 삽입

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

모든 임베딩 행을 단일 `(N_total, D)` 텐서로 쌓는 `predict`에 대한 편리한 래퍼입니다. 모델은 `task="embed"`로 구성되어야 하며, 그렇지 않으면 `NotImplementedError`를 발생시킵니다.

## 트랙

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

`track_id`가 설정된 경우 프레임당 하나의 `Results`를 생성합니다. `tracker`는 `"bytetrack"`, `"botsort"`, `"ocsort"` 또는 `"deepocsort"`이며, 구성 유형이 트래커를 선택하기 때문에 `tracker_config`가 주어지면 무시됩니다. `track_conf`는 ByteTrack 및 BoT-SORT의 경우 `track_high_thresh`에 매핑되고, OC-SORT 및 Deep OC-SORT의 경우 `det_thresh`에 매핑됩니다. `output_path`는 기본값이 `runs/track/<video_stem>.mp4`입니다.

## 값

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

작업에 따라 키가 달라지는 메트릭 사전을 반환합니다; 탐지는 `metrics/precision`, `metrics/recall`, `metrics/mAP50` 및 `metrics/mAP50-95`를 반환합니다. `imgsz`는 정사각형 정수 또는 `(height, width)` 튜플을 허용하며 모델의 기본 입력 크기를 기본값으로 사용합니다. `plots`는 `save_plots`의 별칭입니다. `allow_download_scripts`는 데이터셋 YAML이 `download` 필드에 포함할 수 있는 내장 Python을 제어합니다.

`faster_coco_eval`는 `**kwargs`를 통해 허용되며 기본값은 `True`이며, 패키지가 설치되지 않은 경우 pycocotools로 대체됩니다. 실행된 백엔드는 `model.last_eval_backend`에 보고됩니다.

`obb` 및 `pose` 작업에 대한 증강 검증 인상이 있습니다.

## 학습

`train`는 계열별로 정의되므로 그 인자가 다릅니다. 두 가지 동작은 공유되는데, 이는 기본 클래스가 모든 계열의 `train`를 감싸기 때문입니다:

- `cfg=`는 키가 호출에 병합되는 YAML 경로를 가져옵니다. 명시적 키워드 인수가 파일보다 우선합니다.
- `pretrained=False`가 `g0` 또는 `g1` 보장 그룹의 계열에 대해 모델을 학습하기 전에 처음부터 다시 초기화하며, `resume=True`와 결합할 수 없습니다.

계열마다 실제로 어떤 증강 옵션을 존중하는지는 계열별로 다르므로, [증강 행렬](/docs/reference/augmentation-matrix)을 참조하십시오.

## 내보내기

```python
model.export(format="onnx", **kwargs) -> str
```

작성된 아티팩트의 경로를 반환합니다. `format`는 익스포터 레지스트리를 통해 해결되며, 여기서 `engine`는 `tensorrt`의 별칭이고 `litert`는 `tflite`의 별칭입니다. 모든 익스포터가 공유하는 인수:

| 논쟁 | 기본값 | 의미 |
|---|---|---|
| `output_path` | `None` | 출력 파일 경로; 생략하면 `weights/` 아래에 생성됨 |
| `imgsz` | `None` | `(height, width)` 튜플 또는 단일 정수; 기본값은 기본 크기 |
| `opset` | `None` | ONNX 연산셋 버전 |
| `simplify` | `True` | ONNX 그래프 단순화 실행 |
| `dynamic` | `True` | 동적 축 사용 |
| `half` | `False` | FP16 정밀도 |
| `int8` | `False` | INT8 정밀도 |
| `batch` | `1` | 배치 크기가 아티팩트에 내장됨 |
| `device` | `None` | 따라 그릴 장치 |
| `data` | `None` | INT8 보정을 위한 data.yaml |
| `fraction` | `1.0` | 사용할 보정 데이터셋의 비율 |
| `allow_download_scripts` | `False` | 데이터셋 YAML 다운로드에서 내장 Python 허용 |
| `verbose` | `False` | 상세 내보내기 로깅 |

차단된 조합은 추적 전에 프리플라이트에서 `NotImplementedError`를 발생시킵니다. 적용 범위와 규칙은 [내보내기 매트릭스](/docs/reference/export-matrix) 페이지에 있습니다. 실시간 LoRA 어댑터가 있을 경우, 그것들은 밀집 가중치에 통합되며, 이 병합은 모든 요청이 거부된 후에만 발생합니다.

## 저장

```python
model.save(path) -> str
```

스키마 v1.0 LibreYOLO 체크포인트를 작성합니다: 상태 딕셔너리와 [체크포인트 스키마](/docs/reference/checkpoint-schema)에 설명된 메타데이터를 포함합니다. 양자화된 모델은 추가로 `quant` 매니페스트를 가지고 있어, `LibreYOLO(path)`가 양자화된 구조와 스케일을 복원합니다.

## 양자화, 양자 정보 및 역양자화

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

제자리에서 양자화하고 모델을 반환합니다. `recipe`는 `fp16` 및 `bf16` 중 하나의 캐스트, Conv 및 Linear 레시피 `int8` 및 `fp8`, 또는 Linear 전용 레시피 `w4a16`, `w4a8`, `nvfp4`, `mxfp4` 및 `int2` 중 하나이며, RF-DETR과 같은 트랜스포머 계열가 지원합니다. `int2`는 QAT가 필요합니다. `calib`는 data.yaml 경로 또는 내장 데이터셋 이름을 사용하며 이미지를 순방향Only로 읽습니다; 레이블은 절대 읽지 않습니다. 보정 생략을 위해 `calib=None`를 전달합니다. `algorithm`는 `"minmax"`, `"percentile"` 또는 `"auto"`입니다.

`model.quant_info()`는 양자화 상태 요약을 반환하며, float 모델의 경우 `None`를 사용합니다. `model.dequantize()`는 양자화 학습된 마스터 가중치를 유지하면서 float 모듈을 제자리에서 복원하며, 이는 QAT에서 `export(format="onnx", int8=True, data=...)`로 가는 다리 역할을 합니다.

## 정보와 레이어

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info`는 JSON 친화적인 딕셔너리를 반환하고 `verbose`가 true일 때 사람이 읽을 수 있는 요약을 기록합니다. `get_available_layer_names`는 증류 또는 특징 추출 구성에서 이름을 지정할 수 있는 레이어를 나열합니다.

## CUDA 그래프

`SUPPORTS_CUDA_GRAPH` 클래스 속성이 true인 계열에서 사용할 수 있습니다. 재생은 즉시 실행과 비트 단위로 동일합니다.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # 컨텍스트 관리자
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

캡처된 그래프는 캡처된 정확한 형태에 대해서만 유효하므로 `batch`와 `imgsz`은 이후 `predict` 호출과 일치해야 합니다. `capture_graph`는 첫 번째 요청에서 캡처 비용을 이동합니다. `mode`는 `True` 또는 `"on"`를 사용하여 처음 사용할 때 캡처하도록 허용하고, `"auto"`를 사용하여 형태가 반복될 때까지 기다리며, `False`를 사용하여 아무 작업도 하지 않도록 합니다. `capture_graph`는 family가 선택하지 않은 경우 `NotImplementedError`를 발생시키고, 캡처에 실패하면 `CudaGraphUnavailable`를 발생시킵니다.

## 장치 및 데이터 유형

`Results` 객체는 `.to()`, `.cpu()`, `.cuda()` 및 `.numpy()`를 포함합니다; [Results types](/docs/reference/results-types)를 참조하십시오. 모델 자체는 `device=`를 `predict`에 전달하거나 생성 시 이동됩니다.
