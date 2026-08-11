---
title: NVIDIA DeepStream
seo_title: NVIDIA DeepStream에서 YOLO 모델 실행하기
description: >-
  NVIDIA DeepStream용으로 LibreYOLO 모델을 내보냅니다: ONNX 그래프와 생성된 nvinfer 설정 파일. 파서 빌드와
  파이프라인에 필요한 정확한 명령.
lead: >-
  NVIDIA DeepStream은 nvinfer 엘리먼트로 추론을 실행하며, 여기에는 ONNX 그래프, 그에 맞는 설정 파일, 바운딩 박스
  파서가 필요합니다. ONNX 내보내기에서 deepstream=True를 설정하면 앞의 두 가지가 생성되고 세 번째와 연결됩니다.
keywords:
  - deepstream yolo 연동
  - nvinfer 설정 파일
  - deepstream 바운딩 박스 파서
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app 실행
  - tensorrt 엔진 빌드
  - jetson deepstream
  - yolo onnx deepstream 내보내기
meta:
  - label: 플래그
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: 생성 파일
    value: 'ONNX 그래프, config_infer_primary_<stem>.txt, <stem>_labels.txt'
  - label: 지원 범위
    value: 아홉 가지 작업에 걸친 43개 계열과 작업 조합
  - label: 파서
    value: >-
      Marcos Luciano가 만든 MIT 라이선스 프로젝트 DeepStream-Yolo의 NvDsInferParseYolo. 장치마다
      한 번씩 빌드합니다.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: 제공 버전
    value: v1.5.0에 포함됩니다. 2026-08-08에 풀 리퀘스트 728로 dev에 병합되었습니다.
    links:
      - label: 풀 리퀘스트 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: 이슈 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: 런타임 검증
    value: 'RTX 5070 Ti에서 DeepStream 8.0.0, 탐지만, 2026-08-08'
verification: >-
  2026-08-08 런타임 검증을 바탕으로 작성했습니다. 계열 목록, 설정 키, 기본값은 커밋 5f81e11e의
  libreyolo/export/deepstream.py와 libreyolo/export/exporter.py에서 확인했으며, 이 커밋은 같은
  날 풀 리퀘스트 728로 dev에 병합되었습니다.
snippets:
  install:
    - label: 설치
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt,

        # libreyolo9s_labels.txt를 작업 디렉터리에 생성합니다.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # 탐지 모델은 각각 별도 디렉터리에 둡니다: 모든 탐지 설정이 같은 엔진

        # 캐시 파일 이름을 지정합니다. "알려진 함정"을 참고하십시오.

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: 인자
      language: python
      code: |
        model.export(
            format="onnx",     # 다른 모든 형식에서는 deepstream=True가 거부됩니다
            deepstream=True,
            conf=0.25,         # pre-cluster-threshold의 초깃값 (해당 작업에서는
                               # classifier-threshold, segmentation-threshold에도)
            iou=0.45,          # nms-iou-threshold의 초깃값, cluster-mode=4에서는 생략
            batch=1,           # batch-size와 엔진 캐시 파일 이름의 초깃값
            half=False,        # True면 설정에 network-mode=2 (fp16 빌드)로 표시됩니다
            int8=False,        # True면 설정에 network-mode=1로 표시됩니다
            dynamic=True,      # ONNX 그래프의 동적 배치 축
            imgsz=640,         # infer-dims=3;H;W의 초깃값
        )

        # deepstream=True와 nms=True는 함께 쓸 수 없습니다: DeepStream이 클러스터링
        # 단계에서 억제를 수행하므로 그래프에는 아무것도 포함되지 않습니다.
    - label: D-FINE 가중치 먼저 내려받기
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: 무엇보다 먼저 GPU 패스스루 확인
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, DeepStream 컨테이너 안에서 실행'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # 이 이미지의 /usr/local/cuda-12는 스텁이라서 빌드가

        # "fatal error: crt/host_defines.h: No such file or directory"로 실패합니다.

        # 헤더가 실제로 들어 있는 툴킷을 찾습니다. 8.0 이미지에서는 cuda-12.5입니다.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # 이미지에는 libcublas.so.12와 libcublas.so.12.8.4.1이 있지만 -lcublas가

        # 필요로 하는 버전 없는 libcublas.so는 없어서, 링크 단계가

        # "/usr/bin/ld: cannot find -lcublas"로 실패합니다. 링커가 원하는 이름을 만들어 줍니다.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: 인스턴스 분할은 다른 파서를 사용합니다
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: 실행하기
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: 두 단계를 한 컨테이너에서
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## 제공 버전

DeepStream 내보내기는 v1.5.0에 포함됩니다. 2026-08-08에 풀 리퀘스트 728로 `dev`에
병합되었으므로, 최신 설치본에는 이미 들어 있고 브랜치를 고정할 필요가 없습니다.

<code-tabs name="install" />

2026-08-08 이전에 `deepstream-export` 브랜치를 클론했다면 교체하십시오. 그 브랜치는
리베이스 후 강제 푸시되었고, 이전 히스토리에는 이 내보내기가 CUDA 장치에서 실행되게
하는 수정이 빠져 있습니다.

## 내보내기가 생성하는 파일

`model.export(format="onnx", deepstream=True)`는 파일 세 개를 나란히 생성합니다.
`libreyolo9s.pt`의 경우:

- `libreyolo9s.onnx`는 탐지 그래프로, 형태가 `(batch, num_detections, 6)`인 출력
  텐서 하나를 내보내며 각 행은 네트워크 입력 픽셀 좌표의
  `[x1, y1, x2, y2, score, class_id]`입니다.
- `config_infer_primary_libreyolo9s.txt`는 계열의 전처리 상수, 클래스 개수, 임계값,
  파서 연결을 담은 `nvinfer` 설정입니다.
- `libreyolo9s_labels.txt`는 한 줄당 클래스 이름 하나입니다.

체크포인트에 클래스 이름이 있으면 레이블 파일이 함께 생성됩니다. 깊이 추정 모델에는
클래스 이름이 없으므로 파일도 `labelfile-path` 키도 생기지 않습니다.

LibreYOLO는 `.so`를 만들지 않습니다. DeepStream이 불러오는 `.so`는
`marcoslucianops/DeepStream-Yolo`의 바운딩 박스 파서이며, 장치마다 한 번 컴파일하고,
어떤 LibreYOLO 탐지 모델을 연결하든 같은 바이너리입니다. 모델은 ONNX 쪽입니다.
분류와 시맨틱 분할에는 파서가 전혀 필요하지 않은데, `nvinfer`가 후처리를 직접 하기
때문입니다.

## 모델 내보내기

<code-tabs name="export" />

`LibreDFINE._load_weights`는 파일이 디스크에 없으면 내려받기를 시도하지 않고
`FileNotFoundError`를 발생시키므로, `LibreDFINEs.pt`는 먼저 직접 받아야 합니다. 이
공백은 [이슈 #727](https://github.com/LibreYOLO/libreyolo/issues/727)로 추적하고
있습니다. YOLO9 가중치는 처음 사용할 때 내려받습니다.

이 플래그는 Python 전용입니다. 이 브랜치의 `libreyolo export`에는 `deepstream`
옵션이 없고, CLI는 알 수 없는 키를 그대로 넘기는 대신 고정된 목록에서 내보내기
인자를 만듭니다.

## 바운딩 박스 파서 빌드

탐지에는 파서 라이브러리가 필요하고, 인스턴스 분할에는 다른 라이브러리가 필요하며,
나머지 작업에는 필요하지 않습니다. DeepStream 8.0 이미지에서는 두 가지가 문서화된
빌드 명령을 깨뜨리는데, 둘 다 LibreYOLO 문제가 아니라 환경 문제입니다.

이 이미지에는 `/usr/local` 아래에 `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8`,
`cuda-12.9`가 들어 있습니다. 완전한 툴킷을 갖춘 것은 `cuda-12.5`뿐입니다. 또한
`libcublas.so.12`와 `libcublas.so.12.8.4.1`은 있지만 `-lcublas`가 참조하는 버전
없는 `libcublas.so`는 없습니다. 아래 스크립트가 두 가지를 모두 우회합니다.

<code-tabs name="parser" />

그런 다음 생성된 설정의 `custom-lib-path`가 빌드된
`libnvdsinfer_custom_impl_Yolo.so`를 가리키도록 합니다. 생성되는 값은 상대 경로
`nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`이며, `deepstream-app`을
`DeepStream-Yolo` 체크아웃에서 실행할 때는 그대로 맞고 그 외의 경우에는 수정이
필요합니다.

## 파이프라인 실행

다른 데 시간을 쓰기 전에 컨테이너가 GPU를 인식하는지 확인합니다. 검증 실행도 WSL2의
Blackwell 카드에서 이 확인을 가장 먼저 했습니다.

<code-tabs name="gpu" />

검증 실행은 파일 소스 하나, 디스플레이 싱크 없음, 온스크린 디스플레이 켜기, 그리고
모든 프레임의 탐지 결과가 KITTI 텍스트로 디스크에 저장되도록 `gie-kitti-output-dir`를
설정한 상태로 `deepstream-app`을 구동했습니다. 그 설정을 담은 설정 파일:

<code-tabs name="run" />

`nvinfer`는 첫 실행에서 ONNX로부터 TensorRT 엔진을 빌드해 모델 옆에 캐시하므로, 첫
실행은 엔진 빌드 비용을 치르고 이후 실행은 캐시를 불러옵니다.

## 생성된 설정

아래 두 설정은 모두 검증 실행을 위해 익스포터가 생성한 그대로이며, 이후에 수정하지
않았습니다.

| 키 | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

두 설정은 세 곳에서 다릅니다: `maintain-aspect-ratio`, `cluster-mode`, 그리고
`nms-iou-threshold`가 아예 존재하는지 여부입니다. D-FINE 설정은 그 키를 통째로
생략하는데, `cluster-mode=4`가 요구하는 방식이 그렇습니다.

객체당 최대 하나의 예측만 내보내는 헤드에는 `cluster-mode=4`가 지정되어 DeepStream이
클러스터링을 하지 않습니다. 클러스터링은 실제로 서로 다른 탐지 결과를 병합해 버리기
때문입니다. 여기에 해당하는 것은 `rfdetr`, `dfine`, `deim`, `deimv2`, `ec`,
`rtdetr`, `rtdetrv2`, `rtdetrv4`, `yolo9_e2e`입니다. 그리드 헤드와 앵커 헤드에는
`cluster-mode=2`와 `nms-iou-threshold`가 함께 지정됩니다.

탐지 설정에는 `engine-create-func-name=NvDsInferYoloCudaEngineGet`도 들어 있어 엔진
빌드를 파서 라이브러리에 맡깁니다. 엔진 캐시 파일 이름을 고정하는 것이 바로 이
항목이며, 알려진 함정에서 설명하는 충돌의 원인이기도 합니다.

## 지원하는 작업과 계열

43개 계열과 작업 조합을 내보낼 수 있습니다. `libreyolo/export/deepstream.py`의
`deepstream_supported_tasks()`와 `deepstream_supported_families(task)`가 런타임에
같은 목록을 반환합니다.

| 작업 | `network-type` | 파서 라이브러리 | 계열 |
|---|---|---|---|
| 탐지 | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| 분류 | 1 | 필요 없음 | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| 시맨틱 분할 | 2 | 필요 없음 | pidnet, eomt, dinov2, lingbotvision |
| 인스턴스 분할 | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| 자세 추정 | 100 | 필요 없음 | yolo9, yolonas, rfdetr, ec |
| 깊이 추정 | 100 | 필요 없음 | depth_anything, zipdepth |
| 복원 | 100 | 필요 없음 | nafnet, realesrgan, swinir |
| 매팅 | 100 | 필요 없음 | birefnet |
| 시선 추정 | 100 | 필요 없음 | l2cs |

`network-type=100`은 DeepStream에 해당 작업의 후처리기가 없다는 뜻입니다. 그런
설정은 `output-tensor-meta=1`을 지정하고, 그래프의 원래 출력이 손대지 않은 채 그대로
전달되며, 애플리케이션이 텐서 메타데이터에서 이를 디코딩합니다. 출력이 여러 개인
그래프도 문제없습니다: 모든 출력 레이어가 일반 ONNX 내보내기와 동일한 출력 이름과
동적 축으로 메타데이터에 도달합니다.

인스턴스 분할의 각 행은 탐지 행 뒤에 해당 인스턴스의 마스크가 이어지는 형태이며,
마스크는 seg 파서가 하드코딩한 해상도인 `(netH / 4, netW / 4)`로 평탄화되어
`segmentation-threshold`용 확률값으로 들어갑니다.

분류와 시선 추정은 보조 추론으로 실행됩니다. 탐지 모델 뒤에 분류 모델을 두려면
생성된 설정에서 `process-mode=2`와 `operate-on-gie-id`를 지정합니다. 시선 추정은
헤드만 있는 계약이라 입력마다 얼굴 크롭 하나를 받으므로, 앞단에 얼굴 탐지 모델이
필요합니다.

세 계열은 의도적으로 빠져 있습니다. `segformer`는 공용 시맨틱 내보내기 계약에
연결되어 있지 않아 어떤 형식으로도 ONNX로 내보낼 수 없습니다. RTMDet-Ins와 YOLO9은
인스턴스 분할 내보내기가 LibreYOLO 자체에서 막혀 있습니다. `depth_anything3`은
내보내기 구현이 없습니다.

표의 두 행에는 체크포인트 공백이 있습니다. EoMT 시맨틱 체크포인트는 `l` 크기만
공개되어 있고, DINOv2 분류는 공개된 체크포인트가 아예 없어서 그 조합에는 직접
파인튜닝한 가중치가 필요합니다.

## 전처리 차이

`nvinfer`는 채널마다 `net-scale-factor * (x - offsets)`를 스칼라 스케일로
계산하므로, 채널별 표준편차를 표현할 수 없습니다. 표준편차가 필요한 계열(`rfdetr`,
`ec`, DINO 백본을 쓰는 `deimv2` 크기, `rtmdet`, `picodet`, 그리고 모든 분류 계열)은
정규화가 내보낸 그래프 안에 포함되어 있고, 생성된 설정은 그에 맞는 원본 입력 공간을
그래프에 전달합니다.

LibreYOLO 자체 Python 파이프라인과 `nvinfer`가 여전히 갈라지는 지점은 기하 처리입니다:

- 레터박스 계열(`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`, `yolo3`, `yolo4`,
  `yolo7`)은 자체 파이프라인에서 회색으로 패딩합니다. `nvinfer`는 검은색으로
  패딩합니다.
- `yolonas` 탐지는 자체 파이프라인에서 640 캔버스 안에 가장 긴 변을 636으로
  맞춥니다. `nvinfer`의 `maintain-aspect-ratio`는 640 전체를 사용합니다.
- 분류는 자체 파이프라인에서 가장 짧은 변을 맞춘 뒤 중앙을 잘라냅니다. `nvinfer`는
  프레임이나 객체 ROI를 네트워크 입력 크기로 늘리므로, 바짝 잘린 피사체에서 결과가
  달라집니다.
- EoMT는 자체 파이프라인에서 시맨틱 분할에 슬라이딩 윈도우 타일을 사용합니다.
  내보낸 그래프는 늘린 캔버스 하나여서 더 빠르고 덜 정확합니다.
- `pidnet`은 입력 해상도의 1/8, `lingbotvision`은 1/16 크기로 클래스 맵을
  내보냅니다. DeepStream은 표시할 때 클래스 맵을 업샘플링합니다.

ONNX 패리티 검사는 이미 전처리된 텐서를 입력하므로, 그래프 출력만 확인할 뿐 설정에
잘못된 색상 순서나 패딩 정책이 있어도 잡아내지 못합니다. 정확한 패리티가 필요한
작업을 배포하기 전에 직접 자신의 데이터로 검증하십시오.

## 알려진 함정

### 한 디렉터리에 있는 탐지 모델 두 개는 서로의 엔진을 불러옵니다

모든 탐지 설정에는 똑같은 줄이 들어 있습니다:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

파서의 엔진 빌더가 이 파일 이름을 요구하며, 모델에 따라 달라지지 않습니다. 같은
디렉터리에 두 번째 탐지 모델을 내보내면 두 번째 실행이 첫 번째 모델의 캐시된 엔진을
불러옵니다. 아무것도 죽지 않고, 박스만 틀리게 나옵니다. 탐지 모델마다 별도의
디렉터리를 지정하십시오. 검증 실행에서도 D-FINE을 별도 디렉터리로 분리한 뒤에야
테스트할 수 있었습니다.

### 박스 하나는 클래스 하나만 담을 수 있습니다

`nvinfer`의 행 형식은 `[x1, y1, x2, y2, score, class_id]`로 박스당 클래스 하나이므로,
내보내기는 클래스 점수를 argmax로 축약합니다. `predict`가 두 클래스로 보고하는 박스는
하나의 클래스로만 남습니다. 실측 사례: LibreYOLO는 같은 박스에 대해 `vase 0.773`과
`bottle 0.383`을 보고하지만, DeepStream 그래프는 `vase`를 유지합니다. 이는 파서의 행
형식에서 비롯되며 그 계약을 벗어나지 않고는 바꿀 수 없으므로, 회귀가 아니라 예상된
동작입니다.

## 검증된 항목

`deepstream-app`은 NVIDIA에 번들된 `sample_1080p_h264.mp4`(1443 프레임)에 대해 두 가지
탐지 헤드 유형 모두에서 프레임별 KITTI 덤프를 켠 채 `App run successful`과 함께
EOS까지 실행되었습니다.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| 헤드 유형 | 그리드 | 일대일 |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| 탐지가 있는 프레임 | 1443 | 1443 |
| 전체 탐지 수 | 18031 | 71105 |

1443 프레임 전체의 클래스 히스토그램은 두 모델 모두 자동차가 1위, 사람이 2위였고,
이는 거리 장면으로서 타당한 결과입니다. 탐지 수의 네 배 차이는 `cluster-mode` 차이가
제 역할을 한 결과입니다: `cluster-mode=4`인 D-FINE은 클러스터링을 하지 않으므로
임계값을 넘는 모든 쿼리가 살아남고, 거의 중복인 것들도 함께 남습니다.

독립적으로 학습된 두 모델이 가장 두드러진 객체를 같은 위치에 놓았습니다:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

이 실행으로 다섯 가지가 확인됩니다: TensorRT가 sm_120에서 내보낸 ONNX로부터 엔진을
빌드하고, `nvinfer`가 생성된 설정의 모든 키를 받아들이며, `NvDsInferParseYolo`가 텐서
레이아웃을 올바르게 읽고, 박스가 원본 해상도 1920x1080 좌표에 들어오며, 레이블이
생성된 레이블 파일과 맞아떨어집니다.

실행 환경:

| 구성 요소 | 값 |
|---|---|
| 호스트 OS | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| 드라이버 | 591.86 |
| 컴퓨트 능력 | 12.0 (Blackwell, sm_120) |
| 컨테이너 런타임 | Docker Desktop 29.4.3, WSL2 백엔드 |
| DeepStream 이미지 | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| DeepStream 버전 | 8.0.0 |
| 컨테이너 CUDA | 12.8.1 |
| 파서 | `marcoslucianops/DeepStream-Yolo`의 HEAD |

파이프라인 실행과 별개로, `tests/unit/test_deepstream_export.py`가 그래프 어댑터와
생성된 설정 키를 검사하며, 이 커밋에서 35개 테스트가 모두 통과합니다.

## 검증되지 않은 항목

위의 범위가 실제보다 넓게 읽히지 않도록 밝혀 둡니다.

- Jetson과 aarch64. 내보내기 계약은 아키텍처에 의존하지 않지만, 파이프라인은 x86
  외장 GPU에서만 실행했습니다.
- 43개 조합 중 41개. DeepStream을 거친 것은 `yolo9` 탐지와 `dfine` 탐지뿐입니다.
  분류, 시맨틱 분할, 인스턴스 분할, 원시 텐서 작업은 파이프라인 실행이 아니라 단위
  테스트와 ONNX 패리티 검사로 확인했습니다.
- FP16과 INT8. `network-mode=0`만 실행했습니다.
- 다중 스트림과 배치 처리. 소스 하나, `batch-size=1`.
- 정답(ground truth) 데이터셋 대비 정확도. 탐지 결과는 의미적 타당성과 모델 간 일치
  여부를 확인했을 뿐, DeepStream을 거쳐 mAP로 점수를 매기지는 않았습니다.
