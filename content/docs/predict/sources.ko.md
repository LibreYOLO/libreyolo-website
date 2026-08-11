---
title: 예측 소스
seo_title: LibreYOLO의 예측 소스
description: >-
  모든 소스 예측은 다음을 수락합니다: 이미지, 폴더, URL, 비디오 파일, 웹캠, RTSP, YouTube, 화면 캡처, 이미지 목록 및
  .streams 파일.
lead: >-
  소스 인수는 어떤 것이 열리기 전에 분류되므로, 한 번의 호출로 JPEG, 폴더, MP4, 웹캠 인덱스, RTSP URL, 화면 영역 또는
  카메라 목록을 처리할 수 있습니다.
keywords:
  - yolo 비디오 추론 파이썬
  - rtsp
  - 웹캠 객체 탐지 파이썬
  - 이미지 폴더에서 예측
  - 화면 캡처 객체 탐지
  - 여러 RTSP 스트림
  - streams 파일
  - 유튜브 추론
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  libreyolo/utils/source.py(classify_source, SourceKind, StreamSource,
  MultiStreamSource)에서 읽은 소스 분류. libreyolo/utils/image_loader.py.에서 수락되는 이미지 유형과
  디렉터리 확장자, libreyolo/utils/video.py.에서 비디오 확장자와 저장 경로,
  libreyolo/utils/screen.py.에서 화면 구문, libreyolo/models/base/inference.py.의
  InferenceRunner.__call__에서 반환 형상과 인수 기본값.
snippets:
  images:
    - label: 하나의 이미지
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # 단일 이미지 소스는 리스트가 아닌 하나의 Results를 반환합니다.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: 메모리 내 이미지
      language: python
      code: |
        import numpy as np
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        pil_image = Image.open(SAMPLE_IMAGE)
        array = np.asarray(pil_image)
        raw_bytes = open(SAMPLE_IMAGE, "rb").read()

        for source in (pil_image, array, raw_bytes):
            result = model(source)
            print(type(source).__name__, len(result.boxes))
    - label: 폴더
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("sample_folder")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(3):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # 폴더는 리스트를 반환하며, 각 이미지마다 하나의 Results가 경로 순으로 정렬됩니다.
        results = model(str(folder))
        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: 비디오 파일(자신의 클립 제공)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # clip.mp4를 디스크에 있는 비디오 파일로 교체하세요.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: 세 번째 프레임마다 디스크에 쓰기
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: 웹캠(카메라가 연결되어 있어야 함)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 웹캠 인덱스 0. 실시간 소스는 끝나지 않으므로 루프를 제한하세요.
        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP(접근 가능한 카메라 URL 필요)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: A .streams 파일 (자신의 카메라 제공)
      language: python
      code: >
        import itertools

        from pathlib import Path


        from libreyolo import LibreYOLO


        Path("cameras.streams").write_text(
            "# one source per line, blank lines and comments are skipped\n"
            "rtsp://192.168.1.64:554/Streaming/Channels/101\n"
            "rtsp://192.168.1.65:554/Streaming/Channels/101\n",
            encoding="utf-8",
        )


        model = LibreYOLO("LibreYOLO9s.pt")

        for result in itertools.islice(model("cameras.streams", stream=True),
        100):
            print(result.frame_idx, len(result.boxes))
    - label: 카메라 목록
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: 한 장의 스크린샷 (mss와 데스크톱 세션 필요)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # stream=True 없이 단일 프레임을 가져옵니다.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: '하나의 모니터 영역, 연속으로'
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # "screen <monitor> <left> <top> <width> <height>"

        for result in itertools.islice(model("screen 1 100 200 512 256",
        stream=True), 50):
            print(len(result.boxes))
source_hash: c371965951dd0181
---

## 소스가 분류되는 방법

`classify_source`는 어떤 것이 열리거나 다운로드되기 전에 값을 검사합니다. 순서는 이렇습니다. 일치하는 첫 번째 규칙이 적용됩니다.

| 소스 | 다음으로 읽기 |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | 화면 캡처 |
| 0 이상의 `int`, 또는 그 이름의 파일이 없는 숫자 문자열 | 웹캠 |
| `rtsp://`, `rtmp://`, `tcp://` 또는 `udp://` URL | 네트워크 스트림 |
| 경로가 `.m3u8`로 끝나는 `http(s)://` URL | 네트워크 스트림 |
| 유튜브 페이지 URL | 네트워크 스트림 |
| 항목이 모두 라이브 또는 비디오인 리스트나 튜플 | 여러 라이브 스트림 |
| 다른 리스트나 튜플 | 이미지 배치 |
| `.streams`로 끝나는 경로 | 여러 라이브 스트림 |
| 비디오 확장자를 가진 경로 | 비디오 파일 |
| 기존 디렉토리 | 이미지 폴더 |
| 기타 모든 것 | 단일 이미지 |

라이브 소스와 이미지를 혼합한 리스트는 `TypeError`를 발생시킵니다. 잘못된 웹캠 인덱스는 `ValueError`를 발생시킵니다.

분류기는 네트워크에 접근하지 않으므로 잘못 입력된 URL은 `predict`가 호출될 때가 아니라 캡처가 열릴 때 나타납니다.

## 이미지

<code-tabs name="images" />

단일 이미지 소스는 일곱 가지 유형을 허용합니다.

| 유형 | 읽기 방식 |
|---|---|
| `str` 또는 `pathlib.Path` | 로컬 파일, `http(s)://`, `s3://` 또는 `gs://` |
| `PIL.Image.Image` | RGB로 변환 |
| `numpy.ndarray` | 2D 그레이스케일 또는 3D HWC 혹은 CHW; 4D 배열은 첫 번째 이미지를 사용합니다 |
| `torch.Tensor` | CHW 또는 NCHW, RGB로 읽기; 배치된 텐서는 첫 번째 이미지를 사용합니다 |
| `bytes` | 인코딩된 이미지 데이터 |
| `io.BytesIO` | 인코딩된 이미지 데이터 |

모든 것은 전처리 전에 RGB로 변환됩니다. NumPy 배열은 채널 순서가 모호한 경우이므로 `color_format`가 이를 제어합니다: `"auto"` (기본값)은 배열을 그대로 두고, `"bgr"`는 채널을 반전시키는데, 이는 OpenCV로 읽은 프레임이 필요한 경우입니다.

부동 소수점 배열은 자체 범위에 따라 재조정됩니다: `1.0` 이하의 값은 255로 곱해지고, 더 높은 값은 `[0, 255]`로 잘립니다. RGBA 배열은 알파 채널을 제거합니다.

원격 경로는 각각 하나의 패키지가 필요하며, 기본적으로 설치되어 있지 않습니다: `http(s)://`용 `requests`, `s3://`용 `boto3`, `gs://`용 `gcsfs`.

## 폴더

디렉토리는 재귀적으로 스캔되고 정렬되며, 이 접미사를 가진 모든 파일은 이미지가 됩니다: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`. 폴더의 다른 모든 항목은 건너뜁니다. 빈 폴더는 오류를 발생시키지 않고 빈 목록을 반환합니다.

폴더와 목록은 `batch`를 허용하는 두 가지 소스이며, 이는 이를 지원하는 계열에서 청크마다 한 번의 누적 전방 패스를 실행합니다. [추론 성능](/docs/predict/performance)을 참조하십시오.

## 비디오 파일

<code-tabs name="video" />

경로의 접미사가 `.asf`, `.avi`, `.gif`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`, `.webm` 중 하나이면 비디오로 간주됩니다.

`.gif`는 두 목록 모두에 나타납니다. `.gif` 경로가 `predict`로 직접 전달되면, 비디오 검사가 먼저 실행되므로 비디오로 열립니다; 스캔된 폴더 안에 있는 `.gif`는 정지 이미지로 로드됩니다.

`vid_stride`는 N번째 프레임마다 처리하며 기본값은 `1`입니다. `stream=True` 없이 전체 비디오가 목록으로 디코딩되며, 500프레임 이상이 경과하면 `stream=True`를 권장하는 경고가 표시됩니다.

비디오의 각 `Results`는 `frame_idx`를 포함합니다.

## 웹캠, 네트워크 스트림 및 YouTube

<code-tabs name="live" />

라이브 소스는 제한되지 않으므로 `stream=True`가 필요합니다. 없으면 `predict`는 끝없는 목록을 수집하려고 시도하는 대신 `ValueError`를 발생시킵니다.

프레임은 백그라운드 스레드에서 한 캡처당 하나씩 읽습니다. 기본적으로 큐에는 최신 프레임만 유지되므로, 카메라보다 느린 모델은 뒤처지기보다는 프레임을 건너뛰게 됩니다. `stream_buffer=True`는 모든 캡처된 프레임을 유지하여 지연 시간을 증가시키는 대가로 프레임을 보존합니다.

웹캠 인덱스는 `int` 또는 숫자 문자열입니다. Windows에서는 캡처가 먼저 DirectShow 백엔드를 통해 열리며 실패하면 기본 백엔드로 대체됩니다.

YouTube 페이지 URL은 비디오를 다운로드하지 않고 직접 미디어 URL로 해결되며, 이는 `yt-dlp`가 필요합니다:

```bash
pip install "libreyolo[stream]"
```

스트림 레이블은 기록되거나 파일 이름으로 사용되기 전에 삭제됩니다. 자격 증명을 포함한 URL은 `user:***@host`로 표시되며, 서명된 URL과 베어러 토큰이 존재하기 때문에 직접 스트림 레이블에서 쿼리 문자열은 삭제됩니다. YouTube 동영상 ID는 자격 증명이 아니므로 유지됩니다.

## 여러 카메라 한 번에

<code-tabs name="streams" />

`.streams` 파일은 한 줄에 하나의 소스를 나타냅니다. 빈 줄과 `#`로 시작하는 줄은 무시됩니다. 나머지 모든 줄은 자체적으로 웹캠 인덱스, 네트워크 스트림, YouTube URL 또는 비디오 파일 경로여야 하며, 그렇지 않으면 해당 줄 번호를 명시한 `ValueError`가 발생합니다. 빈 파일은 카메라가 없는 상태에서 시작하는 대신 오류를 발생시킵니다.

라이브 소스의 리스트나 튜플도 파일 없이 동일하게 작동합니다.

각 캡처는 자체 스레드를 가지며, 모든 캡처의 프레임은 하나의 제너레이터로 다중화됩니다. 각각의 패스는 모든 활성 스트림을 폴링하고 준비된 프레임을 반환하므로, 느린 카메라가 빠른 카메라를 지연시키지 않으며, 서로 다른 카메라의 프레임이 섞입니다. 스트림이 종료되면 나머지 스트림이 계속되는 동안 회전에서 제외됩니다.

## 화면 캡처

<code-tabs name="screen" />

화면 소스는 `screen`라는 단어 뒤에 0개, 1개, 4개 또는 5개의 정수가 옵니다. 그 외의 개수는 `ValueError`를 발생시킵니다.

| 양식 | 캡처 |
|---|---|
| `"screen"` | 모든 모니터, 병합됨 |
| `"screen 1"` | 모니터 1 |
| `"screen 100 200 512 256"` | 병합된 데스크톱의 박스 |
| `"screen 1 100 200 512 256"` | 모니터 1의 박스 |

박스 좌표는 선택한 모니터의 좌측 상단 모서리를 기준으로 `left top width height`입니다. 화면 소스는 프레임 속도를 `vid_stride`로 나눈 30으로 보고하며, 이는 저장된 비디오가 기록되는 속도입니다. 캡처에는 `mss` 패키지가 필요합니다:

```bash
pip install mss
```

`stream=True`가 없으면, 화면 소스는 한 프레임을 잡아 단일 `Results`를 반환하며, 이는 이미지 파일을 예측하는 것과 동일한 스크린샷입니다. `stream=True`가 있으면 루프가 끊어질 때까지 캡처합니다.

## predict가 반환하는 것

반환 값의 형태는 소스와 `stream`에 따라 달라집니다.

| 소스 | `stream=False` | `stream=True` |
|---|---|---|
| 단일 이미지 | 한 개의 `Results` | 한 개의 `Results` 생성기 |
| 이미지 목록 | `Results` 목록 | 생성기 |
| 폴더 | `Results` 목록 | 생성기 |
| 비디오 파일 | `Results` 목록 | 생성기 |
| 화면 | 하나의 `Results` | 생성기, 무제한 |
| 웹캠, 네트워크 스트림, `.streams` | `ValueError` | 생성기, 무제한 |

단일 이미지는 `Results` 객체 자체를 반환합니다. 인덱싱은 이미지를 선택하는 것이 아니라 검출을 선택하므로 단일 이미지 예측의 `result[0]`는 첫 번째 그림이 아니라 첫 번째 박스입니다. 해당 객체가 무엇을 포함하는지에 대해서는 [결과 작업](/docs/predict/results)을 참조하십시오.

## 저장 위치

`save=True`는 반환하는 대신 실행 디렉토리 옆에 주석 처리된 출력을 작성합니다.

이미지들은 자동 증가되는 `runs/detect/predict`, `runs/detect/predict2` 등으로 이동하며, 원본 파일명을 유지합니다. 한 프로세스의 모든 이미지는 같은 디렉토리에 저장되므로, 동일한 파일명을 가진 두 입력 폴더는 서로 덮어씁니다. 메모리 내 이미지에는 재사용할 파일명이 없으며 `image0`, `image1` 등으로 번호가 매겨집니다.

비디오와 라이브 소스는 소스 이름을 따르는 단일 `.mp4`로 기록됩니다.

`output_path`는 디렉토리를 무시합니다. 접미사가 있는 경로는 파일로 처리되고, 없는 경로는 디렉토리로 처리됩니다. `output_file_format`는 정지 이미지 인코딩을 선택하며 `jpg`, `png` 또는 `webp`를 허용합니다.

저장 후, 작성된 경로는 결과에 `result.saved_path`로 첨부됩니다.
