---
title: Prediction sources
seo_title: "Prediction sources in LibreYOLO"
description: "Every source predict accepts: images, folders, URLs, video files, webcams, RTSP, YouTube, screen capture, image lists and .streams files."
lead: "The source argument is classified before anything is opened, so one call handles a JPEG, a folder, an MP4, a webcam index, an RTSP URL, a screen region, or a list of cameras."
keywords:
  - yolo video inference python
  - rtsp
  - webcam object detection python
  - predict on a folder of images
  - screen capture object detection
  - multiple rtsp streams
  - streams file
  - youtube inference
  - vid_stride
  - stream=True
last_verified: "1.5.0"
verification: "Source classification read from libreyolo/utils/source.py (classify_source, SourceKind, StreamSource, MultiStreamSource). Accepted image types and directory extensions from libreyolo/utils/image_loader.py. Video extensions and save paths from libreyolo/utils/video.py. Screen syntax from libreyolo/utils/screen.py. Return shapes and argument defaults from InferenceRunner.__call__ in libreyolo/models/base/inference.py."
snippets:
  images:
    - label: One image
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # A single image source returns one Results, not a list.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: In-memory images
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
    - label: A folder
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

        # A folder returns a list, one Results per image, sorted by path.
        results = model(str(folder))
        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: A video file (supply your own clip)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Replace clip.mp4 with a video file on disk.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: Every third frame, written to disk
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Webcam (needs a camera attached)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Webcam index 0. Live sources never end, so bound the loop.
        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (needs a reachable camera URL)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: A .streams file (supply your own cameras)
      language: python
      code: |
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
        for result in itertools.islice(model("cameras.streams", stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: A list of cameras
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: One screenshot (needs mss and a desktop session)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Without stream=True this grabs a single frame.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: A region of one monitor, continuously
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "screen <monitor> <left> <top> <width> <height>"
        for result in itertools.islice(model("screen 1 100 200 512 256", stream=True), 50):
            print(len(result.boxes))
---

## How a source is classified

`classify_source` inspects the value before anything is opened or downloaded,
in this order. The first rule that matches wins.

| Source | Read as |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Screen capture |
| A non-negative `int`, or a digit string with no file of that name | Webcam |
| An `rtsp://`, `rtmp://`, `tcp://` or `udp://` URL | Network stream |
| An `http(s)://` URL whose path ends in `.m3u8` | Network stream |
| A YouTube page URL | Network stream |
| A list or tuple whose entries are all live or video | Several live streams |
| Any other list or tuple | Batch of images |
| A path ending in `.streams` | Several live streams |
| A path with a video extension | Video file |
| An existing directory | Folder of images |
| Anything else | Single image |

A list that mixes live sources with images raises `TypeError`. A negative
webcam index raises `ValueError`.

The classifier never touches the network, so a mistyped URL surfaces when the
capture opens, not when `predict` is called.

## Images

<code-tabs name="images" />

A single image source accepts seven types.

| Type | Read as |
|---|---|
| `str` or `pathlib.Path` | Local file, `http(s)://`, `s3://` or `gs://` |
| `PIL.Image.Image` | Converted to RGB |
| `numpy.ndarray` | 2D grayscale, or 3D HWC or CHW; a 4D array uses its first image |
| `torch.Tensor` | CHW or NCHW, read as RGB; a batched tensor uses its first image |
| `bytes` | Encoded image data |
| `io.BytesIO` | Encoded image data |

Everything is converted to RGB before preprocessing. NumPy arrays are the one
case where channel order is ambiguous, so `color_format` controls it:
`"auto"` (the default) leaves the array as-is, `"bgr"` reverses the channels,
which is what a frame read with OpenCV needs.

Float arrays are rescaled by their own range: values at or below `1.0` are
multiplied by 255, higher values are clipped into `[0, 255]`. An RGBA array
drops its alpha channel.

Remote paths need one package each, and none of them is installed by default:
`requests` for `http(s)://`, `boto3` for `s3://`, and `gcsfs` for `gs://`.

## Folders

A directory is scanned recursively and sorted, and every file with one of these
suffixes becomes an image: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`,
`.tiff`, `.tif`. Anything else in the folder is skipped. An empty folder returns
an empty list rather than raising.

Folders and lists are the two sources that accept `batch`, which runs one
stacked forward pass per chunk on families that support it. See
[Inference performance](/docs/predict/performance).

## Video files

<code-tabs name="video" />

A path counts as video when its suffix is one of `.asf`, `.avi`, `.gif`,
`.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`, `.webm`.

`.gif` appears in both lists. A `.gif` path passed directly to `predict` is
opened as video, because the video check runs first; a `.gif` sitting inside a
scanned folder is loaded as a still image.

`vid_stride` processes every N-th frame and defaults to `1`. Without
`stream=True` the whole video is decoded into a list, and anything above 500
frames after striding emits a warning suggesting `stream=True`.

Each `Results` from a video carries `frame_idx`.

## Webcams, network streams and YouTube

<code-tabs name="live" />

Live sources are unbounded, so they require `stream=True`. Without it,
`predict` raises `ValueError` rather than trying to collect an endless list.

Frames are read on a background thread, one per capture. By default the queue
holds only the newest frame, so a model slower than the camera skips frames
instead of falling behind. `stream_buffer=True` keeps every captured frame,
which preserves them at the cost of growing latency.

A webcam index is an `int` or a digit string. On Windows the capture is opened
through the DirectShow backend first and falls back to the default backend if
that fails.

YouTube page URLs are resolved to a direct media URL without downloading the
video, which needs `yt-dlp`:

```bash
pip install "libreyolo[stream]"
```

Stream labels are redacted before they are logged or used as filenames. A URL
carrying credentials appears as `user:***@host`, and query strings are dropped
from direct stream labels because signed URLs and bearer tokens live there. A
YouTube video id is kept, since it is not a credential.

## Several cameras at once

<code-tabs name="streams" />

A `.streams` file is one source per line. Blank lines and lines starting with
`#` are ignored. Every remaining line must itself be a webcam index, a network
stream, a YouTube URL or a video file path; anything else raises `ValueError`
naming the line number. An empty file raises rather than starting with no
cameras.

A list or tuple of live sources does the same thing without a file.

Each capture gets its own thread, and frames from all of them are multiplexed
into one generator. Every pass polls each active stream and yields whatever is
ready, so a slow camera does not hold up a fast one, and frames from different
cameras interleave. A stream that ends drops out of the rotation while the
others continue.

## Screen capture

<code-tabs name="screen" />

A screen source is the word `screen` followed by zero, one, four or five
integers. Any other count raises `ValueError`.

| Form | Captures |
|---|---|
| `"screen"` | Every monitor, merged |
| `"screen 1"` | Monitor 1 |
| `"screen 100 200 512 256"` | A box on the merged desktop |
| `"screen 1 100 200 512 256"` | A box on monitor 1 |

Box coordinates are `left top width height`, relative to the top-left corner of
the chosen monitor. A screen source reports its frame rate as 30 divided by
`vid_stride`, which is the rate a saved video is written at. Capture needs the
`mss` package:

```bash
pip install mss
```

Without `stream=True`, a screen source grabs one frame and returns a single
`Results`, which is the screenshot equivalent of predicting on an image file.
With `stream=True` it captures until the loop is broken.

## What predict returns

The shape of the return value depends on the source and on `stream`.

| Source | `stream=False` | `stream=True` |
|---|---|---|
| Single image | One `Results` | Generator of one `Results` |
| List of images | List of `Results` | Generator |
| Folder | List of `Results` | Generator |
| Video file | List of `Results` | Generator |
| Screen | One `Results` | Generator, unbounded |
| Webcam, network stream, `.streams` | `ValueError` | Generator, unbounded |

A single image returns the `Results` object itself. Indexing it selects a
detection, not an image, so `result[0]` on a single-image prediction is the
first box rather than the first picture. For what those objects carry, see
[Working with results](/docs/predict/results).

## Where save writes

`save=True` writes annotated output next to a run directory rather than
returning it.

Images go to an auto-incrementing `runs/detect/predict`, `runs/detect/predict2`
and so on, keeping the source filename. Every image in one process lands in the
same directory, so two input folders holding the same filename overwrite each
other. In-memory images have no filename to reuse and are numbered `image0`,
`image1` and so on.

Video and live sources are written as a single `.mp4` named after the source.

`output_path` overrides the directory. A path with a suffix is treated as a
file, a path without one as a directory. `output_file_format` selects the
still-image encoding and accepts `jpg`, `png` or `webp`.

After a save, the written path is also attached to the result as
`result.saved_path`.
