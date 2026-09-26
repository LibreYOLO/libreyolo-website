---
title: 预测输入源
seo_title: LibreYOLO 中的预测输入源
description: predict 接受的每一种输入源：图像、文件夹、URL、视频文件、摄像头、RTSP、YouTube、屏幕捕获、图像列表和 .streams 文件。
lead: source 参数在打开任何东西之前就完成分类，所以一次调用就能处理 JPEG、文件夹、MP4、摄像头索引、RTSP URL、屏幕区域或一组摄像头。
keywords:
  - yolo 视频推理 python
  - rtsp
  - 摄像头目标检测 python
  - yolo 批量预测文件夹图片
  - 屏幕捕获目标检测
  - 多路 rtsp 流
  - streams 文件
  - youtube 推理
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  输入源分类读自
  libreyolo/utils/source.py（classify_source、SourceKind、StreamSource、MultiStreamSource）。接受的图像类型和目录扩展名来自
  libreyolo/utils/image_loader.py。视频扩展名和保存路径来自 libreyolo/utils/video.py。屏幕语法来自
  libreyolo/utils/screen.py。返回形态和参数默认值来自 libreyolo/models/base/inference.py 中的
  InferenceRunner.__call__。
snippets:
  images:
    - label: 单张图像
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # 单张图像源返回一个 Results，而不是列表
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: 内存中的图像
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
    - label: 一个文件夹
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

        # 文件夹返回一个列表，每张图像一个 Results，按路径排序
        results = model(str(folder))
        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: 视频文件（自备片段）
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 把 clip.mp4 换成磁盘上的视频文件
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: 每三帧处理一帧，并写入磁盘
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: 摄像头（需要接一个摄像头）
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 摄像头索引 0，实时源永远不会结束，所以要给循环加个上界
        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP（需要一个能连通的摄像头 URL）
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: .streams 文件（自备摄像头）
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
    - label: 一组摄像头
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: 单张截图（需要 mss 和桌面会话）
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 不加 stream=True 时只抓取一帧
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: 持续捕获某个显示器上的一块区域
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

## 输入源是如何分类的

`classify_source` 会在打开或下载任何东西之前检查这个值，顺序如下。第一条命中的规则生效。

| 输入源 | 识别为 |
|---|---|
| `"screen"`、`"screen 1"`、`"screen 1 100 200 512 256"` | 屏幕捕获 |
| 非负的 `int`，或没有同名文件的数字字符串 | 摄像头 |
| `rtsp://`、`rtmp://`、`tcp://` 或 `udp://` URL | 网络流 |
| 路径以 `.m3u8` 结尾的 `http(s)://` URL | 网络流 |
| YouTube 页面 URL | 网络流 |
| 元素全部是实时源或视频的列表或元组 | 多路实时流 |
| 其他任何列表或元组 | 一批图像 |
| 以 `.streams` 结尾的路径 | 多路实时流 |
| 带视频扩展名的路径 | 视频文件 |
| 已存在的目录 | 图像文件夹 |
| 其他任何情况 | 单张图像 |

把实时源和图像混在一个列表里会抛出 `TypeError`。负的摄像头索引会抛出 `ValueError`。

分类器从不访问网络，所以打错的 URL 会在采集打开时才暴露出来，而不是在调用 `predict` 的时候。

## 图像

<code-tabs name="images" />

单张图像源接受七种类型。

| 类型 | 识别为 |
|---|---|
| `str` 或 `pathlib.Path` | 本地文件、`http(s)://`、`s3://` 或 `gs://` |
| `PIL.Image.Image` | 转换为 RGB |
| `numpy.ndarray` | 2D 灰度，或 3D HWC 或 CHW；4D 数组取其中第一张图像 |
| `torch.Tensor` | CHW 或 NCHW，按 RGB 读取；批量张量取其中第一张图像 |
| `bytes` | 编码后的图像数据 |
| `io.BytesIO` | 编码后的图像数据 |

所有输入在预处理之前都会转换为 RGB。NumPy 数组是唯一通道顺序有歧义的情况，所以由 `color_format` 控制：`"auto"`（默认）保持数组原样，`"bgr"` 反转通道顺序，用 OpenCV 读到的帧需要的就是这个。

浮点数组按自身的取值范围重新缩放：小于等于 `1.0` 的值乘以 255，更大的值裁剪到 `[0, 255]`。RGBA 数组会丢掉自己的 alpha 通道。

远程路径各需要一个包，而且默认都不安装：`http(s)://` 需要 `requests`，`s3://` 需要 `boto3`，`gs://` 需要 `gcsfs`。

## 文件夹

目录会被递归扫描并排序，带下面这些后缀的文件都会当作图像：`.jpg`、`.jpeg`、`.png`、`.gif`、`.webp`、`.bmp`、`.tiff`、`.tif`。文件夹里的其他文件会被跳过。空文件夹返回空列表，而不是抛异常。

文件夹和列表是仅有的两种接受 `batch` 的输入源；在支持的模型家族上，它会对每一块做一次堆叠的前向传播。参见[推理性能](/docs/predict/performance)。

## 视频文件

<code-tabs name="video" />

当路径的后缀是 `.asf`、`.avi`、`.gif`、`.m4v`、`.mkv`、`.mov`、`.mp4`、`.mpeg`、`.mpg`、`.ts`、`.wmv`、`.webm` 之一时，它算作视频。

`.gif` 同时出现在两个列表里。直接传给 `predict` 的 `.gif` 路径会按视频打开，因为视频检查先执行；而扫描文件夹时遇到的 `.gif` 会按静态图像加载。

`vid_stride` 每 N 帧处理一帧，默认为 `1`。不加 `stream=True` 时整段视频会解码成一个列表，抽帧之后超过 500 帧就会发出警告，建议改用 `stream=True`。

视频产生的每个 `Results` 都带有 `frame_idx`。

## 摄像头、网络流和 YouTube

<code-tabs name="live" />

实时源是无界的，所以必须加 `stream=True`。不加的话，`predict` 会抛出 `ValueError`，而不是去收集一个没有尽头的列表。

帧在后台线程上读取，每路采集一个线程。默认队列只保留最新的一帧，所以比摄像头慢的模型会跳过一些帧，而不是越拖越远。`stream_buffer=True` 会保留采集到的每一帧，把它们都留住，代价是延迟不断增长。

摄像头索引是 `int` 或数字字符串。在 Windows 上，采集先通过 DirectShow 后端打开，失败时回退到默认后端。

YouTube 页面 URL 会被解析成直链媒体 URL，不会下载视频，这需要 `yt-dlp`：

```bash
pip install "libreyolo[stream]"
```

流的标签在写入日志或用作文件名之前会先做脱敏。带凭据的 URL 显示为 `user:***@host`，直链流的标签会丢掉查询字符串，因为签名 URL 和 bearer token 就在那里。YouTube 视频 id 会保留，因为它不是凭据。

## 同时接多路摄像头

<code-tabs name="streams" />

`.streams` 文件每行一个输入源。空行和以 `#` 开头的行会被忽略。其余每一行本身必须是摄像头索引、网络流、YouTube URL 或视频文件路径；否则会抛出 `ValueError` 并指出行号。空文件会抛异常，而不是在一路摄像头都没有的情况下启动。

由实时源组成的列表或元组不需要文件就能做到同样的事。

每路采集有自己的线程，所有采集的帧会复用到同一个生成器里。每一轮都会轮询每一路仍在工作的流，把已经就绪的帧产出来，所以慢的摄像头不会拖住快的，不同摄像头的帧会交错出现。结束的流会退出轮转，其余的继续。

## 屏幕捕获

<code-tabs name="screen" />

屏幕源是 `screen` 这个词，后面跟零个、一个、四个或五个整数。其他个数会抛出 `ValueError`。

| 形式 | 捕获内容 |
|---|---|
| `"screen"` | 所有显示器，合并在一起 |
| `"screen 1"` | 显示器 1 |
| `"screen 100 200 512 256"` | 合并桌面上的一块区域 |
| `"screen 1 100 200 512 256"` | 显示器 1 上的一块区域 |

区域坐标是 `left top width height`，相对于所选显示器的左上角。屏幕源报告的帧率是 30 除以 `vid_stride`，保存视频时写入用的就是这个帧率。捕获需要 `mss` 包：

```bash
pip install mss
```

不加 `stream=True` 时，屏幕源抓取一帧并返回单个 `Results`，相当于对图像文件做预测的截图版本。加上 `stream=True` 则会一直捕获，直到循环被打断。

## predict 返回什么

返回值的形态取决于输入源和 `stream`。

| 输入源 | `stream=False` | `stream=True` |
|---|---|---|
| 单张图像 | 一个 `Results` | 产出一个 `Results` 的生成器 |
| 图像列表 | `Results` 列表 | 生成器 |
| 文件夹 | `Results` 列表 | 生成器 |
| 视频文件 | `Results` 列表 | 生成器 |
| 屏幕 | 一个 `Results` | 生成器，无界 |
| 摄像头、网络流、`.streams` | `ValueError` | 生成器，无界 |

单张图像返回的就是 `Results` 对象本身。对它取下标选中的是一个检测结果，而不是一张图像，所以单图预测上的 `result[0]` 是第一个检测框，而不是第一张图片。关于这些对象都带了什么，参见[处理预测结果](/docs/predict/results)。

## save 写到哪里

`save=True` 会把带标注的输出写进一个运行目录，而不是把它返回。

图像写入自动递增的 `runs/detect/predict`、`runs/detect/predict2` 等目录，并保留源文件名。同一个进程里的每张图像都落在同一个目录下，所以两个输入文件夹里有同名文件时会互相覆盖。内存中的图像没有文件名可以复用，会依次编号为 `image0`、`image1` 等。

视频源和实时源写成单个 `.mp4`，以输入源命名。

`output_path` 会覆盖这个目录。带后缀的路径当作文件，不带后缀的当作目录。`output_file_format` 选择静态图像的编码格式，接受 `jpg`、`png` 或 `webp`。

保存之后，写入的路径也会挂到结果上，即 `result.saved_path`。
