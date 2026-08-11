---
title: 目标跟踪
seo_title: LibreYOLO 里的目标跟踪
description: >-
  在 LibreYOLO 里用 ByteTrack、BoT-SORT、OC-SORT 或 Deep OC-SORT
  跨视频帧跟踪目标，可以搭配任意检测、分割或姿态模型。
lead: >-
  跟踪为每个检测结果在视频帧之间赋予一个稳定的身份。LibreYOLO
  没有把它建模成一个自带权重的任务：它是一种预测模式，model.track()，在检测、分割或姿态模型的逐帧输出之上运行选定的跟踪器。
keywords:
  - 目标跟踪 python
  - 多目标跟踪
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - yolo 视频跟踪
  - reid 跟踪
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track() 是一个生成器：每处理一帧产出一个 Results
        for result in model.track("video.mp4"):
            print(result.track_id)        # (N,) 整数张量，与 boxes 对齐
            print(result.boxes.xyxy)
    - label: 选择跟踪器
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack"（默认）、"botsort"、"ocsort" 或 "deepocsort"
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: 保存带标注的视频
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 不传 output_path 时，文件会落在 runs/track/<video_stem>.mp4
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: 调优跟踪器
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # 配置类型决定了跟踪器，所以这里的 tracker= 是多余的

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # 或者把同样的字段作为关键字参数传进去，让 track() 来构建

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## 定义

跟踪不是 LibreYOLO 的任务键之一，也没有跟踪检查点（checkpoint）可以下载。它是模型
上的一个方法 `model.track(source)`，在每一帧上跑检测，再把结果跨时间关联起来。这个
方法是一个生成器：每处理一帧就产出一个 `Results`，其中 `result.track_id` 是一个与
`result.boxes` 对齐的 `(N,)` 整数张量。同样的 ID 也挂在 `result.boxes.id` 上。

只有已确认、当前正在跟踪的目标会被产出。关联丢失的轨迹会在被丢弃前再存活可配置的
帧数——ByteTrack 和 BoT-SORT 用 `track_buffer`，两个 OC-SORT 变体用 `max_age`——所以
在这个窗口内被重新找回的目标会保留原来的 ID。

因为关联发生在检测之后，这一帧的其他载荷也能留下来：跟踪后的 `Results` 就是检测的
`Results` 按匹配到的行切片得到的，所以掩码和关键点会跟着检测框一起带过来。

## 模型

一次跟踪运行涉及两个独立的选择：每帧产出检测框的模型，以及把它们串起来的跟踪器。

任何任务是检测、分割或姿态的原生 LibreYOLO 模型都提供 `track()`，所以检测器的选择
就是平常那一套。完整列表见[模型索引](/docs/models)，也可以从
[YOLO9](/docs/models/yolov9)、[RF-DETR](/docs/models/rf-detr)、
[D-FINE](/docs/models/d-fine) 或 [RTMDet](/docs/models/rtmdet) 开始。结果里没有检测框
可供关联的任务会直接拒绝这次调用，而不是返回没有意义的 ID：分类、旋转框、点、深度、
表面法线、边缘、语义分割和全景分割、图像修复、OCR 以及人体网格在 `track()` 上都会
抛错。

LibreYOLO 还有两个模型层级同样不接受它。通过 `LibreSAM` 加载的模型是图像分割器，
通过 `LibreOpenVocab` 加载的模型是逐帧检测器；两者在 `track()` 上都会抛错，取而代之
的用法是逐帧调用 `predict()`。

跟踪跑在原生 PyTorch 模型上。通过 `LibreYOLO("model.onnx")` 加载的导出产物返回的是一
个运行时后端对象，它带有 `predict()`，但没有 `track()`。

库里自带四个跟踪器，用 `tracker` 参数选择：

`"bytetrack"` 是默认值。它只用运动信息，配一个卡尔曼滤波器和三阶段关联：先是高置信度
检测，然后是第二轮，让低置信度检测在被丢弃前有机会匹配上已有的轨迹，最后是未确认的
轨迹。用 `TrackConfig` 配置。

`"botsort"` 保留了 ByteTrack 的三阶段生命周期，但改用中心点-宽-高的卡尔曼状态，并在
匹配前对预测出的轨迹做相机运动补偿。这是 BoT-SORT 的纯运动变体，它不跑外观模型。用
`BoTSortConfig` 配置，它额外加了 `enable_cmc`、`cmc_method` 和 `cmc_downscale`。

`"ocsort"` 同样只用运动信息，并在关联代价里加了一个速度方向项，加了一轮针对每条轨迹
最后一次真实观测的关联，以及在轨迹被重新找回时沿一条虚拟轨迹对卡尔曼状态做平滑。用
`OCSortConfig` 配置。

`"deepocsort"` 在 OC-SORT 之上补了外观。每条轨迹维护一份按置信度加权的重识别嵌入向量
滑动平均，关联代价里再加上一个余弦相似度项，于是身份能挺过长时间遮挡和目标交叉。它的
代价是每帧多跑一次小型嵌入网络的前向，而它的 OSNet 权重会在首次使用时下载。用
`DeepOCSortConfig` 配置。

## 预测

<code-tabs name="predict" />

`track_conf` 设置第一阶段关联的阈值：ByteTrack 和 BoT-SORT 用 `track_high_thresh`，
OC-SORT 和 Deep OC-SORT 用 `det_thresh`。它不是 `predict()` 的 `conf`，而且对
ByteTrack、BoT-SORT 和 OC-SORT 来说，检测器在内部会以更低的阈值运行，好让弱检测留给
恢复那一轮用。Deep OC-SORT 则直接以 `det_thresh` 运行检测器。对 ByteTrack 和
BoT-SORT 来说，`track_conf` 必须大于或等于 `track_low_thresh`，后者默认是 0.1。

跟踪器设置有两种传法。把一个配置实例传给 `tracker_config=`，它的类型就选定了跟踪器，
`tracker=` 因此变得多余。或者把这些字段作为关键字参数传进去，让 `track()` 为你指定的
跟踪器构建配置；不认识的键会告警，而不是被悄悄应用。无论哪种方式，一旦对应的键被显式
设置，`track_conf` 就会被忽略。

其余参数和预测保持一致：`iou`、`imgsz`、`classes`、`max_det`、`vid_stride`、`show`，
以及 `save` 配 `output_path`。源是一个视频文件路径。结果处理见[预测](/docs/predict)。

## 训练

跟踪器不需要训练。四个里有三个是纯运动模型，完全没有学习得到的参数，而 Deep OC-SORT
的外观网络是一个已发布的重识别检查点，会在首次使用时下载。想提升跟踪质量，就得提升
检测器，或者调上面那些关联阈值。
