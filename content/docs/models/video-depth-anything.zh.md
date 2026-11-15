---
title: Video Depth Anything
families:
  - video_depth_anything
seo_title: Video Depth Anything Python 用法：长视频的一致深度
description: >-
  在 LibreYOLO 里用 Video Depth Anything 为视频各帧生成时间上一致的相对深度。提供 s、b 和 l 三种尺寸；s 采用
  Apache-2.0 许可，b 和 l 仅限非商用。
lead: >-
  Video Depth Anything 出自字节跳动（ByteDance），它在 Depth Anything V2 编码器上加了一个时序
  head，让深度在帧与帧之间保持一致。LibreYOLO
  把一组帧、一个文件夹、一段视频或一个流当作一个序列来运行，为每一帧返回一张深度图，全部处在同一个相对尺度上。
keywords:
  - Video Depth Anything
  - 视频深度估计
  - 时序一致深度
  - Depth Anything V2
  - 相对深度
  - 视频 深度图 python
  - DINOv2
last_verified: 1.8.0
snippets:
  predict:
    - label: 视频
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVideoDepthAnythings-depth.pt")

        # 每帧一个结果，以流式返回，最多延迟一个窗口
        for result in model("clip.mp4", stream=True):
            depth = result.depth_map.data      # (H, W) 相对逆深度
    - label: 单张图像
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVideoDepthAnythings-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)   # 只有一帧的序列
        print(result.depth_map.data.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVideoDepthAnythings-depth.pt
        source=clip.mp4 save=True
source_hash: cfd2451ca039b9df
---

## 安装

Video Depth Anything 不需要任何可选 extra。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 LibreYOLO 的 Hugging Face 镜像下载。

<code-tabs name="predict" />

任何多图像数据源都是一个序列：一组帧、按文件名排序的目录、一个视频文件或一个流。
`N` 帧给出 `N` 个结果，每个结果都带有该帧画布尺寸上的 `result.depth_map`。内部按
上游的做法以相互重叠的 32 帧窗口运行，每个窗口都与之前的窗口做尺度和偏移对齐，所以
整个序列共享同一个相对尺度。单张图像是只有一帧的序列；上游会用它自身的副本把它填满
一个完整窗口。

输出遵循 LibreYOLO 共用的深度约定：相对逆深度，值越大表示越近，这些值没有度量单位。
每个编码器的输入尺寸都是上游默认的 518；宽于 16:9 的帧会像上游一样按比例缩小运行。
载荷说明见[深度估计](/docs/tasks/depth-estimation)任务页。

## 变体

三种编码器尺寸 `s`、`b` 和 `l`，各一个检查点。许可证因尺寸而异（见下文），所以在
商业产品里应该用 `s`。上游的度量深度检查点没有托管，转换脚本也会拒绝它们。

不提供训练：`train()` 会抛出异常。上游的相对深度检查点可以用库里的
`weights/convert_video_depth_anything_weights.py` 转换。

## 导出

<export-matrix />

不支持导出。网络读取的是一个 32 帧窗口，各窗口在主机端拼接，而单图像深度导出约定
无法描述这一点，所以 `export()` 会抛出异常。

## 检查点

<checkpoint-table />

## 许可证

<provenance-box>

Video Depth Anything 出自字节跳动
（[DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)）。
内置的代码采用 Apache-2.0 许可。权重的许可证取决于尺寸：

- `s`（Video-Depth-Anything-Small）：Apache-2.0，允许商用。
- `b` 和 `l`（Base 和 Large）：CC BY-NC 4.0，**仅限非商用**。
  它们附带一份非商用声明。

LibreYOLO 镜像了转换后的检查点（checkpoint），张量未作改动，命名为
`LibreYOLO/LibreVideoDepthAnything{s,b,l}-depth`，不需要申请访问。

</provenance-box>

## 引用

<citation-block />
