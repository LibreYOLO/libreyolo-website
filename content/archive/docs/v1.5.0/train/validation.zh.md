---
title: 验证与指标
seo_title: LibreYOLO 中的验证与指标
description: 在任意模型上运行 val()，查看每种任务返回的指标键，选择评估后端，并在精度指标之外打开验证损失。
lead: >-
  验证通过 val() 把一个模型跑在一个数据集划分（split）上，返回一个由指标键和 float
  值组成的扁平字典。这些键都是字面字符串，你拿到哪些取决于任务，而不是家族。
keywords:
  - map50-95
  - coco 评估
  - 验证指标
  - faster-coco-eval
  - pycocotools
  - 验证损失
  - miou
  - 全景质量 pq
  - top1 准确率
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: 在其他划分上
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: 写出 COCO 格式的预测
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## 运行一次验证

`val()` 接收数据集，返回指标。

<code-tabs name="val" />

返回值是一个普通的 `dict[str, float]`。每个键都是字面量，所以按名字读，而不是按
位置读。

主要参数是 `data`、`split`、`batch`、`imgsz`、`conf`、`iou`、`workers`、`device`、
`augment`、`save_json` 和 `verbose`。`conf` 默认是 `0.001`，`iou` 默认是 `0.6`，
两个都比预测的默认值宽松得多，因为一次 mAP 扫描需要低置信度的那条长尾。`imgsz`
默认取模型自身的输入尺寸，而不是一个固定数字。`split` 只接受 `val`、`test` 或
`train`，别的都不行。

验证配置里的其他字段都会作为关键字参数透传，包括 `save_dir`、`max_det`、
`eval_max_det`、`half`、`amp_dtype`、`cache` 和 `save_plots`。

## 每种任务的指标键

目标检测返回 COCO 那一族数字：

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

其中有两个是坑。`metrics/precision` 和 `metrics/recall` 是为向后兼容保留的别名：
它们装的是 mAP 50-95 和 AR@100 的值，而不是一对查准率和查全率。用带名字的键。

实例分割把上面那些 mAP 和 AR 数字当作掩码（mask）数字放在不带后缀的键下，检测框
版本放在 `(B)` 后缀下，掩码版本在 `(M)` 下再重复一遍。这个任务里的查准率和查全率
只有带后缀的形式，也就是 `metrics/precision(B)`/`metrics/recall(B)` 和
`metrics/precision(M)`/`metrics/recall(M)`，两对装的都是和 detect 一样的别名值：
`(B)` 那对是检测框 mAP50-95 和检测框 AR@100，`(M)` 那对是掩码 mAP50-95 和掩码
AR@100。

| 任务 | 键 |
|---|---|
| detect | `metrics/mAP50-95`、`metrics/mAP50`、`metrics/mAP75`，以及上面按尺寸和查全率的细分 |
| segment | 上面 detect 那些键的掩码版本（不带后缀的键就是掩码）；`precision`/`recall` 只以 `(B)`/`(M)` 的形式存在，两者的别名方式相同 |
| pose | `metrics/keypoints_mAP50-95`、`metrics/keypoints_mAP50`、`metrics/keypoints_mAP75`、`metrics/keypoints_mAP_M`、`metrics/keypoints_mAP_L`，以及对应的 `keypoints_AR` 各键 |
| obb | `metrics/mAP50-95`、`metrics/mAP50`、`metrics/mAP75`、`metrics/precision`、`metrics/recall`，外加带 `(OBB)` 后缀的副本 |
| classify | `metrics/accuracy_top1`、`metrics/accuracy_top5` |
| semantic | `metrics/mIoU`、`metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`、`metrics/SQ`、`metrics/RQ`、`metrics/PQ_things`、`metrics/PQ_stuff`、`metrics/categories` |
| depth | `metrics/abs_rel`、`metrics/rmse`、`metrics/delta1`、`metrics/delta2`、`metrics/delta3` |
| normal | `metrics/mean_angular_error`、`metrics/median_angular_error`、`metrics/within_11_25`、`metrics/within_22_5`、`metrics/within_30` |
| edge | `metrics/ODS`、`metrics/OIS`、`metrics/best_threshold` |
| restore | `metrics/PSNR`、`metrics/SSIM` |
| matte | `metrics/MAE`、`metrics/Smeasure` |
| ocr | `metrics/det_precision`、`metrics/det_recall`、`metrics/det_hmean`、`metrics/e2e_precision`、`metrics/e2e_recall`、`metrics/e2e_f1`、`metrics/rec_1-NED` |
| point | `metrics/precision`、`metrics/recall`、`metrics/f1`、`metrics/MLE`、`metrics/MAE`、`metrics/RMSE`，外加一个 mAP 扫描键 |

OBB 的 `metrics/precision` 和 `metrics/recall` 不是别名：它们是 IoU 0.50 下真实的
查准率和查全率，取在最宽松的工作点上（每一条挺过 `conf` 的预测，默认 `0.001`）。
带 `(OBB)` 后缀的副本以任务专属的名字把同样这四个值重复一遍，和上面 `(B)` 与
`(M)` 的约定一样。

`accuracy_top5` 其实是 top-`min(5, num_classes)`，所以在一个三类的数据集上它就是
top-3，每个样本都满足，因此读数是 1.0。

point 任务的扫描键是按距离阈值拼出来的，所以用默认值时它是
`metrics/mAP@[0.01:0.10]`，单阈值的那个键是 `metrics/mAP@0.01`。传入
`dist_thresholds` 会把这两个字符串都改掉。

大多数任务还会返回一个 `fitness` 键，也就是最佳检查点（checkpoint）选择默认用的
那个单一数字。目标检测、分割和 OBB 不带这个键；它们的家族按 `metrics/mAP50-95`
来选，而这个键它们的字典确实会返回。姿态既不返回 `fitness` 也不返回
`metrics/mAP50-95`；它的训练器改把 `best_metric_key` 设成
`metrics/keypoints_mAP50-95`。

## 速度键

每个验证器都会加上计时：

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

这些是整轮运行平均下来的每张图像毫秒数。它们描述的是你运行时的机器和设置，所以从
里面取出来的数字，只有连同硬件、批大小和数值精度一起报告才有意义。

## 评估后端

目标检测和分割的指标是通过一个 COCO 评估器算出来的，默认的
`faster_coco_eval=True` 会在装了 `faster-coco-eval` 包时选用 C++ 后端。没装的
时候，这次运行会回退到 pycocotools，每个进程警告一次：

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

实际跑的是哪个后端会以 `last_eval_backend` 记录在模型上，CLI 对检测类任务会在输出
里报出来。设置 `LIBREYOLO_FASTER_COCO_EVAL` 可以从环境覆盖配置里的值。

`iou_thresholds` 只在 OBB 这条路径上生效。COCO 那条路径按它自己固定的 0.50 到
0.95 扫描来评估，忽略这个值。

## 验证损失

默认情况下验证只报告精度。`val_loss=True` 会在验证批次上同时计算这个家族的训练
目标。

<code-tabs name="valloss" />

它会输出 `metrics/loss`，外加每一项对应的一个 `metrics/loss/<component>`，加权
方式和训练时完全一致，所以各分量加起来正好是总和。经过 logger 时它们显示为
`val/loss` 和 `val/loss/<component>`，而 `libreyolo monitor` 会把 `metrics/loss`
和 `train/loss` 叠在一起画。

各分量是每个家族自己的：

| 任务 | 家族 | 分量 |
|---|---|---|
| detect | `yolo9`、`yolo9_p2`、`yolo9_e2e` | `box`、`cls`、`dfl` |
| detect | `yolonas` | `cls`、`iou`、`dfl` |
| detect | `rfdetr` | `ce`、`bbox`、`giou` |
| detect | `rtdetr`、`rtdetrv2` | `vfl`、`bbox`、`giou` |
| detect | `dfine` | `vfl`、`bbox`、`giou`、`fgl`、`ddf` |
| detect | `domedetr` | `vfl`、`bbox`、`giou`、`fgl`、`ddf`、`defe_density`、`defe_reg` |
| detect | `deim`、`deimv2`、`rtdetrv4`、`ec` | `mal`、`bbox`、`giou`、`fgl`、`ddf` |
| detect | `rtmdet` | `cls`、`bbox` |
| detect | `picodet` | `cls`、`bbox`、`dfl` |
| detect | `yolox` | `iou`、`obj`、`cls`、`l1` |
| detect | `yolo7` | `iou`、`obj`、`cls` |
| point | `fomo` | `ce` |
| classify | `resnet`、`convnext`、`mobilenetv4`、`efficientnetv2` | `ce` |
| semantic | `segformer`、`lingbotvision`、`dinov2` | `sem` |
| restore | `nafnet` | `restore` |

它默认关闭，因为标签分配（target assignment）会给验证增加时间和内存开销。验证器
复用已经为精度指标产生的模型输出，而不是再跑一次前向，它在评估模型或 EMA 模型上以
`no_grad` 运行，多卡训练下则在 rank 0 上本地计算，不做任何集合通信。最佳检查点的
选择仍然按精度指标来。

有三件事它是故意不做的。它从不包含对比去噪（contrastive denoising）项，因为那些项
在前向时需要真值（ground truth），而验证的前向拿不到真值。它报告的是评估模式下的
模型，所以当一个家族的训练前向和评估前向确实不同时，比如在 BatchNorm 统计量或者
随机深度（stochastic depth）上，数字反映的是评估模式；这正是想要的那种对比。还有，
家族没有为某个任务实现它时，会在准备阶段抛出一个配置错误，而不是悄悄跳过：

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO 是那个什么都不改变的例外：它的验证器一直都在算这个损失，`val_loss=True` 只
影响它发布在哪些键下。

带数据增强的验证和验证损失不能一起用，两个都要就会抛错。

## 一次验证会写出的文件

`val()` 总会把 `config.yaml` 写进它的保存目录，没给 `save_dir` 时这个目录默认是
`runs/val/<model>_<size>_<timestamp>`。

<code-tabs name="json" />

`save_json=True` 对目标检测写出 `predictions.json`，对分割写出
`predictions_bbox.json` 和 `predictions_masks.json`。OBB 不支持它，而且会明说。

`save_plots=True` 会写进一个 `plots/` 子目录。目标检测会拿到 `box_metrics.png`、
各类别的 AP 和查全率图表、查准率-查全率曲线和置信度曲线、一张混淆矩阵，装了
OpenCV 时还有标注过的样本图像。分割会加上每一项在掩码那侧的副本，姿态有自己的一套
指标和曲线。其他验证器没有实现图表；分类、语义分割、全景分割、深度估计、法向估计、
边缘检测、图像复原、抠图、OCR、OBB 和点检测在那里什么都不写。画图失败只会警告，
绝不会中止这次运行。

## 训练过程中的验证

训练每隔 `eval_interval` 轮就在数据集的 `val` 划分上验证一次，它产生的指标就是
驱动 `best.pt` 选择、`patience` 提前停止以及每个 logger 里 `val/` 各键的东西。
EMA 打开时，验证跑在 EMA 权重上。

`eval_interval`、`patience` 和 `save_plots` 见 [超参数](/docs/train/hyperparameters)，
数字最后去了哪里见 [实验 logger](/docs/train/loggers)。

## 相关

- [数据集](/docs/train/datasets)，验证器读取的划分键和格式都在那里。
