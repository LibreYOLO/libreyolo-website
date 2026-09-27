---
title: 旋转框检测
seo_title: LibreYOLO 中的旋转框检测
description: 在 LibreYOLO 中检测旋转目标：提供旋转框的模型家族、四角点标注行，以及预测、训练、验证和导出的调用方式。
lead: 旋转目标检测用一个旋转矩形而不是轴对齐矩形来定位每个实例，这样倾斜的目标会被紧紧框住，而不是被一个装满背景的框圈起来。任务键是 obb。
keywords:
  - 旋转框检测
  - 旋转目标检测
  - obb 检测 python
  - dota 数据集
  - 航拍目标检测
  - 旋转 iou
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        # 需要 rfdetr 附加依赖：pip install "libreyolo[rfdetr]"
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名里的 -obb 后缀会选定任务，所以不需要传
        # task 参数
        model = LibreYOLO("LibreRFDETRs-obb.pt")
        result = model(SAMPLE_IMAGE, save=True)

        obb = result.obb
        print(obb.xywhr)   # (N, 5)：中心 x、中心 y、宽、高、弧度
        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 用角点代替角度
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # (N, 4, 2) 以像素为单位的角点
        print(obb.xyxyxyxyn.shape)   # 同上，归一化后的
        print(obb.xyxy.shape)        # (N, 4) 外接的轴对齐框
    - label: 更小的检查点
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # DOTA v1.0 权重，1024 px 下的 15 个航拍类别。旋转框计算图
        # 由检查点自带的张量识别出来，所以不需要 task 参数
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)
        print(result.names)   # plane、ship、harbor、helicopter，以及另外 11 个
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # 从已发布的旋转框权重继续训练。data 必须指向一个

        # 标注行带四个角点的数据集

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: 从检测权重开始
      language: bash
      code: |
        # 检测权重不预测角度，所以这是一次显式的迁移。
        # 传入 task=obb 才是授权这次迁移的动作
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() 返回的是普通 dict，不是对象
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # 这里经过验证的目标格式是 ONNX 和 TorchScript，FP32、

        # 批大小为 1、固定 1024×1024 的画布

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载起来和检查点
        # 一样，返回同样的 Results 对象
        model = LibreYOLO("LibreRFDETRs-obb.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## 定义

旋转框检测在检测的基础上多加了一个数：角度。每个实例得到一个旋转矩形、一个类别和一个分数。换来的是贴合度。倾斜 45 度的船、仓库屋顶、一排停放的卡车：给它们中的任何一个套上轴对齐的框，框里大半都是背景，而且哪怕目标本身并不相交，相邻的两个框也会重叠。正因如此，这个任务在航拍影像和文档版面里是标配，它的参考数据集也就是 DOTA。

`obb` 是规范的任务键，检查点（checkpoint）文件名里的 `-obb` 后缀会选中它，所以加载已发布的权重时不需要写 `task=`。

`predict()` 会填充 `result.obb`。`.xywhr` 是规范的 `(N, 5)` 形式：中心 x、中心 y、宽、高，以及一个用弧度表示的角度，给出宽边绕中心的旋转量。`.conf` 和 `.cls` 装的是分数和指向 `result.names` 的类别索引，跟踪时 `.id` 装的是跟踪 id。`.xyxyxyxy` 把每一行转成四个角点，形状为 `(N, 4, 2)` 的像素坐标，`.xyxyxyxyn` 把这些角点归一化，`.xyxy` 给出外接的轴对齐框——下游代码只认识矩形时就用它。`result.boxes` 同样会被填上，装的是轴对齐的形式。

## 模型

有两个模型家族提供这个任务，选哪一个取决于你需不需要训练。

[RF-DETR](/docs/models/rf-detr) 是能训练的那个。它对旋转框可以预测、训练、验证和导出，并且发布了 n、s、m、l 四种尺寸的旋转框检查点。它需要自己的附加依赖 `pip install "libreyolo[rfdetr]"`，权重许可和来源写在它的模型页上。

在围绕这些检查点做规划之前，先读下面那一节，弄清它们实际预测的是什么。

[RT-DETRv2](/docs/models/rt-detr) 是带航拍权重的那个。它发布了从 `LibreRTDETRv2n-obb.pt` 到 `LibreRTDETRv2x-obb.pt` 的一系列权重，是官方 DOTA v1.0 单尺度检查点转成 LibreYOLO 格式的产物，覆盖 DOTA 在 1024 px 下的 15 个类别。除了基础包之外它不需要任何附加依赖，旋转框计算图由检查点自带的张量识别出来，预测、验证以及 ONNX 和 TorchScript 导出都支持。训练不支持：在这个家族上旋转框任务只能做推理，`train()` 会抛异常，也没有从它的检测权重迁移过来的路径，因为那些权重用的是另一套骨干。跟踪和测试时增强对旋转框同样不可用。

所以：要开箱即用的 DOTA 类别，用 RT-DETRv2。要用你自己的旋转框标注，用 RF-DETR。

## 预测

权重会在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

跑 RF-DETR 已发布的检查点之前，先弄清它们是什么。尽管 DOTA 是这个任务的参考基准测试，这些权重并不是在它上面训练的。四个权重全都是从 RF-DETR 的检测权重初始化，然后在一个来自 Roboflow Universe 的无人机航拍数据集上微调出来的，带六个车辆类别：bike、bus、car、other_vehicle、taxi 和 truck。它们的模型卡把它们描述为开发用权重，是在验证旋转框训练支持时产出的，并说明不应该把它们当成生产权重或基准测试的官方权重。

实际用起来，这意味着它们是从上往下看的车辆旋转框的一个可用起点，也可以用来验证你的流水线能端到端跑通。换成别的任何领域，就得用你自己的旋转框标注来训练；而对于 DOTA 出名的那些航拍类别，真正在那份数据上训练过的是 RT-DETRv2 的检查点。`conf` 和 `max_det` 塑造输出的方式和检测任务里一样。图像来源、流式输入和结果处理见[预测](/docs/predict)。

## 数据集格式

目录结构就是检测的那一套：每张图像一个 `.txt` 标注文件，把图像路径里的 `images` 换成 `labels` 再改掉扩展名就能找到。

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

一行正好九个字段，一个类别索引，后面跟着按顺序排列的四个角点：

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

这四个点是 `[0, 1]` 区间内归一化的浮点数，必须构成一个非退化的旋转矩形。标注文件里不存角度：加载器从角点推导出规范的 `xywhr`。解析器默认是严格的，会拒绝超出范围的坐标；而数据集和验证的读取环节，对其他方面都合法的裁剪边界标注可能先裁剪到 `[0, 1]`，之后仍会拒绝退化的框。

行解析是区分任务的。九个字段只有在 `obb` 模式下才表示一个旋转框；在 `segment` 模式下，同样一行会被读成一个四点多边形。

YAML 就是检测的那份 YAML：

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

原生的 COCO JSON 也能加载，用一个把划分名映射到 JSON 文件的 `annotations` 字段。标注按优先级顺序读取：一个装着八个像素坐标角点的 `obb` 字段，一个装着 `[cx, cy, w, h, angle]`、角度以弧度表示的 `obb` 字段，一个重新拟合到最小面积矩形的 `segmentation` 多边形或 RLE，或者一个普通的 COCO `bbox`——它会被当作轴对齐矩形，再规范化成 `xywhr`。

规范的行解析器是 `libreyolo.data.parse_yolo_obb_label_line`。

## 训练

<code-tabs name="train" />

在这个任务上训练就意味着用 RF-DETR。默认情况下，训练是从一个已发布的 `-obb` 检查点继续的。从检测权重起步是一次刻意的迁移：那些权重不预测角度，传入 `task=obb` 才是授权这次替换的动作。和这个家族的其他任务一样，把 `lr0` 保持在 `1e-4` 或更低。RT-DETRv2 的旋转框检查点无法微调；要么照原样使用，要么在你自己的标注上训练一个 RF-DETR 模型。数据集、数据增强、多卡训练和日志器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的普通字典。匹配用的是旋转 IoU，在旋转矩形之间计算，而不是在它们外接的轴对齐框之间计算，所以一个位置对、角度错的预测会被算作漏检。

<code-tabs name="val" />

`metrics/mAP50-95` 是在 0.50 到 0.95、步长 0.05 的一系列 IoU 阈值上平均出来的平均精度均值，也是最主要的那个数字。和检测走的 COCO 路径不同，这个任务会遵循验证配置里的 `iou_thresholds`，所以这个扫描范围可以改。`metrics/mAP50` 和 `metrics/mAP75` 是单阈值的版本。`metrics/precision` 和 `metrics/recall` 是 IoU 0.50 下真正的查准率和查全率，读的是最宽松的工作点：每一个通过置信度阈值的预测都会被计入，而验证时这个阈值默认是 0.001。因此调高 `conf` 会让它们变化，而用上整条查准率-查全率曲线的 mAP 数字则原地不动。其中四个会以 `(OBB)` 后缀重复一遍，即 `metrics/mAP50-95(OBB)`、`metrics/mAP50(OBB)`、`metrics/precision(OBB)` 和 `metrics/recall(OBB)`，调用方就是靠它在同一张表里区分旋转框结果和轴对齐结果的。`metrics/mAP75` 没有带后缀的孪生项。

有两个选项在这个任务上什么也不做。`save_json` 和 `save_plots` 会被接受，并打一条警告日志：旋转框的预测结果导出和验证图表尚未实现。

## 导出

<code-tabs name="export" />

导出的产物会按文件后缀通过 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine` 文件的行为和检查点一样，返回同样的 `Results`。同一个家族上，格式覆盖范围随任务而不同，模型页上的支持矩阵由验证过的集合生成，并会说明某个目标格式不可用的原因。各种格式、它们的附加依赖和限制见[导出与部署](/docs/export)。
