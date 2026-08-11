---
title: 目标检测
seo_title: LibreYOLO 里的目标检测
description: >-
  在 LibreYOLO 里把目标检测成轴对齐的检测框：服务这个任务的家族、标注格式，以及 predict、train、validate 和 export
  调用。
lead: 目标检测定位图像里的每一个目标实例，为每一个返回一个轴对齐的矩形、一个类别标签和一个分数。任务键是 detect。
keywords:
  - 目标检测 python
  - 图片中检测物体
  - 检测框 检测
  - MIT 许可 目标检测库
  - yolo 替代方案
  - 训练自己的目标检测模型
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 换一个家族，调用不变
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按检查点分发，而且每个检测器返回的都是同一个
        # Results 对象，所以换一个家族只是改一行的事
        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy.shape)
    - label: 视频与流
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 库接受的任何输入源：文件、文件夹、URL、摄像头序号、
        # RTSP 流，或者一个 .streams 列表
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco128.yaml 会在首次使用时下载一份 128 张图的样例。真正要跑的时候，
        # 把 data 指向你自己的数据集 YAML
        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: 多卡训练
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() 返回的是一个普通 dict，不是对象
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂函数按文件后缀分发，所以导出产物的加载方式和检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreYOLO9t.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## 定义

目标检测回答每个目标在哪里、是什么。输入一张图像，每个实例输出一行：矩形的四个数、
一个类别索引和一个分数。像素形状、朝向和部件的信息一概不包含，这正是它区别于
[实例分割](/docs/tasks/instance-segmentation)、
[旋转框](/docs/tasks/oriented-detection)和
[姿态](/docs/tasks/pose-estimation)的地方。

`detect` 是这个任务的规范任务键，也是默认值：文件名不带任务后缀的检查点（checkpoint）
会作为检测器加载。

`predict()` 会填充 `result.boxes`。`.xyxy` 给出原图画布上的像素角点，`.conf` 给出
分数，`.cls` 给出指向 `result.names` 的类别索引。`.xywh`、`.xyxyn` 和 `.xywhn` 是
同一批行的派生视图，而 `.id` 在接上跟踪器之后会带上跟踪 id。遍历一个 `Boxes` 对象
得到的是单行切片，所以 `box.cls`、`box.conf` 和 `box.xyxy` 都能按单个检测使用。

## 模型

有十二个家族既能训练也能预测：[YOLOv9](/docs/models/yolov9)、
[RF-DETR](/docs/models/rf-detr)、[EdgeCrafter](/docs/models/edgecrafter)、
[RT-DETR](/docs/models/rt-detr)、[D-FINE](/docs/models/d-fine)、
[DEIM](/docs/models/deim)、[Dome-DETR](/docs/models/dome-detr)、
[YOLO-NAS](/docs/models/yolo-nas)、
[YOLOX](/docs/models/yolox)、[YOLOv7](/docs/models/yolov7)、
[RTMDet](/docs/models/rtmdet) 和 [PicoDet](/docs/models/picodet)。YOLOv9 和
RF-DETR 是两个旗舰家族，新特性会先落到它们上面。RF-DETR 需要自己的 extra，
`pip install "libreyolo[rfdetr]"`；其余的在基础包上就能跑。

另有十一个家族可以预测、验证和导出，但它们的 `train()` 会抛出
`NotImplementedError`：[LW-DETR](/docs/models/lw-detr)、
[DETR](/docs/models/detr)、[Deformable DETR](/docs/models/deformable-detr)、
[DINO-DETR](/docs/models/dino-detr)、[Faster R-CNN](/docs/models/faster-rcnn)、
[Mask R-CNN](/docs/models/mask-rcnn)、[FCOS](/docs/models/fcos)、
[RetinaNet](/docs/models/retinanet)、[SSD](/docs/models/ssd)、
[CenterNet](/docs/models/centernet) 和
[EfficientDet](/docs/models/efficientdet)。

Darknet 一脉的 [YOLOv1](/docs/models/yolov1)、
[YOLOv2](/docs/models/yolov2)、[YOLOv3](/docs/models/yolov3) 和
[YOLOv4](/docs/models/yolov4) 作为冻结的展品保留下来：预测、验证和导出可用，
训练不可用。

另有一组模型的类别列表是在运行时给出的，而不是来自检查点，所以它们能检测训练中
从未见过的名称：
[Grounding DINO](/docs/models/grounding-dino)、[OWLv2](/docs/models/owlv2)、
[OMDet-Turbo](/docs/models/omdet-turbo) 和 [OV-DEIM](/docs/models/ov-deim)，
外加视觉语言家族
[Florence-2](/docs/models/florence-2)、[Kosmos-2](/docs/models/kosmos-2)、
[Qwen3-VL](/docs/models/qwen3-vl)、[SmolVLM2](/docs/models/smolvlm2)、
[InternVL3](/docs/models/internvl3)、[LFM2-VL](/docs/models/lfm2-vl)、
[LocateAnything](/docs/models/locate-anything)、
[SenseNova-Vision](/docs/models/sensenova-vision) 和
[LibreMODUS](/docs/models/libremodus)。这些通过它们自己的工厂函数和 extra 加载；
每个模型页上都写着确切的调用方式。

## 预测

权重会在首次使用时从 Hugging Face 下载，并缓存到本地。

<code-tabs name="predict" />

`conf` 设置置信度阈值，`max_det` 限制输出的行数上限。`iou` 是 NMS 阈值，所以它只对
会跑 NMS 的家族有影响；RF-DETR 和端到端的 YOLOv9 head 解码的是一组固定数量的预测，
会忽略它。输入源、流式处理和结果处理见[预测](/docs/predict)。

## 数据集格式

每张图像对应一个 `.txt` 标注文件，把图像路径里的 `images` 换成 `labels`、再改掉
扩展名就能找到它。

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

每一行正好是五个字段，一个类别索引，后面跟一个归一化的中心点加尺寸的框：

```text
<class_id> <cx> <cy> <w> <h>
```

坐标是 `[0, 1]` 之间的浮点数，相对于原图的宽和高。`w` 和 `h` 必须为正。标注文件缺失
或为空表示这张图里没有目标。行里不带置信度，也不带跟踪 id。

YAML 指明各个划分（split）和类别：

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` 和 `val` 可以是图像目录、图像列表 `.txt` 文件，或者两者的列表。`nc` 是可选
的，出现时必须和 `names` 对得上。原生的 COCO JSON 也可以：加一个 `annotations` 映射，
把划分名映射到 JSON 文件，此时划分路径给出的就是图像根目录。当 `names` 存在时，它
定义了标签 id，所以 JSON 里的类别名必须和它一致。

## 训练

<code-tabs name="train" />

`epochs`、`imgsz`、`batch` 和 `lr0` 是最先要动的参数。`lr0` 是那个不能跨家族照搬的：
一个卷积检测器能接受的学习率会让 transformer 检测器发散，所以要从模型页上取这个值，
而不是从另一个家族的例子里取。一个家族也可能干脆忽略某个参数，它的页面上列出了是
哪些。数据集、数据增强、多卡训练和日志记录见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的普通字典，是在数据集 YAML 里 `val` 指定的划分
上用 COCO 评测算出来的。

<code-tabs name="val" />

`metrics/mAP50-95` 是在 0.50 到 0.95 的 IoU 阈值上取平均的 mean average precision，
也是最主要的那个数。`metrics/mAP50` 和 `metrics/mAP75` 是单阈值的版本。
`metrics/mAP_small`、`metrics/mAP_medium` 和 `metrics/mAP_large` 把同一个平均值按
目标面积拆开，而 `metrics/AR1`、`metrics/AR10`、`metrics/AR100`、
`metrics/AR_small`、`metrics/AR_medium` 和 `metrics/AR_large` 是对应的平均查全率
数字。`metrics/AR_max_det` 和 `metrics/max_det` 记录这次运行用的检测数上限。

在这个任务上，`metrics/precision` 和 `metrics/recall` 要仔细读。它们是为了向后兼容
保留下来的别名，而不是一个工作点：`metrics/precision` 存的值和 `metrics/mAP50-95`
相同，`metrics/recall` 存的值和 `metrics/AR100` 相同。把它们当成一对查准率-查全率
来画图，等于把同一个数报了两遍。还有四个键会在 `(B)` 后缀下再重复一遍，`B` 表示
box，这样在同时也预测掩码的模型上，检测的键读起来是一样的：`metrics/mAP50-95(B)`、
`metrics/mAP50(B)`、`metrics/precision(B)` 和 `metrics/recall(B)`。

## 导出

<code-tabs name="export" />

导出的产物会按文件后缀通过 `LibreYOLO()` 加载回来，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。各个家族支持的格式不同；每个模型页上
的矩阵是从已验证的集合生成的，而不是手写的。格式、它们的 extra 和限制见
[导出与部署](/docs/export)。
