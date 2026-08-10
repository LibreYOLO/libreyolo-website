---
title: RT-DETR
families: [rtdetr]
seo_title: "在 LibreYOLO 里使用 RT-DETR、RT-DETRv2 和 RT-DETRv4"
description: "在 LibreYOLO 里用 RT-DETR、RT-DETRv2 和 RT-DETRv4 做目标检测，RT-DETRv2 还支持旋转框。安装、预测、训练、验证、导出，权重采用 Apache-2.0 许可。"
lead: "一个为实时推理而生的检测 transformer：它解码的是一组固定的 query，而不是一张稠密网格，所以不跑 NMS。LibreYOLO 收录了它的三个版本，靠你加载的检查点（checkpoint）来区分，版本 2 还提供旋转框。"
keywords: [RT-DETR, RT-DETRv2, RT-DETRv4, "实时目标检测", DETR, "目标检测 python", "旋转框检测", OBB, DOTA]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 视频
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 版本是文件名的一部分，工厂按检查点分发，所以三个版本的加载
        # 方式完全一样
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # 库接受的任何数据源：文件、文件夹、URL、摄像头索引、RTSP 流，
        # 或者一个 .streams 列表
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: 旋转框
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 仅限版本 2。-obb 后缀选中任务，而检查点是从它自己的张量里被
        # 认出为旋转的，所以不需要传 task 参数。这些权重是 DOTA v1.0，
        # 15 个航拍类别，1024 px
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)     # (N, 5)：cx, cy, w, h, 弧度
        print(obb.xyxyxyxy)  # 同样这些行，表示成四个角点
        print(result.boxes.xyxy)  # 外接的轴对齐检测框
    - label: 旋转框，CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # coco128.yaml 会在首次使用时下载一份 128 张图片的样本；真正要
        # 跑的时候，把 `data` 指向你自己的数据集 YAML
        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # 需要 lora extra：pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: 多卡训练
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() 返回的是普通 dict，不是对象
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: 在 COCO 上验证
      language: bash
      code: |
        # coco-val-only.yaml 会取回 5000 张 val2017 图片并跳过训练集；
        # 它带着一个内嵌的下载脚本，所以除非数据集已经在本地，否则
        # 需要显式放行
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: 旋转框
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 旋转验证用旋转 IoU 来匹配，所以一个位置对、角度错的预测算
        # 漏检
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # 需要 onnx extra：pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: 旋转框
      language: bash
      code: |
        # ONNX 和 TorchScript 是旋转任务上已验证的导出目标，FP32、
        # 批大小 1、固定的 1024 乘 1024 画布
        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024
        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript imgsz=1024
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

RT-DETR 不需要任何可选 extra。它导入的一切都在基础安装里，而 `rtdetr` 这个 extra
只是一个稳定的名字，并不会在此之上添加任何东西。

```bash
pip install libreyolo
```

用 `lora=True` 做适配器（adapter）微调是个例外，它需要 `lora` extra。

```bash
pip install "libreyolo[lora]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个检测器只是一行的
改动。`conf` 和 `max_det` 过滤的是在 query 和类别上做的 top-k 解码；这里没有 NMS
步骤需要调，`iou` 会被接受但不起作用。旋转检查点原生地填充 `result.obb`，同时也用
外接的轴对齐矩形填充 `result.boxes`。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

三个版本，它们之间一共两个任务，而且尺寸代号不在同一条序列上。版本 1 按骨干给尺寸
命名，ResNet 或 HGNetv2。版本 2 只复用 ResNet 那套名字：两个 HGNetv2 尺寸版本 1
已经提供，而版本 2 在那里的结果足够接近，LibreYOLO 就不再为它们发布重复的权重。
版本 4 用的是一套纯字母序列，和版本 1 的 HGNetv2 名字撞了，所以单看一个尺寸代号
并不能确定是哪个模型。版本写在检查点的文件名里。

<benchmark-table task="detect" />

<va-embed />

版本 2 保留了版本 1 的架构和 state dict 布局，改的是可变形注意力的采样方式，这也
是为什么区分两者靠的是检查点里的元数据，而不是形状。版本 4 是另一条谱系：它复用了
D-FINE 的架构和训练器，它的权重来自把一个 DINOv3 视觉基础模型教师蒸馏进 HGNetv2
学生。在 LibreYOLO 里，`LibreRTDETRv4` 是 `LibreDFINE` 的子类，mask head 被钉死为
关闭，所以它只做检测。

### 版本 2 上的旋转框

版本 2 是唯一一个带第二个任务的版本。它支持的任务是 `detect` 和 `obb`，两者既不
共享计算图，也不共享一条尺寸序列。检测用 ResNet 那几档尺寸，在 640 px 下；旋转
检测用一套 HGNetv2 序列，n、s、m、l 和 x，在 1024 px 下，而且输入尺寸是按任务解析
的，不是按家族。一个检查点是不是旋转的，是从它自己的张量里认出来的，靠的是五坐标的
检测框 head 和版本 2 的采样参数，所以 `-obb` 权重不需要 `task` 参数就能加载进旋转
计算图，两者不匹配会直接报硬错误，而不是被悄悄按另一种方式重新解释。

已发布的文件是从 `LibreRTDETRv2n-obb.pt` 到 `LibreRTDETRv2x-obb.pt`。它们是官方的
DOTA v1.0 单尺度检查点转换成 LibreYOLO 格式的产物，15 个航拍类别，从飞机和船只
到港口和直升机，类别名已经写进检查点里。和检测那边不同，旋转任务只做推理：预测、
验证和导出都能用，而对一个旋转模型调用 `train()` 会抛异常。跟踪和测试时增强同样
不支持旋转框。[旋转框检测](/docs/tasks/oriented-detection)讲了这个任务、标注格式
和指标。

## 训练

训练从一个已发布的检查点开始。三个版本都接受 `pretrained`，然后把它丢掉，所以
`pretrained=False` 并不会给你一个随机初始化的模型。本节讲的全是检测：版本 2 的旋转
任务只做推理，而且从检测权重到它没有迁移路径，因为两者用的骨干不同。

<code-tabs name="train" />

学习率是那个必须调对的参数，而且每个版本带的是自己的默认值，不是库级别的那个。
Python 的 `train()` 签名从该版本的训练配置里读它，不传 `lr0` 时 CLI 解析出来的
也是同一个值。版本 1 和版本 2 还接受 `lr_backbone`，并按原始配方把它默认设为
`lr0` 的二十分之一；版本 4 走的是 D-FINE 的训练器，它改用 `backbone_lr_mult` 来
缩放骨干的参数组。

除非你有理由改，否则把 `imgsz` 保持在检查点的原生尺寸上。在其他尺寸下验证和预测
都能工作，但残留一个问题：一个 token 数和原生尺寸相同的矩形尺寸，仍然会复用一个
按错误宽高比构建的嵌入向量。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你训练时那种格式的数据集上测量。

<code-tabs name="val" />

上面基准测试表格里的那些行来自 LibreYOLO 的基准测试框架；表格下方的说明记录了
它们出自哪个数据集，并链接到运行记录。

旋转验证走的是同一个调用，报告的键也一样，另外还有四个带 `(OBB)` 后缀的重复项。
匹配用的是旋转 IoU，而不是外接矩形的 IoU，所以角度错了就算漏检。这个任务上
`augment=True` 会被拒绝。

## 导出

<export-matrix />

这张矩阵把整条谱系当成一页来覆盖：三个版本对某种格式意见不一致时，单元格显示的是
三者中最弱的那个，所以不管你加载的是哪个版本，这里都没有把话说满。旋转那一行只
属于版本 2。ONNX 和 TorchScript 在那里通过了验证，FP32、批大小 1、固定的 1024 乘
1024 画布；OpenVINO、TensorRT 和 ExecuTorch 能转换也能重新加载，但还没有在整个
query 集合上达到原始输出的一致，所以靠前的检测框能对到不到一个像素，尾部则会漂移。

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

文件名里先是版本，然后是尺寸，再是任务。检测权重是 `LibreRTDETR<size>.pt`、
`LibreRTDETRv2<size>.pt` 和 `LibreRTDETRv4<size>.pt`，全都在 640 px 下。旋转权重
只有版本 2 才有，并加上任务后缀，从 `LibreRTDETRv2n-obb.pt` 到
`LibreRTDETRv2x-obb.pt`，全都在 1024 px 下，训练用的是 DOTA v1.0 而不是 COCO。

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />

上面这段是作者为版本 1 和版本 2 的检测发布的内容。版本 2 的旋转权重还有第三个
上游，即采用 Apache-2.0 许可的 RiO-DETR 仓库
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR)，DOTA
检查点就来自那里；如果你用了其中之一，请引用那个项目。版本 4 是
另一组人写的另一篇论文，有自己的引用区块，见
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation)；
如果你用的是版本 4 的检查点，请引用那一个。
