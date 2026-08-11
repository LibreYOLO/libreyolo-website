---
title: 许可证
seo_title: LibreYOLO 许可证：代码与权重
description: LibreYOLO 自己的代码采用 MIT 许可。内置的上游代码和已发布的检查点各自带着自己的许可证，其中有几个是非商用的。
lead: LibreYOLO 里装着三样各自单独许可的东西：它自己的代码、内置进某个模型家族的上游代码，以及预训练检查点。它们的许可证往往并不相同。
keywords:
  - libreyolo 许可证
  - mit 许可的视觉库
  - 模型权重 非商用
  - 检查点许可证
  - apache-2.0 目标检测
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## LibreYOLO 自己的代码

这个库采用 MIT 许可。它覆盖 Python API、CLI、训练器、验证器和导出器、数据集加载器，
以及 `weights/` 下的转换脚本。你可以把它用在商业或闭源产品里，在你再分发的每一份副本里
保留版权行和许可证文本，义务到此为止。

这份许可到代码为止。
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE) 文件
说得很直白：

> 那些许可证各不相同，也并非都是宽松许可：部分已发布的权重是非商用的，或者受到其他
> 限制，本 MIT 许可证并不延伸到它们身上。选一个模型，就是选它的许可证。

## 上游代码，按家族

大多数家族是对已发表研究的移植，其中有几个直接内置了上游源码。内置进来的文件保留它原本的
版权头和原本的许可证。MIT 不会覆盖它，LibreYOLO 也不会给任何人的作品重新许可。出现得最多的
是 Apache-2.0 和 BSD-3-Clause 这两个。

Apache-2.0 覆盖 DETR 这条线和大部分 transformer 相关的工作：Meta AI（FAIR）的 DETR、
商汤（SenseTime）的 Deformable DETR、百度的 LW-DETR、Leilei Wang 和合作者的 OV-DEIM、
LibreYOLO 从 Hugging Face Transformers 移植的 SegFormer 实现、PaddlePaddle Authors 的
PP-OCRv5、苏黎世联邦理工学院（ETH Zurich）计算机视觉实验室的 SwinIR，以及字节跳动
Seed 的 Depth Anything 3。它还覆盖派生自 timm 的那些分类器，timm 出自 Ross Wightman
和 timm 贡献者之手，其中包括 ResNet、DeiT、EfficientNetV2、MobileNetV4 和 Swin，
它们的模块名与 timm 保持一致，因此 timm 的 ImageNet 张量可以原样加载。

BSD-3-Clause 覆盖所有派生自 torchvision 的部分：Faster R-CNN、Mask R-CNN、FCOS、
RetinaNet、SSD300、AlexNet、VGG、FCN 和 DeepLabv3。

MIT 覆盖的是更小的一组，包括旷视（Megvii）的 NAFNet、Xingyi Zhou 的 CenterNet，以及由
原作者 Kin-Yiu Wong 和 Hao-Tang Tsui 在 MultimediaTechLab 重新发布的 YOLOv7。YOLOv1
到 YOLOv4 这几个家族复现的是 Darknet 项目里的架构，作者是 Joseph Redmon，YOLOv4 那部分
则是 Alexey Bochkovskiy。Darknet 属于公有领域，所以它们完全不带任何义务。

有一个随包附带的子树用的不是开源许可证。DEIMv2 家族附带了来自 Meta Platforms 的 DINOv3
骨干代码，依据的是 DINOv3 License Agreement，一份自定义的非 OSI 许可证。再分发这段代码
就意味着要随附一份该协议的副本，而协议禁止将其用于受 ITAR 管制的活动、军事或战争用途、
核工业、间谍活动以及武器研发。这些条款只约束那一个子树。

仓库里有两个文件给出了完整的图景。
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE) 列出
每一个随包附带的第三方子树，以及它的路径、许可证文件和上游来源。
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
列出 LibreYOLO 所派生的上游项目，并完整复制每一份许可证文本。

## 权重，按检查点

包里不附带任何预训练权重文件。已发布的检查点（checkpoint）放在 Hugging Face 上的
[LibreYOLO 组织](https://huggingface.co/LibreYOLO)下，每个仓库都带着自己的 `LICENSE`
和署名信息，对应权重所出自的那个项目。

那个仓库才是这些条款的权威来源。不是这一页，不是模型页面，也不是源码树里的摘要。文件如何
命名、从哪里下载，见[检查点与权重](/docs/weights)。

许可证在不同家族之间不一样，在同一个家族内部的不同文件之间也不一样。后者的两个例子：

- YOLO9 的 COCO 检查点采用 MIT 许可。在 VisDrone2019-DET 上训练的
  `LibreYOLO9P2s-visdrone.pt` 采用 CC BY-NC-SA 3.0，那是非商用的。
- RF-DETR 的检测检查点采用 Apache-2.0。旋转框检查点采用 CC BY 4.0，因为它们是在一个以
  CC BY 4.0 发布的 Roboflow Universe 数据集上微调出来的，权重把那个数据集的署名要求
  一路带了下来。

跨家族来看，范围还要更宽，有几个已发布的检查点不能用在商业产品里：

- SegFormer 是这两层之间分得最清楚的例子。它的实现是对 Hugging Face Transformers 代码的
  Apache-2.0 移植。已发布的 ADE20K 检查点则是从英伟达在 NVIDIA Source Code License 下的
  发布转换而来，那份许可证允许再分发，但把使用限制在非商业的研究或评估，并把这条限制一路
  带进衍生作品。这些检查点不在 LibreYOLO 的宽松许可覆盖范围之内。
- OV-DEIM 的检查点采用 CC BY-NC 4.0，这一点得到了上游作者的确认。每次预测还会加载
  Apple 的 MobileCLIP-B(LT) 文本塔，它的许可证把使用限制在研究用途，比检查点自身的条款
  更严。
- SenseNova-Vision 的代码采用 Apache-2.0，权重采用 CC BY-NC 4.0。加载器会在每次自动下载
  之前打印那条非商用提示。

有些家族在 LibreYOLO 这边根本没有托管任何检查点，它们的页面会在 Weights 那一行里说明这一
点。SAM 3 在 Hugging Face 上按 Meta 自定义的 SAM License 做了访问限制，直接从 Meta 下载。
MiDaS 的发布资源是从官方 URL 获取并做哈希校验的，而不是转托管。Dome-DETR 给的是上游链接，
因为它的模型卡在元数据里没有写许可证，而正文一边声称 Apache-2.0，一边又把使用限制在学术
研究，两者对不上。TEED 和 DexiNed 的架构采用 MIT 许可，但作者发布的检查点是在 BIPED 上
训练的，那个数据集的条款是非商用的，所以 LibreYOLO 既不打包也不自动下载它们。

有几个 torchvision 检查点自己没有许可证文件。LibreYOLO 按发布项目所用的许可证来对应它们，
在每张模型卡上说明这个依据是推定出来的、而不是逐个检查点授予的，并重复 torchvision 自己的
提醒：预训练模型的条款可能来自训练数据。

## 查清某一个模型的条款

模型页面的头部带着一行 **Licenses**，形式是 `Code X, weights Y`，它会向下链接到该页的
Licensing 小节。那一节会列出原始工作及其作者、上游许可证、上游来源、LibreYOLO 的代码
许可证、权重，以及对这些条款允许做什么的一份解读。同一页上的 Checkpoints 表格有一列
**Weights license**，每个已发布文件占一行，所以条款混杂的家族会逐个文件把它们摆出来。

这一切都由校验这个库时所用的同一份数据渲染出来，这也是本页不再用一张表格重复一遍的原因。
手敲的许可证矩阵在一个版本之内就会出错，而在这件事上出错代价很高。

在源码树里，与之对应的是：随包代码看 `NOTICE`，上游项目及其许可证文本看
`THIRD_PARTY_NOTICES.txt`，已发布检查点的按家族摘要看
[`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)。

然后去查你正要下载的那个文件对应的 Hugging Face 仓库。它才是权威的，而且它可能变，文档
页面不一定跟着变。

## 商用

代码很少是问题所在。MIT、Apache-2.0 和 BSD-3-Clause 都允许商用和闭源使用。它们都要求你在
再分发的副本里保留各自的许可证文本和署名声明，Apache-2.0 还额外授予一份专利许可，而且没有
哪一个会对你自己的应用代码提出条件。

产品真正卡住的地方是检查点。不管外围的代码有多宽松，非商用的检查点仍然是非商用的，转换文件
也不会改变它适用的条款，`weights/LICENSE_NOTICE.txt` 里就是这么直说的。从一个受限检查点
构建出来的 ONNX 或 TensorRT 产物会继承那份限制。

如果一份许可证像 NVIDIA Source Code License 那样把限制带进衍生作品，那么微调也躲不掉。用
你有权使用的数据从头训练同一个架构则可以：代码是宽松许可的，所以你自己训练出来的模型就是
你的，预训练检查点的条款从来不会进到里面。SegFormer 页面针对它自己的权重把这一点讲得很
清楚；你打算发布哪个家族，就去读那一页的 Interpretation 行。

许可证的问题要在挑模型的时候定下来，而不是在发布的时候，并且要读你实际下载的那个文件上的
条款，因为一个家族里有一个宽松许可的检查点，旁边就可能摆着一个受限的。

## 不构成法律意见

本页描述的是其中涉及的许可证。它是一份描述，不是法律意见，也不构成任何担保。如果这个答案在
商业上很重要，请自己去读那些许可证，并咨询你自己的法律顾问。
