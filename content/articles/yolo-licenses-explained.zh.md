---
title: "YOLO 许可全解：YOLOv1 到 YOLO26 的许可证与商用（2026 指南）"
description: "2026 年完整的 YOLO 许可版图：YOLOv1 到 YOLOv13、YOLO26、YOLO-World 和 YOLOE，涵盖原始仓库与宽松许可的重写版，附论文和 GitHub 链接，并说明哪些真正可以用在商业产品中。"
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "YOLOv9 可以免费商用吗？"
    a: "取决于你用的是哪个仓库。论文仓库（WongKinYiu/yolov9）采用 GPL-3.0。同一实验室还在 MultimediaTechLab/YOLO 另行发布了一个采用 MIT 许可的 YOLOv9 和 YOLOv7 实现，是在商业用户要求提供宽松许可选项之后编写的。它是一次重写，而不是把那些 GPL 文件改换成新的许可。代码的许可是清楚的；预训练权重在两个仓库中都没有单独的许可声明，因此请把权重的来源当作一个单独的问题来对待。"
  - q: "哪些 YOLO 版本可以免费用于闭源商用？"
    a: "至少可以通过一个宽松许可仓库使用的有：YOLOv1 到 YOLOv4（Darknet 原版，公有领域）、YOLOv7 和 YOLOv9（该实验室自己的 MIT 重写版）、YOLOX 和 PP-YOLOE（Apache-2.0），以及 LibreYOLO 中的 MIT 重新实现。截至 2026 年 7 月，YOLOv5、v6、v8、v10、YOLO11、v12、v13、YOLO26、YOLO-World 和 YOLOE 都没有宽松许可的实现。在依赖这些信息之前，请自己查看当前的 LICENSE 文件：许可会变化，而且这是一般性信息，不是法律意见。"
  - q: "YOLOv10 采用什么许可？"
    a: "AGPL-3.0。YOLOv10 是清华大学发布的学术成果，但它的代码基于 Ultralytics 代码库构建，因此继承了 AGPL-3.0。目前没有宽松许可的重新实现。"
  - q: "YOLOv12 和 YOLOv13 采用什么许可？"
    a: "两者都是 AGPL-3.0，已在各自的 LICENSE 文件中确认。与 YOLOv10 一样，它们是学术论文，其参考代码基于 Ultralytics 仓库构建，因此无论论文作者是谁，AGPL 都会延续下来。把 YOLOv13 列为 Apache-2.0 的第三方页面是错误的。"
  - q: "Ultralytics 是什么时候把 YOLO 改为 AGPL-3.0 的？"
    a: "2023 年 4 月 14 日，而不是广为流传的 2022 年。ultralytics/yolov5 中的 LICENSE 文件在历史上只被修改过三次，而改为 AGPL 的那次提交（34cf749，PR #11359）日期为 2023-04-14。2022 年的最后一个版本 YOLOv5 v7.0 仍以 GPL-3.0 发布。ultralytics/ultralytics 仓库在同一天更换了许可。因此 YOLOv8 于 2023 年 1 月以 GPL-3.0 发布：PyPI 上的 8.0.0 到 8.0.76 声明的都是 GPL-3.0，而 8.0.80（2023 年 4 月 16 日）是第一个声明 AGPL-3.0 的版本。"
  - q: "YOLO26 可以免费商用吗？"
    a: "闭源产品不行。YOLO26 以 AGPL-3.0 发布，替代方案是付费的企业许可，与 YOLOv8 和 YOLO11 的条款相同。"
  - q: "原版 Darknet YOLO 真的属于公有领域吗？"
    a: "是的。Joseph Redmon 的 Darknet LICENSE 写着「Darknet is public domain. Do whatever you want with it.」（Darknet 属于公有领域，随你怎么用。）AlexeyAB 的 YOLOv4 分支也带有同样的公有领域文本。问题在于，人们实际使用的 PyTorch 移植版带有各自的许可，从 Apache-2.0 到 AGPL-3.0，再到完全没有许可。"
  - q: "YOLO-NAS 的预训练权重可以商用吗？"
    a: "不可以。super-gradients 的代码是 Apache-2.0，但官方 YOLO-NAS 权重采用单独的许可，其中写明你「may not use the Software for any commercial use, including in connection with any models used in a production environment.」（不得将本软件用于任何商业用途，包括与生产环境中使用的任何模型相关的用途。）"
  - q: "神经网络权重会继承训练代码的许可吗？"
    a: "尚无定论。Ultralytics 表示 AGPL-3.0 涵盖「the training code and the models produced by that training code」（训练代码以及由该训练代码产出的模型），而作为版权持有者，由他们为自己发布的内容设定条款。至于法院是否会认同你自己训练的权重属于训练代码的衍生作品，这一点从未经过检验。请把已发布的 AGPL 权重视为有权利负担，并在把它们加载到宽松许可的重新实现中时保持谨慎。"
  - q: "为什么同一个 YOLO 的不同实现可以有不同的许可？"
    a: "版权保护的是表达，而不是思想（17 U.S.C. 102(b)）。论文中描述的架构可以被独立实现，并按作者选择的任何方式许可。你不能做的是复制 GPL 或 AGPL 源文件并为其更换许可。注意，这只是一个版权层面的论点：独立实现不能作为针对专利的抗辩。"
---

每年都有更多的 YOLO 出现，每年许可问题都会被重新提起，而且通常是在做产品决策前五分钟。令人困惑的是，「YOLO」并不是一个项目。它是二十多个来自不同作者的模型家族共用的品牌名，而这里有一个大多数许可指南都忽略的细节：**许可属于仓库，而不属于模型。** 同一个 YOLO 版本可以同时以 GPL 仓库和 MIT 仓库的形式存在，出自同一批作者。YOLOv9 就是如此，而这改变了整个决策。

本指南逐个仓库梳理整个版图，附上每篇论文和每个代码库的链接，内容截至 2026 年 7 月。下文的每一项许可都对照了实际仓库中的实际 LICENSE 文件进行核查，因为关于 YOLO 许可的现有文字（包括在这个问题上排名最靠前的那些指南）有数量惊人的部分是错的。

如果你只想弄清 Ultralytics 的 AGPL 问题，我们在 [YOLO 可以免费商用吗？](/articles/yolo-commercial-license) 一文中做了深入讨论。本文是完整的版图。

**开始之前，先做两点披露。** 第一，我们维护 LibreYOLO，这是一个采用 MIT 许可的库，与下文中的若干项目存在竞争关系，本文最后一节讲的就是它。这是你核查我们工作的理由，而不是盲目相信的理由，这也是为什么这里的每一项许可说法都链接到了一手来源。第二，本文是关于已发布许可文本和公开声明的一般性信息，截至上述日期。它不是法律意见，也不能替代你所在司法辖区的律师。许可和维护者的立场都会变化。在做出交付决定之前，请阅读任何仓库当前的 LICENSE 文件。

## YOLO 许可对照表

「宽松路线」指：一个实现该模型的仓库，你可以把它用于闭源商业产品，既不必开源你的应用，也不必为许可付费。

| 模型 | 年份 | 原始代码 | 宽松路线 | 论文 |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet)（公有领域） | 原版本身 | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet)（公有领域） | 原版本身 | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet)（公有领域） | 原版；当心 AGPL 的 PyTorch 移植版 | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)（公有领域） | 原版；[pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)（Apache-2.0） | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5)（AGPL-3.0） | 无 | 无论文 |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6)（GPL-3.0） | 无 | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)（GPL-3.0） | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)（MIT，同一实验室） | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0；发布时为 GPL-3.0，见下文） | 无（Keras 移植版已被放弃，见下文） | 无论文 |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)（GPL-3.0） | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)（MIT，同一实验室） | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)（AGPL-3.0） | 无 | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0） | 无 | 无论文 |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)（AGPL-3.0） | 无 | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13)（AGPL-3.0） | 无 | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0） | 无 | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)（Apache-2.0） | 原版本身 | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)（Apache-2.0） | 原版，取自 PaddleDetection（不是 PaddleYOLO） | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients)（代码为 Apache-2.0，权重为非商用） | 代码可以，官方权重不行 | 无论文 |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)（GPL-3.0） | 无 | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)（AGPL-3.0） | 无 | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo)（GPL-3.0，2024 年起停滞） | 无 | 无论文 |

按列逐一看下来，有一点显而易见：版本号告诉不了你任何关于许可的信息。仓库才能。

## 同一模型，两种许可：YOLOv9 的案例

YOLOv9 最清楚地证明了「YOLOvN 是什么许可？」是个错误的问题，而这段时间线值得准确讲述，因为这是 YOLO 历史上唯一一次社区压力真正改变了结果。

- **2024 年 2 月 18 日**：[WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) 随[论文](https://arxiv.org/abs/2402.13616)一同公开。
- **2024 年 2 月 22 日**：一位用户提交 issue #10，询问许可是什么。Chien-Yao Wang（WongKinYiu）回复「I think it should be GPL3」，并在四天后添加了 GPL-3.0 文件。
- **2024 年 2 月 26 日**：第二位用户提交了 [issue #82「An Apache/MIT rewrite」](https://github.com/WongKinYiu/yolov9/issues/82)。在接下来的两周半里，十来个商业用户纷纷加入。
- **2024 年 3 月 14 日**：Wang 在该讨论串中发帖：「*Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training.*」

那个仓库最初名为 `yolov9mit`，如今就是 **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)**：采用 MIT 许可，版权声明为「Kin-Yiu, Wong and Hao-Tang, Tsui」，并自称是「*the official implementation of YOLOv7 and YOLOv9, YOLO-RD.*」重写的大部分工作并不是由讨论串里的志愿者完成的，而是由 Wang 的同实验室同学 Hao-Tang Tsui 完成，该仓库大约 90% 的提交出自他之手。

所以，取决于你克隆的是哪个仓库，YOLOv9 同时是「不能交付」和「可以交付」的。同样的架构，同一个实验室，两种许可。

**在你把产品押在它身上之前，有三点注意事项是该仓库首页不会告诉你的：**

1. **它仍在成熟中。** 2026 年 2 月的一个[未关闭 issue](https://github.com/MultimediaTechLab/YOLO/issues/231) 报告称，随仓库发布的检查点（checkpoint）无法复现论文中的 COCO 数据，参数量也明显多于论文所称。分割和关键点训练[仍未实现](https://github.com/MultimediaTechLab/YOLO/issues/232)。自 2025 年 12 月的 v1.0 标签以来，main 分支上没有合并任何内容。
2. **权重没有单独的许可声明。** 按惯例，该仓库的 MIT LICENSE 涵盖其发布资产，而且技术证据表明这些检查点是由该项目自己训练的，而不是从 GPL 仓库复制来的：这些文件在大小和参数量上都与 GPL 仓库的检查点不同。但没有任何文件写明这些 .pt 文件是 MIT 许可，而社区提出的一项[针对 AGPL 污染进行正式审计的请求](https://github.com/MultimediaTechLab/YOLO/issues/51)仍未关闭。这是文书上的空白：没有人声称「已验证无污染」，也没有人指称存在问题。
3. **它是一个不同的代码库**，不能直接替换 GPL 仓库的脚本。

这些都不意味着它不可用。它意味着这是一个需要评估的项目，而不是一个打个勾就完事的选项。

## 第一阶段：公有领域时代（YOLOv1 到 YOLOv4）

Joseph Redmon 在他的 Darknet 框架中发布了 [YOLOv1](https://arxiv.org/abs/1506.02640)、[YOLOv2](https://arxiv.org/abs/1612.08242)（以 YOLO9000 论文的形式发表）和 [YOLOv3](https://arxiv.org/abs/1804.02767)，而其 [LICENSE 文件](https://github.com/pjreddie/darknet/blob/master/LICENSE) 以只有三行而闻名：

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

这是一份真正的公有领域奉献声明（public domain dedication）。[YOLOv4](https://arxiv.org/abs/2004.10934) 由 Alexey Bochkovskiy 在一个 [Darknet 分支](https://github.com/AlexeyAB/darknet) 中维护，带有*同样*的文件，而不是好几份许可指南所说的 Unlicense。GitHub 自己的分类器也识别不了它，报告为「Other」；如果你的合规工具读取这个字段，这一点值得了解：扫描器会把它标记为未识别，而不是干净的宽松许可，尽管这段文字已经宽松得不能再宽松了。

Darknet 本身也还没有完全消亡。Redmon 的仓库自 2022 年以来一直闲置，但 AlexeyAB 的 README 现在指向 [hank-ai/darknet](https://github.com/hank-ai/darknet)，将其作为推荐的、持续维护的 C/C++ 后继项目。

**陷阱在移植版。** 2026 年几乎没有人再用 Darknet 训练；人们会选用 PyTorch 重新实现，而这些实现都带有各自的许可：

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3) 是最流行的 YOLOv3 移植版，采用 **AGPL-3.0**。一个公有领域的架构，以这一领域中最严格的常见许可重新分发。你得到什么许可，完全取决于你 pip 安装的是谁的移植版。
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) 采用 **GPL-3.0**。
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) 采用 **Apache-2.0**，因此 v4 的宽松路线在 PyTorch 中得以延续。
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4) 出自 YOLOv4 的共同作者，**完全没有 LICENSE 文件**。这比 copyleft 更糟：没有许可就不授予任何权利，默认即保留所有权利。

请检查你克隆的仓库的许可，而不是论文上的版本号。

## 第二阶段：GPL-3.0（YOLOv6、v7、v9）

[YOLOv6](https://github.com/meituan/YOLOv6)（美团）、[YOLOv7](https://github.com/WongKinYiu/yolov7) 和 [YOLOv9](https://github.com/WongKinYiu/yolov9) 以 GPL-3.0 发布。

GPL-3.0 是 copyleft：如果你分发修改后的版本，或者把 GPL 代码与你的代码组合成单个程序，你就必须以 GPL-3.0 传递（convey）该组合作品的完整对应源代码。有两条边界很重要。**单纯聚合（mere aggregation）** 指的是，仅仅一起分发的独立程序并不会自动构成组合。而触发条件是**分发**：与 AGPL 不同，如果你从不交付该组合作品，普通 GPL 不会施加任何公开义务，这就是为什么有些公司把 GPL 模型严格地放在自己的 API 后面。

这条路比听上去更窄，而且它会以三种人们很少提前考虑的方式失效。模型一旦落到客户设备上或进入本地部署（on-prem）安装，你就是在分发。一旦副本离开你的法律实体，「仅限内部」就不再成立：把构建产物交给承包商、外包开发团队或单独注册的关联公司就是分发，尽管在一家公司内部到处复制它并不是。而且，只有当从该 API 可以触及的*所有东西*都是 GPL 或更宽松的许可时，这道屏障才成立，因为被提供服务的作品中任何地方只要有一个 AGPL 组件，就会把整个作品拉入第 13 条的范围，无论 GPL 文件本身怎么说。

**YOLOv7 和 YOLOv9 有上文所述的 MIT 出路。** YOLOv6 没有，而这方面最有力的证据来自百度：PaddleDetection 采用 Apache-2.0，其维护者特意把 YOLOv5、YOLOv6、YOLOv7 和 YOLOv8 隔离到一个单独的 GPL-3.0 仓库 [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO) 中，并表示这些模型的代码「will not be merged into PaddleDetection」（不会合并进 PaddleDetection）。当百度这样规模的公司专门筑起一道墙，把 YOLOv6 挡在自己的宽松许可框架之外时，这就说明了该许可的分量。

## 第三阶段：AGPL-3.0（YOLOv5、v8、v10、11、v12、v13、26、YOLOE）

Ultralytics 于 **2023 年 4 月 14 日**改为 AGPL-3.0，而不是流传极广的 2022 年（包括在这个问题上排在搜索结果第一页的指南也这么说）。此后的一切（[YOLOv8](https://github.com/ultralytics/ultralytics)、YOLO11 和 [YOLO26](https://arxiv.org/abs/2606.03748)）都采用 AGPL-3.0，替代方案是付费的企业许可（Enterprise License）。他们的[许可页面](https://ultralytics.com/license)写明，企业版涵盖「YOLO26, earlier YOLO versions, and any future YOLO models」（YOLO26、更早的 YOLO 版本以及任何未来的 YOLO 模型），因此这一政策是刻意且面向未来的。定价不公开。

这个日期是可以核查的，也值得核查，因为这个故事的流行版本是错的，而且错得很关键：

- `ultralytics/yolov5` 中的 `LICENSE` 文件在其历史上恰好被改动过三次。第三次是提交 [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d)「Update LICENSE to AGPL-3.0 (#11359)」，日期为 **2023-04-14**，它在 101 个文件中把 `GNU GENERAL PUBLIC LICENSE` 替换为 `GNU AFFERO GENERAL PUBLIC LICENSE`。
- 2022 年最后一个 YOLOv5 版本 **v7.0（2022 年 11 月 22 日）仍以 GPL-3.0 发布**，在它之前的 v6.2、v6.1 和 v6.0 也是如此。在 2023 年 4 月之前停止同步的成千上万个分支也一样：随便挑一个，GitHub 如今仍然显示 GPL-3.0。
- 同一天，41 分钟后，`ultralytics/ultralytics` 也受到了[同样的处理](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4)（PR #2031）。

由此得出一个几乎没人知道的事实：**YOLOv8 首发时用的并不是 AGPL。** 它于 2023 年 1 月以 **GPL-3.0** 发布，三个月后才更换许可。PyPI 的记录毫不含糊：`ultralytics` 8.0.0（2023 年 1 月 10 日）到 8.0.76（2023 年 4 月 13 日）声明的都是 `GPL-3.0`。2023 年 4 月 16 日上传的 8.0.80 是第一个声明 `AGPL-3.0` 的版本。

这不是漏洞。已经授予的许可不会被追溯撤销，因此那些旧版本仍可按其发布时的条款获取。但它们已经过时三年，无人维护，也没有补丁；GPL-3.0 仍然是 copyleft（你只是把网络条款换成了分发条款，并没有摆脱 copyleft）；而且 Ultralytics 的商业立场本来就比许可文本更宽泛。有用的结论更窄，也更普适：**许可附着在你收到的那个版本上，而不是项目的名字上。** 锁定你的依赖版本，并记录你获取副本那天许可是怎么写的，因为项目明天就可能更改许可，而你的合规说法取决于你实际拿到的是哪一个。

AGPL-3.0 就是 GPL 加上第 13 条，即网络条款：*如果你修改了本程序*（if you modify the Program），就必须向通过网络与你修改后的版本交互的用户提供其源代码。这堵上了普通 GPL 留下的 SaaS 路径。

请注意这个限定条件，因为大多数文章都把它丢掉了。AGPL 第 0 条把「修改」（modify）定义为*以需要获得版权许可的方式*（in a fashion requiring copyright permission）复制或改编该作品。在你自己的数据集上运行未经修改的训练代码，是在执行程序，而不是改编其源代码，就像用未经修改的 GPL 编译器编译你的代码并不会修改编译器一样。因此，单纯训练并不自动构成触发条件，任何告诉你它会触发的人都跳过了这个定义。

真正让你落入该许可范围的，是把该软件包组合或嵌入到你自己的代码库中，使两者作为一个程序一同交付或提供服务。FSF 自己的解读是，将 GPL 或 AGPL 作品与其他模块链接会构成组合作品；法院是否会把这一点延伸到 Python import，从未经过检验。还要注意，Ultralytics 自己的[商业条款](https://ultralytics.com/license)要求在幕后使用 YOLO 的 SaaS 平台、API 和云系统取得企业许可，因此厂商的立场比单纯的许可文本更宽泛。请把集成到你提供服务或交付的产品中视为风险所在；不要认定单独训练就是风险所在。

关于 YOLO26，日期到处都被搞混，所以在此说明：它**于 2025 年 9 月预发布**，地点是 YOLO Vision，**于 2026 年 1 月才正式发布**（`ultralytics` 8.4.0 版本，其发布说明写着「Ultralytics YOLO26 has arrived」），而**论文于 2026 年 6 月发表**。它位于主仓库 `ultralytics/ultralytics` 中；`ultralytics/yolo26` 只是一个落地页。

**让人意外的部分**：过去两年的学术版 YOLO 也是 AGPL，而且并不是因为它们出自 Ultralytics 之手。

- [YOLOv10](https://arxiv.org/abs/2405.14458) 出自清华大学。它的 README 写着「*The code base is built with ultralytics.*」
- [YOLOv12](https://arxiv.org/abs/2502.12524)：「*The code is based on ultralytics.*」
- [YOLOv13](https://arxiv.org/abs/2506.17733)：「*The code is based on Ultralytics.*」
- [YOLOE](https://arxiv.org/abs/2503.07465) 是 YOLOv10 团队推出的开放词汇「see anything」模型，同样采用 AGPL-3.0，并基于 Ultralytics 构建。

copyleft 是会继承的：AGPL 代码的衍生作品必须以 AGPL 传递，因此这些分支一旦被交付或对外提供服务，许可也会随之而来。（修改它并严格只留给自己使用，则不会产生任何义务，这就是为什么研究用途不受影响。）研究是独立的；许可不是。自 2024 年以来，把「下一代 YOLO」作为 Ultralytics 的衍生作品发布已经成为默认的工作流程，这意味着 AGPL 如今会沿着版本号自动向上传播。

如果你看到有页面声称 YOLOv13 是 Apache-2.0，那是错的。LICENSE 文件、GitHub API 和模型卡片都写着 AGPL-3.0。

### 已经不复存在的 YOLOv8 宽松路线

许多仍在网上的指南（包括本文的一个早期草稿）都指向 KerasCV 的 Apache-2.0 版 YOLOv8，把它当作摆脱 AGPL 版 YOLOv8 的干净出路。**如今你不能依赖它。** 以下是公开记录：

- 在 KerasCV 的讨论 [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032) 中，一位贡献者写道「many parts are derived form ultralytics」（许多部分源自 ultralytics）。没有维护者回复该讨论串，因此该实现到底有多独立，这个问题在公开场合从未以任何方式得到解决。
- KerasHub 的 issue [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) 于 2025 年 7 月被一位 Keras 维护者关闭，留言是：「Hey all!! Because of license issues, we have decided to drop this.」（大家好！！由于许可问题，我们决定放弃这项工作。）随后有两个人询问这对 KerasCV 版本现有的商业用户意味着什么。两人都没有得到回复。
- KerasCV 现已归档，而其持续维护的后继项目 KerasHub **不提供 YOLOv8**。

我们不知道这两件事是否有关联，也并不断言发生了任何不当行为。这不是可以在上面构建产品的基础。

## 宽松许可的例外：YOLOX、PP-YOLOE、YOLO-NAS

并不是每个 YOLO 都上了 copyleft 这趟车。

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)**（旷视，2021）采用 Apache-2.0，没有任何例外条款，代码和权重都是如此。它仍然是商业用途中最干净的经典 YOLO，不过该仓库自 2025 年年中以来没有任何提交，而一个关于[在商业应用中捆绑预训练权重的问题](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865)自 2026 年 3 月起一直未关闭，也无人回复。许可文本毫不含糊。只是目前没有人在回答关于它的问题。我们写过一篇关于[用 LibreYOLO 运行 YOLOX](/articles/yolox-with-libreyolo) 的文章。
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)**（百度）**在 PaddleDetection 中**采用 Apache-2.0。请从那里获取，而不是从 PaddleYOLO 获取，后者包含几乎相同的配置目录，且采用 GPL-3.0。同一个模型，同一家公司，两个仓库，两种许可。这正是 YOLOv9 情况的反面。
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)**（Deci，2023）是最容易让人中招的一个。代码是 Apache-2.0；官方权重不是。他们的[单独许可](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md)写明，你「*may not use the Software for any commercial use, including in connection with any models used in a production environment.*」（不得将本软件用于任何商业用途，包括与生产环境中使用的任何模型相关的用途。）自 2024 年英伟达收购 Deci 以来，我们没有在该仓库中发现新的功能开发，而原来的权重 URL 也已失效（检查点现在托管在另一个地方）。有时有人提出一种变通办法，即从头训练该架构就能让你只受 Apache 许可约束，这种说法看似合理，但尚未得到 Deci 或英伟达的确认，而且它与许可文本中延伸到「any components comprising the model」（构成该模型的任何组件）的措辞并不协调。更多关于 YOLO-NAS 的内容见[这里](/articles/yolo-nas-with-libreyolo)。
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** 在一些较早的指南中有时被列为 Apache-2.0。并非如此：MMYOLO 采用 GPL-3.0（它的兄弟项目 MMDetection 才是 Apache 许可的那个），并且自 2024 年 7 月以来没有任何提交。
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)**（腾讯）采用 **GPL-3.0**，不是 AGPL，也不是宽松许可。之所以值得一提，是因为开放词汇 YOLO 常常因为它们所借鉴的 CLIP 类生态，而被想当然地认为是宽松许可。

这些案例涵盖了该领域三种最常见的许可错误：以为权重跟随代码的许可，以为同一组织的所有东西共享同一种许可，以及根据模型的名称推断其许可。

## 许可覆盖的是代码，不是思想（但要当心专利）

上面每一条宽松路线的法律基础都是同一个原则：**版权保护的是表达，而不是思想。** 这不是一句口号，而是大西洋两岸都明文确立的法律。

在美国，这是 [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102)，它把任何「idea, procedure, process, system, method of operation, concept, principle, or discovery」（思想、程序、过程、系统、操作方法、概念、原理或发现）排除在版权保护之外，可追溯到 *Baker v. Selden*（1879）。在欧盟（这也是管辖我们、大概也管辖相当多读者的法律），[软件指令（2009/24/EC）](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024)第 1(2) 条规定，「ideas and principles which underlie any element of a computer program」（构成计算机程序任何元素之基础的思想和原理）不受保护，而欧盟法院在 *SAS Institute v World Programming*（[C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10)，2012）一案中确认，程序的功能、编程语言和数据格式本身都不是受保护的表达。只有代码才是。

写在 arXiv 论文里的架构是已公开的思想。任何人都可以实现它。你不能做的是复制或改编别人的 GPL/AGPL 源文件，再为结果更换许可。一个独立实现，由依据论文而非源代码工作的人编写，是一件新作品，由其作者选择许可。

大多数厂商都会略过的两点坦率的注意事项：

- **「独立」必须属实。** 站得住脚的做法是净室（clean-room）流程：写代码的人依据论文和规格说明工作，而不是依据 copyleft 源代码。仔细阅读 GPL 仓库然后再「重写」并不是一回事，而一旦有人追究，这种区别就很重要。
- **这是版权层面的论点，不是专利层面的。** 独立发明*不能*作为针对专利的抗辩。净室重新实现解决了版权问题，而专利问题原封不动。如今并没有被广泛主张的专利笼罩在 YOLO 架构之上，但「还没人起诉」并不等于「没有可以起诉的东西」。

## 权重是第二重许可

这整个领域里代价最高的错误，是审计了代码却忘了检查点。有三层内容带有许可，而且它们可能各不相同：

**1. 代码。** 即以上所有内容。

**2. 权重。** 有时是明确规定的，比如 YOLO-NAS。有时是单方面主张的：Ultralytics 的[许可页面](https://ultralytics.com/license)声称 AGPL-3.0「covers the training code and the models produced by that training code」（涵盖训练代码以及由该训练代码产出的模型），并将其延伸到你用他们的代码自己训练的模型。作为其所发布内容的版权持有者，他们可以为自己的检查点设定条款。至于*你*产出的权重在法律上是否属于训练代码的衍生作品，这是一个真正悬而未决的问题：没有判例，而且这与 FSF 自己的一般规则相悖，即程序的输出并不自动受该程序版权的覆盖。不过，悬而未决并不等于安全。实践中，请把已发布的 AGPL 权重视为有权利负担（encumbered）。

对于采纳本文建议的人，有一处最需要当心：**GPL 版 YOLOv7 和 YOLOv9 仓库对其权重只字未提。** 它们的检查点是 GPL-3.0 仓库中的发布资产，没有单独的授予。把这些 .pt 文件加载进 MIT 重新实现，你得到的是干净的代码，加上一个状态无人澄清的检查点。如果许可正是你切换的原因，请使用宽松许可项目自己训练的权重，或者自己训练。

**3. 训练数据。** COCO 的*标注*采用 CC BY 4.0，所以标签没有问题。*图像*则不归 COCO 来许可：它们是 Flickr 照片，适用各不相同的个人条款，而 COCO 联盟明确表示并不拥有这些图像。业内大多数人把这视为低风险，然后继续往前走，但「COCO 没问题」这句话只针对标注。在 COCO 之外，情况很快会变得更严格：大量航拍、医疗和驾驶数据集是非商用的，而模型代码的宽松许可并不能把它们洗白。

每一次都要审计这三层。

## 关于 MIT 与 Apache-2.0

我们发布的是一个 MIT 许可的库，所以告诉你 MIT 就是最好的许可，对我们来说会很方便。坦率的比较要更有意思。

MIT 并非零义务：你必须在你分发的副本中保留版权声明和许可文本。这是一个真实的条件，只是成本很低。而 Apache-2.0 有 MIT 没有的东西：来自每一位贡献者的**明示专利许可**，外加一条报复条款，任何就该作品提起诉讼的人，其所获授予都会被终止。MIT 对专利只字未提。对于主要担心专利风险的公司来说，Apache-2.0 可以说是两者中*更安全*的那个，而宽松许可检测生态中的很大一部分（YOLOX、RT-DETR、D-FINE、DEIM）正是出于这个原因采用了 Apache-2.0。

MIT 给你带来的是简单和兼容性。这两种许可都永远不会强迫你公开源代码，而这才是本文真正关注的维度。如果有人告诉你 MIT 还是 Apache 才是重要的决定，那他是在推销什么东西。重要的决定是宽松许可还是 copyleft。

## 那么，你到底能用哪个 YOLO？

- **闭源商业产品**：通过该实验室的 [MIT 仓库](https://github.com/MultimediaTechLab/YOLO)使用 YOLOv7 或 YOLOv9，原版中的 YOLOX 或 PP-YOLOE（取自 PaddleDetection），或者如果你不想自己拼装和审计这一切，选择一个持续维护的 MIT 框架，比如 LibreYOLO。除非你愿意开源整个应用或购买企业许可，否则避开任何 GPL 或 AGPL 的东西。
- **开源项目**：什么都可以，只要你的许可兼容。如果你的项目是 AGPL，Ultralytics 系列就是自然的选择，而最新的研究（YOLOv13、YOLO26、YOLOE）也最先落地在那里。
- **研究与基准测试**：许可几乎不会限制你。你对比的那篇论文用什么，你就用什么。
- **「以后再处理」的计划**：行不通。产品建成之后再拆除一个 AGPL 依赖，意味着要重新训练、重新验证、重新导出所有东西，包括权重。先选定仓库，也就是先选定许可。

如果你想比较的是框架本身而不是它们的许可，请看 [2026 年最佳 Ultralytics 替代方案](/articles/best-ultralytics-alternatives)。

## MIT 选项：LibreYOLO 提供什么

声明：LibreYOLO 是我们的项目，它的许可正是本文存在的原因。

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) 是一个采用 MIT 许可的代码库**，用一套 API 覆盖下列检测器，并内置训练以及 ONNX、TensorRT、OpenVINO 和 NCNN 导出。没有 copyleft，没有网络条款，没有企业版。

以下是完整的目标检测阵容，列出了每个家族的上游及其许可，让你清楚地看到每条宽松路线来自哪里：

| LibreYOLO 中的名称 | 模型 | 上游 | 上游许可 |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | Darknet 时代的 YOLO，PyTorch 版 | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | 公有领域 |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT（该实验室自己的重写版，而非 GPL 仓库） |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9，外加一个无需 NMS 的端到端变体 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT（该实验室自己的重写版，而非 GPL 仓库） |
| `LibreYOLO9P2` | 带有面向小目标的 stride-4 head 的 YOLOv9 | LibreYOLO 原创 | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS，检测与姿态 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | 代码为 Apache-2.0。受限的上游权重**不**做再分发 |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR 及 v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR，检测、分割、姿态、OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | N/S/M/L 为 Apache-2.0。XL/2XL 采用 Roboflow 的 Platform Model License，因此我们只提供 Apache 许可的尺寸 |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | 代码为 Apache-2.0。较大尺寸使用 Meta 的 DINOv3 骨干，采用 Meta 自己的非 OSI 许可，该许可允许再分发，但**禁止用于军事、核、间谍和武器用途**。这些尺寸以 `other` 发布，而不是 Apache。若你要交付任何军民两用的东西，请先读一遍该许可 |
| `LibreRTMDet` | RTMDet（[不依赖 MMDetection](/articles/rtmdet-without-mmdetection)） | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0（Apache 许可的兄弟项目，而不是 GPL-3.0 的 mmyolo） |
| `LibrePICODET` | PP-PicoDet | 架构来自 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)，检查点来自 [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) 重新移植版 | Apache-2.0（两者皆是） |
| `LibreEC` | EdgeCrafter，检测、姿态、分割 | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | 面向微控制器级硬件的质心检测器 | LibreYOLO 原创 | MIT |
| `LibreGroundingDINO` | Grounding DINO，开放词汇（推理） | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2，开放词汇（推理） | Google，通过 `transformers` | Apache-2.0 |

关于这一列，有两点是我们刻意坦诚交代的。

**代码库中没有复制任何 GPL 或 AGPL 源代码。** 上面每一个家族都基于公有领域、MIT 或 Apache-2.0 的上游构建，这正是让框架保持 MIT 的规则。当一个架构同时有宽松许可和 copyleft 两个出处时，我们选择从宽松的那个构建：我们的 YOLOv9 和 YOLOv7 源自该实验室的 MIT 仓库，而不是 GPL 原版；我们的 RTMDet 源自 Apache 许可的 MMDetection，而不是 GPL 许可的 MMYOLO。YOLO-World 不在列表中，尽管经常有人要求，因为它是 GPL-3.0，没有任何宽松许可的来源可以取用。

**宽松许可不等于毫无限制，我们宁愿把这一点说清楚，也不愿让你在审计中才发现。** DEIMv2 的较大尺寸继承了 Meta 的 DINOv3 条款，包括禁止用于军事、核、间谍和武器用途，如果你的工作与军民两用沾边，这是一个真实的约束。少数研究预览版检查点是在非商用数据集（DOTA、VisDrone）上训练的，它们在 Hugging Face 上被标注为 `cc-by-nc`，而不是悄悄当作免费的来发布。我们对别人权重的那份谨慎同样适用于我们自己：我们的 YOLO9 检查点是从该 MIT 实验室仓库自己的检查点转换而来的，因此它们继承了那些检查点的一切状况，而如上文所述，那是「有主张，未经正式审计」。MIT 许可覆盖的是我们编写的代码。每个检查点都带有其训练数据和训练起点的条款，并按权重逐一公布。

除了检测之外，同一套 API 还覆盖实例分割和语义分割、姿态、旋转框、分类（MobileNetV4、ConvNeXt、EfficientNetV2、ResNet、CLIP）、单目深度、图像复原和视线估计。

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

同样的 YOLO 工作流，不用做任何许可功课。

在 GitHub 上点个 star：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 文档：[libreyolo.com/docs](https://libreyolo.com/docs)

---

*商标与关联声明：YOLO、YOLOv8、YOLO11、YOLO26 和 Ultralytics 是其各自所有者（包括 Ultralytics Inc.）的商标。LibreYOLO 是一个独立项目，与 Ultralytics 或本文提及的任何其他组织均无关联，也未获得其赞助或认可。产品和仓库名称仅用于识别和比较文中讨论的软件。*

*本文为一般性信息，不构成法律意见，内容截至 2026 年 7 月 11 日，作者是本文所比较的厂商之一。交付之前，请核实当前的许可。*
