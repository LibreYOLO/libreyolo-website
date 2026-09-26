---
title: 引用
seo_title: 引用 LibreYOLO 和上游作者
description: 如何在论文里引用 LibreYOLO，以及如何引用你所运行的那个模型家族的作者。两者都该写进同一个方法部分。
lead: 一条完整的 LibreYOLO 引用有两部分：库本身，以及产出这个结果的模型家族背后已发表的工作。
keywords:
  - libreyolo 引用
  - libreyolo bibtex
  - citation cff 怎么写
  - 模型论文引用格式
  - 计算机视觉论文引用
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## 引用 LibreYOLO

仓库把引用元数据发布为
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff)，
而不是一段 BibTeX。GitHub 会读这个文件，并在仓库页面上给出一个 Cite this repository
按钮，由它生成 APA 和 BibTeX。从那里取条目，不要自己手敲。

文件全文：

```yaml
cff-version: 1.2.0
message: "If you use LibreYOLO in your research or software, please cite it as below."
title: "LibreYOLO"
type: software
authors:
  - family-names: Ceccon
    given-names: Xuban
  - name: "The LibreYOLO contributors"
license: MIT
url: "https://github.com/LibreYOLO/libreyolo"
repository-code: "https://github.com/LibreYOLO/libreyolo"
```

它刻意不带版本号，也不带发布日期。
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
要求维护者在发布时永远不要给 `CITATION.cff` 或 `.zenodo.json` 升版本、加日期或改标题，
这样每一条引用都落在同一条记录上，而不是散落到各个版本。你运行的是哪个版本，写在自己的
正文里，引用本身不要动。

## 引用模型家族

LibreYOLO 是一个移植。运行 `LibreRFDETRm.pt` 就是在运行 RF-DETR，而写出 RF-DETR 的
那些人，才是审稿人期望看到被致谢的对象。只引用这个库，等于把他们的工作归到了错误的项目上。

需要的一切都在该家族的页面上。头部的 Upstream 一行给出原始工作和它背后的机构，并链接论文
和源码仓库。再往下的 Citation 部分放着 BibTeX。

那段 BibTeX 是从作者自己的引用区块逐字复制来的，通常是上游 README 的 Citation 部分或一个
`CITATION.cff`，并且渲染时带一个指回来源区块的链接，方便你对着原文核对。它绝不是用论文元数据
拼出来的。手工重建的条目会悄无声息地出错，代价却很高：漏掉一位共同作者、会议写错、条目类型
写错、年份取自预印本。预印本也会被录用，所以即使你读到的版本在 arXiv 上，条目也可能是
`@inproceedings`。

按原样复制这个区块。如果你的参考文献样式需要另一种条目类型，就转换这个条目，而不是重新敲
一遍，并保持作者列表的原始顺序。

## 方法部分需要什么

三样东西能让一个 LibreYOLO 结果可复现、且归属正确：

- 库本身，引用自 `CITATION.cff`，连同你运行的版本。`libreyolo version` 会打印版本号，
  以及它所运行的 Python、torch 和 CUDA 版本。
- 上游工作，引用自该家族页面的 Citation 部分。
- 确切的检查点（checkpoint）文件名，比如 `LibreRFDETRm.pt`。同一家族内不同尺寸的表现
  并不一样，而且有几个家族会把在不同数据集上训练的检查点放在同一个前缀下，所以只写家族名
  并不能确定跑的到底是什么。

对 LibreYOLO 发布的大部分内容来说，署名同时也是一条许可条款。Apache-2.0 和 CC BY 系列
都要求这份声明随你再分发的权重一起传递，这与引用一篇论文是两回事。哪些条款适用于哪个检查点，
见[许可](/docs/licensing)。
