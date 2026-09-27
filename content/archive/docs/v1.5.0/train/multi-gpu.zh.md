---
title: 多卡训练
seo_title: LibreYOLO 中的多卡训练
description: >-
  用 device="0,1" 在多张 GPU 上训练。库怎么派生 DDP worker、为什么 batch 是全局批大小、什么时候要设 sync_bn，以及
  torchrun 这条路。
lead: >-
  LibreYOLO 的多卡训练就是 PyTorch 的 DistributedDataParallel：每张 GPU
  一个进程，每个进程持有一份完整的模型副本和每个批次的一个分片，每一步在各 rank 之间对梯度求平均。
keywords:
  - pytorch ddp 分布式训练
  - 多卡训练 yolo
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm 同步 bn
  - 全局批大小
  - nccl gloo 后端
  - windows 多卡训练
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # __main__ 保护是必须的：每个派生出来的 worker 都会重新导入这个
        # 模块，没有这层保护它就会递归地重新启动训练
        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # 全局批大小：两张 GPU 上每张 16 张图
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: 启动
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # 在 GPU 0 上探测一次，再放大到 world size 的整数倍
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## 在两张 GPU 上训练

传一个设备列表。其他什么都不用改。

<code-tabs name="train" />

拿到多于一个设备、而且没有 torchrun 环境时，模型的 `train()` 会把权重存到一个临时
文件，按需要解析 autobatch，然后用 `torch.multiprocessing.spawn` 为每张 GPU 派生一
个 worker 进程。每个 worker 重新导入模型类，从存下来的权重把它重建出来，再走普通的
单设备路径，因为在派生出来的 worker 内部，torchrun 的那些环境变量是设好的。运行结
束时，rank 0 的最佳检查点（checkpoint）会被加载回调用方的模型实例。

`device` 接受 `"0,1"`、`[0, 1]`、`0`、`"cuda:0"`、`"cpu"`、`"mps"` 和 `"auto"`。
只有超过一个 CUDA 索引的列表才会触发派生。

## `__main__` 保护是必须的

派生出来的 worker 会重新导入它们所来自的那个模块。没有 `if __name__ == "__main__":`
这层保护，这次导入就会把训练调用重跑一遍，每个 worker 又派生出自己的 worker。库会
检测到这种情况并抛错，而不是任由它递归下去：

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

所有进入 worker 的东西都会被 pickle，所以 `callbacks=` 必须是可 pickle 的。模块级
别的类可以；闭包或者 lambda 不行，错误信息会这么说，并指向内置的那些 logger 作为
替代。

## batch 是全局批大小

`batch` 是所有 GPU 加起来每个优化器步的图片数。每个 rank 的 dataloader 按
`batch // world_size` 配上 `DistributedSampler` 构建，所以两张 GPU 上 `batch=32`
意味着每张 GPU 16 张图，而不是 32 张。

不能被 world size 整除的批大小会抛错，而不是悄悄按另一个大小训练：

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

梯度由 DDP 自己求平均，所以损失是不加缩放直接传下去的。在这之上再乘以 world size，
会把实际的学习率放大差不多 GPU 数量那么多倍。

## DDP 下的 autobatch

`batch=-1` 可以用，返回的是一个能被 world size 整除的全局批大小。

<code-tabs name="autobatch" />

走派生这条路径时，探测在任何 worker 存在之前就已经在父进程里、在第一个设备上跑完
了，所以每个 worker 拿到的都是一个具体的整数，不需要任何进程间的协调。在 torchrun
下，由 rank 0 探测，再把结果作为一个 long 张量广播出去。

探测量的是一张 GPU 的容量，再乘以 world size。设了 `nbs` 时，全局批大小会被限制在
`nbs`，并向下取整到 world size 的整数倍，所以加 GPU 减少的是梯度累积的步数，而不是
缩小每张 GPU 的批大小。探测本身的机制在[超参数](/docs/train/hyperparameters)。

## SyncBatchNorm

在 DDP 下，每个 rank 的 BatchNorm 层只看得到自己那一份分片。在
`batch // world_size` 这个规模上，那份分片可能小到让滑动统计量拖累收敛后的模型，比
单卡跑出来的更差。

`sync_bn=True` 会把每个 BatchNorm 转成 SyncBatchNorm，让统计量在全局批上计算。这个
转换只在分布式生效时才发生，所以单卡训练不管这个 flag 怎么设都不受影响。

在 BatchNorm 用得多的卷积家族里它默认就是开的：YOLOX、YOLOv7、YOLOv9 及其变体、
YOLO-NAS、PicoDet、RTMDet 和 FOMO。其他每个家族默认都是关的。当模型里含有
BatchNorm、`sync_bn` 是关的、而且每个 rank 的批大小低于 16 时，训练器会告警。

<code-tabs name="syncbn" />

`sync_bn` 没有对应的 CLI flag。它是一个 Python 参数。

## 用 torchrun 启动

torchrun 也能用，而且当集群调度器已经掌管进程启动时，它才是对的选择。按单设备去写
脚本，让 torchrun 来设置 rank 相关的环境。

<code-tabs name="torchrun" />

不要把两者混着用。torchrun 环境在的时候，`device="0,1"` 不会派生进程；训练器取
`cuda:LOCAL_RANK`，进程数由 torchrun 掌管。

## rank 的行为

所有副作用都归 rank 0。它解析运行目录，并把解析出来的名字广播出去让所有 rank 保持
一致，写检查点和产物，触发用户回调和 logger。其他 rank 负责训练并贡献梯度。

每个 rank 用不同的种子初始化自己的 dataloader 和数据增强 RNG，种子由配置的 `seed`
派生而来，这样各个 rank 不会抽到一模一样的增强。

## 平台与后端

后端是自动选的：CUDA 和 NCCL 都可用时选 NCCL，否则用 Gloo。NCCL 在 Windows 上没有
构建，所以 Windows 上的运行不用任何配置就会用 Gloo。进程组初始化时带的是三小时的超
时。

## DDP 下跑不了的东西

- CUDA 图捕获。`cuda_graph=True` 会记一行日志，然后按 eager 模式训练。见[训练性能](/docs/train/performance)。
- 训练 profiler。`profile=True` 会被忽略，并给出一条警告。

不是每个家族都支持自动派生。有二十四个支持，覆盖了会训练的检测、分类、语义和复原家
族。不支持的家族拿到一个多卡设备时，会抛出一个错误，指明模型 API 和 torchrun 命
令，而不是悄悄在一张 GPU 上训练。

## 相关

- [超参数](/docs/train/hyperparameters)，讲 `batch`、`nbs` 和断点续训。
- [实验 logger](/docs/train/loggers)，讲回调的可 pickle 约束。
- [在租来的 GPU 上训练](/docs/train/cloud-gpus)，讲怎么租一台多卡机器。
