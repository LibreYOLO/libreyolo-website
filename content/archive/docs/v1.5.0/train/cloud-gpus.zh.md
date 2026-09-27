---
title: 在租来的 GPU 上训练
seo_title: 在租来的云 GPU 上训练 LibreYOLO
description: 在租来的或 serverless 的 GPU 上跑一次 LibreYOLO 训练任务：准备数据、安装、启动、实时观察、取回权重，然后停止计费。
lead: >-
  租来的 GPU
  会把一次训练变成一个有开始、有结束、有账单的任务。要做的事和本地训练一样；变的是怎么把数据送进去、怎么从外面观察、怎么把权重取出来，以及怎么把机器关掉。
keywords:
  - 云 gpu 训练
  - 租 gpu 跑训练
  - vast.ai 训练
  - modal serverless gpu
  - beam gpu 训练
  - 远程训练
  - hugging face 数据集上传
  - gpu 每轮训练成本
last_verified: 1.5.0
snippets:
  install:
    - label: 在机器上
      language: bash
      code: |
        pip install libreyolo

        # 只装这次运行需要的 extras，rfdetr 用于 RF-DETR 训练，
        # lora 用于参数高效的微调，onnx 用于之后导出
        pip install "libreyolo[rfdetr,lora]"
    - label: 先检查 GPU，别的都往后放
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # 为另一种架构编译的 wheel 会报告 True，然后在第一个真正的 kernel 上
        # 失败，所以这里就跑一个
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: 在你自己的机器上打包并上传一次
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: 在机器上准备数据
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: 后台运行，断线也不会带走任务
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: 多卡训练，从一个 Python 文件启动
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # 所有 GPU 上的全局批大小
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: 一次廉价的读取
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: 从脚本里读
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: 在浏览器里，通过 SSH 隧道
      language: bash
      code: |
        # 在机器上（默认绑定 127.0.0.1:8420）：
        libreyolo monitor /root/runs/run1 --no-browser

        # 在你自己的机器上执行，然后在本地打开 http://localhost:8420：
        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: 把权重推到一个长期的地方
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## 在租任何机器之前

有两个决定，拖到后面做的代价比现在做要大。

先把数据集放到 CDN 上。把它打包成单个 tar 放进一个 Hugging Face 数据集仓库，在每
家厂商那里的用法都一样，对它们的下载速度都很快，而且仓库是私有的时候，也只需要在
任务环境里放一个 `HF_TOKEN`。从家庭宽带往上传数据集，或者在机器上从一个慢速源站拉
取，都是在按 GPU 计费的时间里干等。

<code-tabs name="stage" />

然后确定磁盘的大小。按存储计费的厂商是按分配的容量计费，而不是按用掉的容量，而且
磁盘创建之后无法缩小。把准备好的数据、检查点（checkpoint）和大约 30% 的余量加起
来，到此为止。

## 在机器上安装

<code-tabs name="install" />

如果镜像里没有自带与这张卡匹配的 CUDA 构建，就先装 PyTorch，再装 LibreYOLO，这样
pip 不会自己解析出一个仅 CPU 的 torch。第二个代码片段不是可有可无的形式：为错误的
GPU 架构编译的 wheel 会报告 `torch.cuda.is_available() == True`，然后在第一个真正
的操作上以 `CUDA error: no kernel image is available for execution on the device`
失败。一次矩阵乘法能提前发现它，一小时的环境搭建不能。

如果厂商提供卷，就把 `HF_HOME` 指向持久化存储，这样检查点和数据集的下载能在多次运
行之间留存下来。

## 启动

让任务以后台方式运行。一个随网络连接一起断掉的交互式会话，会把训练一起带走。

<code-tabs name="launch" />

`batch=-1` 恰恰在这里值得用，因为你通常是在一张此前没有训练过的卡上。它会用一次真
正的反向传播在训练模式下探测模型，选出放得下的最大的 2 的幂，这比二十分钟之后用一
个显存不足的错误去发现上限要快。参见[超参数](/docs/train/hyperparameters)。

在多卡机器上，`device="0,1,2,3"` 自己就会为每块 GPU 启动一个 worker，而 `batch`
仍然是所有卡上的全局批大小。`__main__` 保护是必须的，因为每个 worker 都会重新导入
这个脚本。这一点，以及其余的分布式行为，都在[多卡训练](/docs/train/multi-gpu)里。

## 从外面观察

每次运行都会把 `status.json` 写进它的运行目录，并在每一轮原子地重写一次。它就是那
次廉价的读取：几百字节，带着状态、当前轮次、ETA 和最新的指标，不用去解析日志。

<code-tabs name="watch" />

它旁边的 `metrics.jsonl` 有完整的逐轮历史，`train.log` 有控制台输出。`libreyolo
monitor` 只用标准库就把这三者做成一个浏览器面板，所以除了 LibreYOLO 本身，机器上
不需要再装任何东西。通过 SSH 端口转发访问它。

这些都不会碰到训练进程，所以它们既可以接上一次正在跑的运行，也可以重新打开一次已
结束的运行，或者检查一次崩掉的运行。

## 在停止付费之前把权重取出来

机器是一次性的。在中间的里程碑上就推送检查点，而不是只在最后推送，否则一次崩溃、
一次抢占或者额度用完，就会把整次运行都赔进去。

<code-tabs name="push" />

`weights/best.pt` 和 `weights/last.pt` 在每一轮以及每次有改进时都会写出。
`save_period=N` 会在此之上再加上 `weights/epoch_<N>.pt` 快照，正是它让运行中途的推
送变得廉价。`summary.json` 和 `results.csv`，在家族会写出它们的地方，都很小，也值
得一并带走。

在 `on_train_epoch_end` 上挂一个回调，是把推送自动化的干净做法。参见[实验日志记录
器](/docs/train/loggers)，那里托管的后端还能让你完全不碰这台机器就拿到指标。

## 停止付费

这是出错时真正花钱的部分，而且规则随厂商的模式而变。

在一个租裸机的市场平台上，计费按墙钟时间走，直到实例被销毁为止。空闲的 GPU 和忙碌
的 GPU 计费完全一样，所以光把训练进程杀掉本身省不了钱。已停止的实例仍然要为它的磁
盘付费。

在一个任务就是一个带装饰器的函数的 serverless 平台上，函数返回时容器会缩容到零，
所以忘掉一台机器的可能性要小得多。一个没有超时的挂起任务仍然在计费，所以永远要设
一个。

停止而不是销毁，是一个真实的杠杆，也是一个真实的陷阱。2026 年 7 月 31 日在一台租
来的 8x RTX 4090、250 GB 磁盘的机器上实测：运行中每小时计费 3.4828 美元，停止后仅
磁盘每小时计费 0.0694 美元，销毁后不计费。这是 98% 的节省，同时环境、准备好的数据
和检查点都原地保留。

停止状态的费率是一道你在租之前就能算的算术题：

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

拿它和重建一次的成本比一比：重新租、拉镜像、安装，再重新准备数据。在同一台机器上，
一次重建大约是 15 分钟的环境搭建加上 43 GB 的入站流量，总共大约 1.00 美元。相对于
每小时 0.0694 美元，大约 14 小时之内回来更划算的是停止，间隔更长则更划算的是销毁
并从准备好的副本重建。

有一个风险会让停止对稀缺硬件变得不安全：停止会释放 GPU。没有任何东西为你保留它
们，所以只有宿主机上它们仍然空闲时，重新启动才会成功。你的磁盘是安全的；你的 GPU
不是。

## Serverless，以函数的形式

如果你宁可不去管理一台机器，Modal 和 Beam 都可以在 GPU 上运行一个带装饰器的 Python
函数，并在它返回时缩容到零。LibreYOLO 自己的每夜测试套件就跑在 Modal 上，库仓库里
的 `tools/ci/modal_nightly.py` 就是那个可以照抄的、仓库内可用的示例。

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # OpenCV 的系统库
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # 在多次运行之间缓存权重

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # 持久化这个卷


@app.local_entrypoint()
def main():
    train.remote()
```

用 `modal run modal_train.py` 运行它。容器的文件系统是临时的，所以任何值得留下的东
西都要放进卷里，或者推送出去。显式设置 `timeout=`；它是挂起的运行和一张没有上限的
账单之间唯一的东西。

Beam 是同样的形状：一个 `@function` 装饰器、一个 `Volume`，以及从 `__main__` 里调
用的 `train.remote()`。

## 按每个任务的成本来选合适的卡

每小时多少钱是错误的优化目标。小模型会让一张大卡半闲着，所以更便宜、更慢的 GPU 往
往每轮反而更便宜。在投入一次长时间运行之前，先在租来的卡上跑几步 profiler：如果结
论是 `dataloader` 或 `host / launch`，更快的 GPU 什么也换不来，而更多的 worker 或
更大的批大小能换来很多。参见[训练性能](/docs/train/performance)。

## 相关

- [数据集](/docs/train/datasets)：准备好的归档应该有的目录结构，以及那个能在 GPU
  开始计费之前就发现问题的 doctor 命令。
- [多卡训练](/docs/train/multi-gpu)：面向多卡机器。
