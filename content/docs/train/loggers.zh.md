---
title: 实验 logger
seo_title: LibreYOLO 中的实验 logger 与回调
description: >-
  把训练指标发送到 TensorBoard、MLflow、Weights & Biases、Comet、ClearML、Neptune 或
  DVCLive，并在四个训练钩子上写你自己的回调。
lead: 每个可训练的家族都会发出四个训练事件。内置的 logger 就是监听同样这些事件的回调对象，所以接一个后端和写一个自定义钩子用的是同一套接口。
keywords:
  - tensorboard 训练可视化
  - mlflow 实验跟踪
  - wandb 记录训练指标
  - clearml
  - comet ml
  - neptune
  - dvclive
  - 训练回调函数
  - 训练指标 csv
  - libreyolo monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: 用名字
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: 配置好的实例
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import MLflowLogger

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="coco8.yaml",
            epochs=10,
            loggers=[MLflowLogger(tracking_uri="sqlite:///mlflow.db"), "tensorboard"],
        )
  callback:
    - label: 一个普通函数
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: 一个带多个钩子的对象
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.training import TrainEndEvent, TrainEpochEvent,
        TrainStartEvent



        class RunLog:
            def on_train_start(self, event: TrainStartEvent) -> None:
                print(f"{event.model_family}{event.model_size} -> {event.save_dir}")

            def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
                if event.is_best:
                    print(f"new best at epoch {event.epoch}: {event.best_metric}")

            def on_train_end(self, event: TrainEndEvent) -> None:
                print(f"done in {event.total_seconds:.0f}s")


        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="coco8.yaml", epochs=10, callbacks=RunLog())
  monitor:
    - label: 在浏览器里看一次 run
      language: bash
      code: |
        libreyolo monitor                     # runs/ 下最近的那次 run
        libreyolo monitor runs/train/exp      # 指定某一次 run
source_hash: de035acbaed32804
---

## 打开一个 logger

`loggers=` 接受一个已注册的名字、一个配置好的实例，或者两者混着放的可迭代对象。

<code-tabs name="logger" />

名字不区分大小写。已注册的一组是 `tensorboard`、`mlflow`、`wandb`、`comet`、
`clearml`、`neptune`、`dvclive` 和 `dvc`，最后一个是 `dvclive` 的别名。其他任何值
都会立刻报错，并列出合法的名字。没有哪个取值能一次打开全部，也没有对应的 CLI
参数：`loggers=` 是一个 Python 参数。

## 每个后端都记录什么

它们写出的指标名都一样，所以不管你选哪个，仪表盘看起来都是一样的：

| 键 | 值 |
|---|---|
| `train/loss` | 这一轮的平均训练损失 |
| `train/loss/<component>` | 家族上报的每一个损失分量 |
| `lr/<group>` | 优化器每个参数组的学习率 |
| `val/<metric>` | 每一个验证指标，`metrics/` 前缀已经去掉 |
| `time/epoch_seconds` | 这一轮的墙钟时间 |

step 就是轮次，从 1 开始计。完全解析后的训练配置会在训练开始时作为参数记录下来，
run 名默认是 `<family><size>-<task>`，例如 `yolo9s-detect`。

训练结束时，支持产物的后端会上传 `results.csv`、`train_config.yaml` 和
`summary.json`（存在的话），带 `log_checkpoints=True` 时还会加上
`weights/best.pt`。TensorBoard 什么都不传，因为它没有产物这个概念。没有哪个
logger 会上传验证绘图的图片。

## 失败时的行为

后端的包没装时会在构造阶段就报错，并给出安装命令，因为要了一个 logger 却悄悄什么
都没拿到，只会把 bug 藏起来。

运行期间的后端故障则正相反。处理函数抛出的第一个异常会在这次 run 剩下的时间里禁用
那个 logger，把异常记下来，把后端上的 run 按失败收尾，训练继续。跟踪服务器挂掉，
不会让你赔上这次训练。

## 各个后端

每个后端都需要它自己的 extra。

| 名字 | Extra | 构造函数 |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

这些类从 `libreyolo.training` 导入。

第一次跑之前值得知道的、各后端特有的几点：

TensorBoard 的事件文件默认写到 `<save_dir>/tensorboard`。用
`tensorboard --logdir runs/train` 查看。

MLflow 3.x 弃用了本地的 `./mlruns` 文件存储，除非设了
`MLFLOW_ALLOW_FILE_STORE=true`，否则会报错。想在本地不起服务地做跟踪，就像上面的
片段那样改传一个数据库 URI，再用 `mlflow ui --backend-store-uri sqlite:///mlflow.db`
读它。

Weights & Biases 会退回到 `WANDB_PROJECT` 环境变量，再退回到 `libreyolo`。Comet 会
退回到 `COMET_PROJECT_NAME`，再退回到 `libreyolo`，凭据取自它自己的配置；
`online=False` 给出一个离线实验。ClearML 会新建一个 task，把配置报在 `TrainConfig`
下，并关掉自动的框架捕获，这样指标不会被报两遍。Neptune 用的是当前的
`neptune-scale` 客户端，而不是旧的那个包，`mode="offline"` 会记录到本地。

DVCLive 写到 `<save_dir>/dvclive`。它按 `/` 构建自己的汇总树，没法在一个同时又是父
节点的路径上放一个浮点数，所以 `train/loss/box` 会写成 `train/loss.box`，而
`train/loss` 保留原名。LibreYOLO 还关掉了 DVCLive 那两个常规默认行为——保存一个 DVC
实验、写一个根目录下的 `dvc.yaml`，这样一个按需开启的 logger 不会在 run 目录之外留
下任何版本控制状态；传 `save_dvc_exp=True` 或一个显式的 `dvcyaml=` 就能把它们要
回来。

Neptune 被特意排除在 `libreyolo[all]` 之外：它的稳定版客户端要求 protobuf 低于 7，
而 TFLite extra 要求 protobuf 7。在没有 TFLite extra 的环境里安装
`libreyolo[neptune]`。

## 写一个回调

一切都由同样的这四个事件驱动。

<code-tabs name="callback" />

| 事件 | 时机 | 携带 |
|---|---|---|
| `TrainStartEvent` | 准备工作做完之后、第 1 轮之前 | `start_epoch`、`total_epochs`、`model_family`、`model_size`、`task`、`save_dir`、`config` |
| `TrainEpochEvent` | 每一轮的训练和验证都跑完之后 | `epoch`、`train_loss`、`train_loss_items`、`lr`、`val_metrics`、`validated`、`is_best`、`current_metric`、`best_metric`、`best_epoch`、`epoch_seconds` |
| `TrainEndEvent` | 训练完成之后 | `completed_epochs`、`final_loss`、`best_metric`、`best_epoch`、`total_seconds`、`results` |
| `TrainExceptionEvent` | 训练抛出异常时 | `epoch`、`exception`、`exception_type`、`exception_message`、`elapsed_seconds` |

一个普通可调用对象只会收到 `TrainEpochEvent`。一个对象可以实现
`on_train_start`、`on_train_epoch_end`、`on_train_end` 和 `on_train_exception` 里的
任意子集；没有的方法会被跳过。

`TrainStartEvent.config` 是完全解析后的配置，也就是用户传的 kwargs 与家族默认值合并
的结果，以一个只读映射的形式给出。这些事件是 frozen dataclass，它们的映射也是只读
的，所以回调没法靠往里写东西来改变这次 run。

从 `on_train_start`、`on_train_epoch_end` 或 `on_train_end` 抛出的异常会往上传播并
结束这次 run。只有 `on_train_exception` 被保护起来，这样它不会掩盖原本的那个失败。

多卡训练下，回调只在 rank 0 上触发。用自动的 DDP spawn 时，它们还必须可 pickle，也
就是说要写成模块级的类或函数，而不是闭包或 lambda。见[多卡训练](/docs/train/multi-gpu)。

## 每次 run 无论如何都会写的东西

有三个文件不需要任何配置就会落到 run 目录里，每个家族都一样：

| 文件 | 写入时机 | 内容 |
|---|---|---|
| `status.json` | 原子写入，每一轮以及开始、结束和失败时各写一次 | `state` 取值为 `running`、`completed` 或 `failed`，以及 `current_epoch`、`total_epochs`、`progress`、`eta_seconds`、最新的 `metrics`、`best_metric`、`best_epoch`，失败时还有一个 `error` 对象 |
| `metrics.jsonl` | 每轮追加一次 | 每轮一行 JSON，schema 与 `results.csv` 相同 |
| `train.log` | 实时 | 这次 run 的控制台输出 |

对于轮询一次 run 的脚本或 agent，`status.json` 是那个读起来最便宜的文件，而原子写入
意味着读的一方永远不会看到一个写了一半的文件。

`results.csv` 和 `summary.json` 是分开的，而且按家族开关。它们会为 YOLOv9、
YOLOv9-E2E、YOLOv9-P2、YOLOv7、YOLO-NAS、RF-DETR、EC 和 DINOv2 写出，其他家族则
不写。`results.csv` 每轮拿到一行，列是损失分量、验证指标和学习率，出现新列时它的
表头会变宽。断点续训时它会被裁回到续训那一轮之前的行，而不是把它们重复一遍。

除这些之外，训练器总是会在准备阶段写出 `train_config.yaml`，以及 `weights/` 下的
检查点（checkpoint）。

## 实时看一次 run

<code-tabs name="monitor" />

`libreyolo monitor` 只用标准库，就在上面这些文件之上提供一个浏览器仪表盘：指标图
表、日志尾部，以及所有验证图像，run 还在跑的时候会自动刷新。它是只读的，从不碰训练
进程，所以它可以挂上一次正在跑的 run、重新打开一次已经结束的 run，或者查看一次崩溃
的 run。

## 相关

- [验证与指标](/docs/train/validation)，讲 `val/` 这些键的含义，以及怎么加一条验证
  损失。
- [训练性能](/docs/train/performance)，讲 profiler，那是回答另一个问题的另一个
  工具。
