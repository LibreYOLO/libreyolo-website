---
title: 実験ロガー
seo_title: "LibreYOLOの実験ロガーとコールバック"
description: "学習指標をTensorBoard、MLflow、Weights & Biases、Comet、ClearML、Neptune、DVCLiveへ送信し、4つの学習フックに独自のコールバックを記述します。"
lead: "学習可能なすべてのファミリーは4つの学習イベントを発行します。組み込みロガーは同じイベントを監視するコールバックオブジェクトなので、バックエンド統合とカスタムフックは1つのインターフェースを使います。"
keywords:
  - tensorboard 学習
  - mlflow 実験管理
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - 学習 コールバック
  - 学習 指標 csv
  - libreyolo monitor
last_verified: "1.5.0"
snippets:
  logger:
    - label: 名前で指定
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: 設定済みインスタンス
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
    - label: 通常の関数
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: 複数のフックを持つオブジェクト
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEndEvent, TrainEpochEvent, TrainStartEvent


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
    - label: ブラウザーで実行を監視
      language: bash
      code: |
        libreyolo monitor                     # runs/以下の最新実行
        libreyolo monitor runs/train/exp      # 指定した実行
---

## ロガーを有効化

`loggers=`は、登録名、設定済みインスタンス、または両者を混在させた反復可能オブジェクトを
受け取ります。

<code-tabs name="logger" />

名前は大文字と小文字を区別しません。登録済みの集合は`tensorboard`、`mlflow`、`wandb`、
`comet`、`clearml`、`neptune`、`dvclive`、`dvc`で、最後のものは`dvclive`の別名です。
それ以外は直ちに例外を発生させ、有効な名前を一覧表示します。すべてを有効にする値はなく、
CLIフラグもありません。`loggers=`はPython引数です。

## すべてのバックエンドが記録するもの

どれを選んでもダッシュボードが同じ形式になるよう、すべてが同じ指標名を書き出します。

| キー | 値 |
|---|---|
| `train/loss` | エポックの平均学習損失 |
| `train/loss/<component>` | ファミリーが報告する各損失成分 |
| `lr/<group>` | optimizerの各パラメータグループの学習率 |
| `val/<metric>` | `metrics/`プレフィックスを除去した各検証指標 |
| `time/epoch_seconds` | エポックの実時間 |

ステップは1から始まるエポックです。完全に解決された学習設定が学習開始時にパラメータとして
記録され、実行名のデフォルトは`<family><size>-<task>`です。たとえば`yolo9s-detect`です。

学習終了時、成果物に対応するバックエンドは、存在する場合に`results.csv`、`train_config.yaml`、
`summary.json`をアップロードし、`log_checkpoints=True`なら`weights/best.pt`もアップロードします。
TensorBoardには成果物という概念がないため、何もアップロードしません。検証プロット画像を
アップロードするロガーはありません。

## 失敗時の動作

バックエンドパッケージがない場合、構築時にインストールコマンドを示して例外を発生させます。
ロガーを要求したのに何も得られない状態を通知なく許すと、不具合が隠れるためです。

実行中にバックエンドが失敗した場合は逆の動作になります。ハンドラーで最初の例外が発生すると、
残りの実行ではそのロガーを無効にし、例外をログに記録し、バックエンドの実行を失敗として終了
しますが、学習は続行されます。追跡サーバーが停止しても、学習を失うことはありません。

## バックエンド

それぞれに専用のextraが必要です。

| 名前 | Extra | コンストラクター |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`、`dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

クラスは`libreyolo.training`からインポートします。

最初の実行前に知っておくべきバックエンド固有の注意事項は次のとおりです。

TensorBoardイベントファイルのデフォルト出力先は`<save_dir>/tensorboard`です。
`tensorboard --logdir runs/train`で表示します。

MLflow 3.xはローカルの`./mlruns`ファイルストアを非推奨とし、
`MLFLOW_ALLOW_FILE_STORE=true`がない場合に例外を発生させます。サーバーなしのローカル追跡では、
上のスニペットのように代わりにデータベースURIを渡し、
`mlflow ui --backend-store-uri sqlite:///mlflow.db`で読み取ります。

Weights & Biasesは`WANDB_PROJECT`環境変数、次に`libreyolo`へフォールバックします。Cometは
`COMET_PROJECT_NAME`、次に`libreyolo`へフォールバックし、認証情報を独自設定から取得します。
`online=False`はオフライン実験を作成します。ClearMLは新しいタスクを作成し、`TrainConfig`の
下に設定を報告し、指標が二重に報告されないようフレームワークの自動キャプチャを無効にします。
Neptuneは従来のパッケージではなく現在の`neptune-scale`クライアントを使い、`mode="offline"`は
ローカルに記録します。

DVCLiveは`<save_dir>/dvclive`へ書き出します。概要ツリーを`/`から構築しますが、親でもあるパスに
浮動小数点数を保持できないため、`train/loss`が名前を維持する一方、`train/loss/box`は
`train/loss.box`として書き出されます。LibreYOLOは、DVC実験の保存とルート`dvc.yaml`の書き出し
というDVCLiveの通常のデフォルトも無効にします。そのため、オプトインのロガーが実行ディレクトリ
外にバージョン管理状態を作ることはありません。元に戻すには、`save_dvc_exp=True`または明示的な
`dvcyaml=`を渡します。

Neptuneは意図的に`libreyolo[all]`から除外されています。安定版クライアントは7未満のprotobufを
必要とする一方、TFLiteのextraはprotobuf 7を必要とするためです。TFLiteのextraがない環境に
`libreyolo[neptune]`をインストールしてください。

## コールバックの記述

同じ4つのイベントがすべてを駆動します。

<code-tabs name="callback" />

| イベント | タイミング | 保持するもの |
|---|---|---|
| `TrainStartEvent` | 設定後、エポック1の前 | `start_epoch`、`total_epochs`、`model_family`、`model_size`、`task`、`save_dir`、`config` |
| `TrainEpochEvent` | 学習と検証を含む各エポック後 | `epoch`、`train_loss`、`train_loss_items`、`lr`、`val_metrics`、`validated`、`is_best`、`current_metric`、`best_metric`、`best_epoch`、`epoch_seconds` |
| `TrainEndEvent` | 学習完了後 | `completed_epochs`、`final_loss`、`best_metric`、`best_epoch`、`total_seconds`、`results` |
| `TrainExceptionEvent` | 学習が例外を発生させた場合 | `epoch`、`exception`、`exception_type`、`exception_message`、`elapsed_seconds` |

通常の呼び出し可能オブジェクトは`TrainEpochEvent`だけを受け取ります。オブジェクトは
`on_train_start`、`on_train_epoch_end`、`on_train_end`、`on_train_exception`の任意の組み合わせを
実装でき、存在しないメソッドはスキップされます。

`TrainStartEvent.config`は、ユーザーのkwargsとファミリーのデフォルトをマージした完全な解決済み
設定で、読み取り専用のマッピングです。イベントは凍結されたdataclassで、そのマッピングも
読み取り専用なので、コールバックが書き込みによって実行を変更することはできません。

`on_train_start`、`on_train_epoch_end`、`on_train_end`から発生した例外は伝播し、実行を終了します。
保護されるのは`on_train_exception`だけなので、元の失敗を隠すことはありません。

マルチGPU学習では、コールバックはrank 0だけで発生します。DDPの自動生成ではpickle化も可能で
なければならないため、クロージャーやlambdaではなくモジュールレベルのクラスまたは関数が必要です。
[マルチGPU学習](/docs/train/multi-gpu)を参照してください。

## 各実行が常に書き出すもの

すべてのファミリーで、設定なしでも3つのファイルが実行ディレクトリに置かれます。

| ファイル | 書き出すタイミング | 内容 |
|---|---|---|
| `status.json` | エポックごとと、開始時、終了時、失敗時にアトミックに書き出す | `running`、`completed`、`failed`のいずれかの`state`、`current_epoch`、`total_epochs`、`progress`、`eta_seconds`、最新の`metrics`、`best_metric`、`best_epoch`、失敗時の`error`オブジェクト |
| `metrics.jsonl` | エポックごとに1回追記 | エポックごとに1行のJSON。`results.csv`と同じスキーマ |
| `train.log` | リアルタイム | 実行のコンソール出力 |

`status.json`は実行をポーリングするスクリプトまたはエージェント向けの低コストな読み取り手段です。
アトミックな書き込みにより、読み取り側が書き込み途中のファイルを見ることはありません。

`results.csv`と`summary.json`は別で、ファミリーによって制限されます。YOLOv9、YOLOv9-E2E、
YOLOv9-P2、YOLOv7、YOLO-NAS、RF-DETR、EC、DINOv2では書き出されますが、他のファミリーでは
書き出されません。`results.csv`は損失成分、検証指標、学習率を列としてエポックごとに1行を
追加し、新しい列が現れるとヘッダーが広がります。再開時には行が重複せず、再開したエポックより
前まで切り詰められます。

これらとともに、学習器は設定時に必ず`train_config.yaml`を書き出し、`weights/`以下に
チェックポイントを書き出します。

## 実行をリアルタイム監視

<code-tabs name="monitor" />

`libreyolo monitor`は標準ライブラリだけを使い、上記のファイルからブラウザーダッシュボードを
提供します。指標チャート、ログ末尾、検証画像を表示し、実行中は更新されます。読み取り専用で
学習プロセスには一切触れないため、実行中の処理への接続、完了済み処理の再表示、クラッシュした
処理の調査ができます。

## 関連項目

- `val/`キーの意味と検証損失の追加方法については[検証と指標](/docs/train/validation)を参照してください。
- 別の問いを扱う別ツールであるプロファイラーについては[学習パフォーマンス](/docs/train/performance)を参照してください。
