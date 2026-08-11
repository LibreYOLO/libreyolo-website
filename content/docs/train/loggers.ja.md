---
title: 実験ロガー
seo_title: LibreYOLOの実験ロガーとコールバック
description: >-
  学習指標をTensorBoard、MLflow、Weights &
  Biases、Comet、ClearML、Neptune、DVCLiveへ送信し、4つの学習フックに独自のコールバックを作成します。
lead: >-
  学習可能なすべてのファミリーが4つの学習イベントを発行します。組み込みロガーは同じイベントを監視するコールバックオブジェクトなので、バックエンド統合とカスタムフックは1つのインターフェースを使用します。
keywords:
  - TensorBoard 学習
  - MLflow トラッキング
  - Weights & Biases
  - ClearML
  - Comet ML
  - Neptune
  - DVCLive
  - 学習 コールバック
  - 学習指標 CSV
  - LibreYOLO monitor
last_verified: 1.5.0
snippets:
  logger:
    - label: 名前で指定
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: 構成済みインスタンス
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
    - label: ブラウザーで実行を監視
      language: bash
      code: |
        libreyolo monitor                     # runs/にある最新の実行
        libreyolo monitor runs/train/exp      # 特定の実行
source_hash: de035acbaed32804
---

## ロガーを有効にする

`loggers=`は登録済みの名前、構成済みインスタンス、または両方を混在させた反復可能オブジェクトを受け取ります。

<code-tabs name="logger" />

名前では大文字と小文字を区別しません。登録済みの名前は`tensorboard`、`mlflow`、`wandb`、`comet`、`clearml`、`neptune`、`dvclive`、`dvc`です。最後の名前は`dvclive`のエイリアスです。それ以外を指定すると即座に例外が発生し、有効な名前が一覧表示されます。すべてを有効にする値はなく、CLIフラグもありません。`loggers=`はPython引数です。

## すべてのバックエンドが記録する内容

すべてが同じ指標名を書き込むため、どのバックエンドを選んでもダッシュボードの表示は同じです。

| キー | 値 |
|---|---|
| `train/loss` | エポックの平均学習loss |
| `train/loss/<component>` | ファミリーが報告する各loss成分 |
| `lr/<group>` | オプティマイザーの各パラメーターグループの学習率 |
| `val/<metric>` | `metrics/`プレフィックスを除いた各検証指標 |
| `time/epoch_seconds` | エポックの実時間 |

stepは1始まりのエポックです。完全に解決された学習構成が学習開始時にパラメーターとして記録され、実行名のデフォルトは`<family><size>-<task>`です。たとえば`yolo9s-detect`になります。

学習終了時、アーティファクトに対応するバックエンドは、存在する場合に`results.csv`、`train_config.yaml`、`summary.json`をアップロードします。`log_checkpoints=True`なら`weights/best.pt`もアップロードします。TensorBoardにはアーティファクトの概念がないため、何もアップロードしません。検証プロット画像をアップロードするロガーはありません。

## 障害時の動作

バックエンドパッケージがない場合は構築時に例外が発生し、インストールコマンドが表示されます。ロガーを要求したのに通知なく何も得られない動作では、不具合が隠れるためです。

実行中にバックエンドで障害が発生した場合は逆に動作します。ハンドラーで最初の例外が発生すると、その実行の残りではロガーを無効にし、例外をログへ記録して、バックエンドの実行を失敗として終了します。学習自体は継続します。追跡サーバーの停止によって学習結果を失うことはありません。

## バックエンド

それぞれ独自の追加パッケージが必要です。

| 名前 | 追加パッケージ | コンストラクター |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`、`dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

クラスは`libreyolo.training`からインポートします。

初回実行前に知っておくとよいバックエンド固有の注意点があります。

TensorBoardのイベントファイルはデフォルトで`<save_dir>/tensorboard`に保存されます。`tensorboard --logdir runs/train`で表示します。

MLflow 3.xではローカルの`./mlruns`ファイルストアが非推奨になり、`MLFLOW_ALLOW_FILE_STORE=true`でなければ例外が発生します。サーバーを使わずローカルで追跡する場合は、上のスニペットのようにデータベースURIを渡し、`mlflow ui --backend-store-uri sqlite:///mlflow.db`で読み取ってください。

Weights & Biasesは`WANDB_PROJECT`環境変数、次に`libreyolo`へフォールバックします。Cometは`COMET_PROJECT_NAME`、次に`libreyolo`へフォールバックし、認証情報は自身の構成から取得します。`online=False`でオフライン実験になります。ClearMLは新しいタスクを作成し、`TrainConfig`の下へ構成を報告します。指標が二重に報告されないよう、フレームワークの自動取得は無効にします。Neptuneは従来のパッケージではなく現在の`neptune-scale`クライアントを使用し、`mode="offline"`ではローカルに記録します。

DVCLiveは`<save_dir>/dvclive`へ書き込みます。要約ツリーを`/`から構築し、親でもあるパスに浮動小数点値を保持できないため、`train/loss`は名前を維持しますが、`train/loss/box`は`train/loss.box`として書き込まれます。またLibreYOLOは、DVC実験を保存してルートの`dvc.yaml`を書き込むDVCLiveの通常のデフォルトを無効にします。このため、明示的に有効化したロガーが実行ディレクトリ外にバージョン管理状態を作ることはありません。元に戻すには`save_dvc_exp=True`または明示的な`dvcyaml=`を渡してください。

Neptuneは意図的に`libreyolo[all]`から除外されています。安定版クライアントはprotobuf 7未満を要求する一方、TFLiteの追加パッケージはprotobuf 7を要求するためです。TFLite追加パッケージがない環境へ`libreyolo[neptune]`をインストールしてください。

## コールバックの作成

同じ4つのイベントがすべてを駆動します。

<code-tabs name="callback" />

| イベント | タイミング | 保持する値 |
|---|---|---|
| `TrainStartEvent` | セットアップ後、エポック1の前 | `start_epoch`、`total_epochs`、`model_family`、`model_size`、`task`、`save_dir`、`config` |
| `TrainEpochEvent` | 各エポックの学習と検証後 | `epoch`、`train_loss`、`train_loss_items`、`lr`、`val_metrics`、`validated`、`is_best`、`current_metric`、`best_metric`、`best_epoch`、`epoch_seconds` |
| `TrainEndEvent` | 学習完了後 | `completed_epochs`、`final_loss`、`best_metric`、`best_epoch`、`total_seconds`、`results` |
| `TrainExceptionEvent` | 学習で例外が発生した場合 | `epoch`、`exception`、`exception_type`、`exception_message`、`elapsed_seconds` |

通常のcallableは`TrainEpochEvent`だけを受け取ります。オブジェクトは`on_train_start`、`on_train_epoch_end`、`on_train_end`、`on_train_exception`の任意の組み合わせを実装でき、存在しないメソッドはスキップされます。

`TrainStartEvent.config`は完全に解決された構成です。ユーザーのキーワード引数とファミリーのデフォルトを統合した、読み取り専用マッピングとして提供されます。イベントは固定データクラスであり、そのマッピングも読み取り専用なので、コールバックから値を書き込んで実行を変更することはできません。

`on_train_start`、`on_train_epoch_end`、`on_train_end`で発生した例外は伝播し、実行を終了します。保護されるのは`on_train_exception`だけなので、元の障害を隠すことはありません。

マルチGPU学習では、コールバックはrank 0だけで発火します。自動DDP spawnではpickle化も可能でなければならないため、クロージャーやlambdaではなく、モジュールレベルのクラスまたは関数を使用します。[マルチGPU学習](/docs/train/multi-gpu)を参照してください。

## すべての実行が常に書き込むもの

どのファミリーでも、構成なしで3つのファイルが実行ディレクトリに保存されます。

| ファイル | 書き込み | 内容 |
|---|---|---|
| `status.json` | 各エポックと開始、終了、失敗時にアトミックに書き込み | `running`、`completed`、`failed`のいずれかの`state`、`current_epoch`、`total_epochs`、`progress`、`eta_seconds`、最新の`metrics`、`best_metric`、`best_epoch`、失敗時の`error`オブジェクト |
| `metrics.jsonl` | エポックごとに1回追記 | エポックごとに1行のJSON。`results.csv`と同じスキーマ |
| `train.log` | リアルタイム | 実行のコンソール出力 |

`status.json`は、実行をポーリングするスクリプトやエージェントが低コストで読み取れるファイルです。アトミック書き込みにより、読み取り側に書き込み途中のファイルが見えることはありません。

`results.csv`と`summary.json`は別で、ファミリーによって書き込みが制限されます。YOLOv9、YOLOv9-E2E、YOLOv9-P2、YOLOv7、YOLO-NAS、RF-DETR、EC、DINOv2では書き込まれ、その他のファミリーでは書き込まれません。`results.csv`には、loss成分、検証指標、学習率を列として、エポックごとに1行が追加されます。新しい列が現れるとヘッダーが拡張されます。再開時には行を重複させず、再開するエポックより前の行まで切り詰められます。

これらに加え、トレーナーはセットアップ時に必ず`train_config.yaml`を、`weights/`の下にチェックポイントを書き込みます。

## 実行をリアルタイムで監視

<code-tabs name="monitor" />

`libreyolo monitor`は標準ライブラリだけを使い、上記のファイルをブラウザーダッシュボードで提供します。指標チャート、ログ末尾、検証画像があればそれらを表示し、実行中は更新します。読み取り専用で学習プロセスには触れないため、実行中の処理へ接続したり、完了済みの実行を再度開いたり、異常終了した実行を調査したりできます。

## 関連項目

- `val/`キーの意味と検証lossの追加方法については、[検証と指標](/docs/train/validation)を参照してください。
- 別の目的を持つツールであるプロファイラーについては、[学習パフォーマンス](/docs/train/performance)を参照してください。
