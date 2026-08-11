---
title: レンタルGPUでの学習
seo_title: レンタルクラウドGPUでLibreYOLOを学習
description: >-
  レンタルGPUまたはサーバーレスGPUでLibreYOLOの学習ジョブを実行します。データの準備、インストール、起動、リアルタイム監視、重みの取得、課金の停止までを説明します。
lead: >-
  レンタルGPUでは、学習が開始、終了、請求を伴うジョブになります。作業自体はローカル学習と同じですが、データの搬入、外部からの監視、重みの搬出、マシンの停止が加わります。
keywords:
  - クラウドGPU 学習
  - GPU レンタル
  - Vast.ai 学習
  - Modal サーバーレス GPU
  - Beam GPU
  - リモート学習
  - Hugging Face データセット 配置
  - GPU エポック コスト
last_verified: 1.5.0
snippets:
  install:
    - label: レンタルマシン上
      language: bash
      code: |
        pip install libreyolo

        # 実行に必要な追加パッケージだけを加えます。RF-DETRの学習にはrfdetr、
        # パラメーター効率の高いファインチューニングにはlora、後のエクスポートにはonnxです。
        pip install "libreyolo[rfdetr,lora]"
    - label: 最初にGPUを確認
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # 別のアーキテクチャ向けのwheelでもTrueを返し、最初の実カーネルで
        # 失敗するため、1つ実行します。
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: 手元のマシンから一度だけ圧縮してアップロード
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: レンタルマシン上に配置
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
    - label: 切断後もジョブが継続するようバックグラウンドで起動
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: PythonファイルからマルチGPUで実行
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # すべてのGPUにわたるグローバルバッチ
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: 低コストな1回の読み取り
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: スクリプトから確認
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: SSHトンネル経由でブラウザーから確認
      language: bash
      code: |
        # レンタルマシン上で実行します（デフォルトでは127.0.0.1:8420へbind）。
        libreyolo monitor /root/runs/run1 --no-browser

        # 手元のマシンで実行し、http://localhost:8420をローカルで開きます。
        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: 重みを永続的な場所へ送信
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## レンタル前の準備

事前に決めておくと、後からのコストを大きく抑えられる項目が2つあります。

まずデータセットをCDNへ配置してください。Hugging Faceのデータセットリポジトリに単一のtarとして圧縮する方法は、どのプロバイダーでも同じように使え、高速に配信されます。非公開リポジトリでも、ジョブ環境に必要なのは`HF_TOKEN`だけです。自宅回線からデータセットをアップロードしたり、レンタルマシン上で遅い配信元から取得したりすると、待ち時間にもGPU料金がかかります。

<code-tabs name="stage" />

次にディスク容量を決めます。ストレージ課金を行うプロバイダーは使用量ではなく割り当て容量で請求し、作成後にディスクを縮小できません。配置するデータ、チェックポイント、約30%の余裕を合計し、それ以上は確保しないでください。

## レンタルマシンへのインストール

<code-tabs name="install" />

イメージにGPUと一致するCUDAビルドが含まれない場合は、先にPyTorch、次にLibreYOLOをインストールしてください。そうしないと、pipがCPU専用のPyTorchを解決する可能性があります。2番目のスニペットは単なる形式的な確認ではありません。異なるGPUアーキテクチャ向けにビルドされたwheelでも`torch.cuda.is_available() == True`を返し、最初の実処理で`CUDA error: no kernel image is available for execution on the device`として失敗します。1回の行列積なら、1時間のセットアップ後ではなく事前に検出できます。

プロバイダーがvolumeを提供する場合は、`HF_HOME`を永続ストレージへ向けてください。チェックポイントとデータセットのダウンロードを実行間で維持できます。

## 起動

ジョブはバックグラウンドで実行してください。ネットワーク接続とともに終了する対話セッションでは、学習も終了します。

<code-tabs name="launch" />

ここでは特に`batch=-1`が有効です。通常、これまで学習に使ったことのないGPUを使用するためです。実際の逆伝播を使って学習モードのモデルを調べ、収まる最大の2の累乗を選択します。20分後にメモリ不足エラーで上限を知るより高速です。[ハイパーパラメーター](/docs/train/hyperparameters)を参照してください。

マルチGPUマシンでは、`device="0,1,2,3"`を指定するとGPUごとに1つのワーカーが自動的に起動し、`batch`はすべてにわたるグローバルバッチのままです。各ワーカーがスクリプトを再インポートするため、`__main__`ガードは必須です。この点とその他の分散動作は[マルチGPU学習](/docs/train/multi-gpu)にあります。

## 外部から監視する

すべての実行は、実行ディレクトリに`status.json`を書き込み、エポックごとにアトミックに更新します。ログを解析せず、数百バイトの読み取りだけで状態、現在のエポック、残り時間、最新指標を取得できます。

<code-tabs name="watch" />

隣の`metrics.jsonl`にはエポックごとの完全な履歴があり、`train.log`にはコンソール出力があります。`libreyolo monitor`は標準ライブラリだけを使用して3つすべてをブラウザーダッシュボードで提供するため、マシン上でLibreYOLO以外をインストールする必要はありません。SSHポートフォワーディング経由で接続してください。

いずれも学習プロセスには触れないため、実行中の処理へ接続したり、完了済みの実行を再度開いたり、異常終了した実行を調査したりできます。

## 課金を止める前に重みを取り出す

レンタルマシンは一時的なものです。最後だけでなく節目ごとにチェックポイントを送信してください。そうしないと、クラッシュ、プリエンプション、クレジット不足によって実行全体を失う可能性があります。

<code-tabs name="push" />

`weights/best.pt`と`weights/last.pt`は各エポックと改善時に書き込まれます。`save_period=N`を指定すると`weights/epoch_<N>.pt`スナップショットも加わり、実行途中の送信が容易になります。ファミリーが書き込む場合は、サイズの小さい`summary.json`と`results.csv`も取得する価値があります。

`on_train_epoch_end`のコールバックが送信の自動化に適した方法です。[実験ロガー](/docs/train/loggers)では、ホスト型バックエンドを使い、マシンへ直接アクセスせず指標を取得する方法も説明しています。

## 課金を止める

失敗すると実際に大きな料金が発生する部分であり、プロバイダーのモデルによってルールが異なります。

生のマシンを借りるマーケットプレイスでは、インスタンスを破棄するまで実時間で課金されます。アイドル状態のGPUも使用中と同じ料金なので、学習プロセスを終了するだけでは料金を節約できません。停止したインスタンスでもディスク料金は発生します。

ジョブをデコレーター付き関数として実行するサーバーレスプラットフォームでは、関数が戻るとコンテナがゼロまでスケールダウンするため、放置したマシンによる課金は起こりにくくなります。それでも、タイムアウトのないハングしたジョブには料金がかかるため、必ず設定してください。

破棄ではなく停止する方法は有効ですが、注意も必要です。2026-07-31に、250 GBディスクを持つレンタル8x RTX 4090で測定した料金は、実行中が1時間$3.4828、停止中がディスクだけで1時間$0.0694、破棄後は料金なしでした。環境、配置済みデータ、チェックポイントを維持しながら98%節約できます。

停止中の料金はレンタル前に計算できます。

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

再構築のコストと比較してください。再レンタル、イメージの取得、インストール、データの再配置が含まれます。同じマシンでは、再構築に約15分のセットアップと43 GBの受信転送が必要で、合計約$1.00でした。1時間$0.0694と比べると、約14時間以内に戻るなら停止、それより長く空くなら配置済みコピーから破棄して再構築する方が有利です。

希少なハードウェアでは、停止が安全ではない理由が1つあります。停止するとGPUが解放され、予約は維持されません。再起動はホスト上でGPUがまだ空いている場合だけ成功します。ディスクは安全ですが、GPUは確保されません。

## 関数としてのサーバーレス実行

マシンを管理したくない場合、ModalとBeamはどちらもデコレーター付きPython関数をGPUで実行し、関数が戻るとゼロまでスケールダウンします。LibreYOLO自身のnightlyテストスイートはModal上で動作しており、ライブラリリポジトリの`tools/ci/modal_nightly.py`がコピーできる実例です。

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # OpenCVのシステムライブラリ
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # 実行間で重みをキャッシュ

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # volumeを永続化


@app.local_entrypoint()
def main():
    train.remote()
```

`modal run modal_train.py`で実行します。コンテナのファイルシステムは一時的なので、保持する必要があるものはvolumeに置くか外部へ送信してください。`timeout=`を明示的に設定します。ハングした実行による無期限の課金を防ぐ唯一の仕組みです。

Beamも同じ構成で、`@function`デコレーター、`Volume`、`__main__`から呼び出す`train.remote()`を使用します。

## ジョブ単位のコストで適切なサイズを選ぶ

最適化すべき数値は$/hrではありません。小規模モデルは大きなGPUを半分しか利用できないため、安価で低速なGPUの方がエポックあたりの料金を抑えられることがよくあります。長時間の実行を開始する前に、レンタルGPU上でプロファイラーを数step実行してください。判定が`dataloader`または`host / launch`なら、より高速なGPUを選んでも効果はなく、ワーカーやバッチを増やす方が有効です。[学習パフォーマンス](/docs/train/performance)を参照してください。

## 関連項目

- 配置するアーカイブの構成と、GPU課金前に問題を検出するdoctorコマンドについては[データセット](/docs/train/datasets)を参照してください。
- 複数GPUを搭載するマシンについては[マルチGPU学習](/docs/train/multi-gpu)を参照してください。
