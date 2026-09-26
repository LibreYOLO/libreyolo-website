---
title: レンタルGPUでの学習
seo_title: レンタルしたクラウドGPUでLibreYOLOを学習
description: >-
  レンタルまたはサーバーレスGPUでLibreYOLOの学習ジョブを実行します：データの配置、インストール、起動、ライブ監視、重みの取得、課金停止までを説明します。
lead: >-
  レンタルGPUを使うと、学習実行は開始、終了、請求を伴うジョブになります。作業自体はローカル学習と同じです。異なるのは、データの搬入、外部からの監視、重みの搬出、マシンの停止です。
keywords:
  - クラウド gpu 学習
  - gpu レンタル
  - vast.ai 学習
  - modal サーバーレス gpu
  - beam gpu
  - リモート gpu 学習
  - hugging face データセット 配置
  - gpu エポック 費用
last_verified: 1.5.0
snippets:
  install:
    - label: マシン上
      language: bash
      code: |
        pip install libreyolo

        # 実行に必要なextraだけを追加。RF-DETR学習にはrfdetr、
        # パラメータ効率の高いファインチューニングにはlora、後でエクスポートするならonnx
        pip install "libreyolo[rfdetr,lora]"
    - label: 最初にGPUを確認
      language: python
      code: |
        import torch

        print(torch.__version__, torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        # 別アーキテクチャ向けのwheelでもTrueを返した後、最初の実カーネルで
        # 失敗するため、1つ実行
        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  stage:
    - label: 手元のマシンから1回だけ梱包してアップロード
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
    - label: 切断後もジョブが続くように分離して実行
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
                batch=64,          # 全GPU合計のグローバルバッチ
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: 低コストな1回の読み取り
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: スクリプトから
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: SSHトンネル経由でブラウザーから
      language: bash
      code: |
        # レンタルマシン上、デフォルトでは127.0.0.1:8420にバインド
        libreyolo monitor /root/runs/run1 --no-browser

        # 手元のマシンで実行し、ローカルのhttp://localhost:8420を開く
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

後で行うより今決めた方が安く済むことが2つあります。

まずデータセットをCDNに置いてください。1つのtarにまとめてHugging Faceのデータセット
リポジトリに置く方法なら、どのプロバイダーでも同じように動作し、すべての環境へ高速に配信
できます。リポジトリが非公開の場合も、ジョブ環境に`HF_TOKEN`を設定するだけです。自宅回線から
データセットをアップロードしたり、レンタルマシン上で低速な配信元から取得したりすると、待機中も
GPU時間として課金されます。

<code-tabs name="stage" />

次にディスク容量を決めます。ストレージに課金するプロバイダーは使用量ではなく割り当て容量で
請求し、ディスクは作成後に縮小できません。配置済みデータとチェックポイントを合計し、約30
パーセントの余裕を加えた容量に留めてください。

## レンタルマシンにインストール

<code-tabs name="install" />

イメージにGPUと一致するCUDAビルドが含まれていない場合は、先にPyTorch、次にLibreYOLOを
インストールしてください。これにより、pipが独自にCPU専用torchを解決するのを防ぎます。
2つ目のスニペットは省略できる形式的な確認ではありません。異なるGPUアーキテクチャ向けに
ビルドされた`wheel`は`torch.cuda.is_available() == True`を返した後、最初の実際の演算で
`CUDA error: no kernel image is available for execution on the device`として失敗します。
1回の行列乗算なら、1時間の設定後ではなく事前に検出できます。

プロバイダーがボリュームを提供する場合は、`HF_HOME`を永続ストレージへ向けてください。
チェックポイントとデータセットのダウンロードを実行間で維持できます。

## 起動

ジョブは分離して実行してください。ネットワーク接続とともに終了する対話セッションでは、学習も
終了します。

<code-tabs name="launch" />

ここでは特に`batch=-1`を使う価値があります。通常は初めて学習に使うGPUだからです。実際の
逆伝播を使って学習モードでモデルを調査し、収まる最大の2の累乗を選びます。20分後にメモリ不足
エラーで上限を知るより高速です。[ハイパーパラメータ](/docs/train/hyperparameters)を参照して
ください。

マルチGPUマシンでは、`device="0,1,2,3"`がGPUごとに1つのワーカーを自動生成し、`batch`は
全GPU合計のグローバルバッチのままです。各ワーカーがスクリプトを再インポートするため、
`__main__`ガードは必須です。これとその他の分散動作については
[マルチGPU学習](/docs/train/multi-gpu)を参照してください。

## 外部から監視

各実行は実行ディレクトリに`status.json`を書き出し、エポックごとにアトミックに再書き込みします。
これは低コストな読み取り手段です。ログを解析せず、状態、現在のエポック、残り時間、最新の指標を
数百バイトで取得できます。

<code-tabs name="watch" />

同じ場所にある`metrics.jsonl`にはエポックごとの完全な履歴があり、`train.log`にはコンソール
出力があります。`libreyolo monitor`は標準ライブラリだけを使って3つすべてをブラウザー
ダッシュボードで提供するため、LibreYOLO自体以外をマシンにインストールする必要はありません。
SSHポート転送でアクセスしてください。

いずれも学習プロセスには触れないため、実行中の処理に接続することも、完了済みの処理を再度
開くことも、クラッシュした処理を調べることもできます。

## 課金停止前に重みを取得

レンタルマシンは使い捨てです。終了時だけでなく節目ごとにチェックポイントを送信してください。
そうしないと、クラッシュ、プリエンプション、クレジット不足で実行全体を失います。

<code-tabs name="push" />

`weights/best.pt`と`weights/last.pt`は、エポックごとおよび改善のたびに書き出されます。
`save_period=N`を指定すると、さらに`weights/epoch_<N>.pt`のスナップショットが追加されるため、
実行途中でも低コストで送信できます。ファミリーが`summary.json`と`results.csv`を書き出す場合、
それらも小さく、取得する価値があります。

`on_train_epoch_end`のコールバックが送信を自動化する明確な方法です。
[実験ロガー](/docs/train/loggers)も参照してください。ホスト型バックエンドなら、マシンに一切
触れずに指標も取得できます。

## 課金を停止

問題が起きると実際に高額となる部分であり、規則はプロバイダーのモデルによって異なります。

生のマシンを借りるマーケットプレイスでは、インスタンスを破棄するまで実時間で課金されます。
アイドル状態のGPUも稼働中とまったく同じように課金されるため、学習プロセスだけを終了しても
費用は節約できません。停止済みインスタンスもディスク料金が発生します。

ジョブがデコレーター付き関数であるサーバーレスプラットフォームでは、関数が戻るとコンテナーが
ゼロにスケールダウンするため、忘れたマシンで課金される可能性は大幅に低くなります。それでも
タイムアウトなしで停止したジョブには課金されるため、必ず設定してください。

破棄せず停止することには実際の効果がある一方、落とし穴もあります。2026-07-31に250 GBの
ディスクを持つ8基のRTX 4090をレンタルして測定したところ、実行中は1時間あたり$3.4828、
停止中はディスクだけで1時間あたり$0.0694、破棄後は課金なしでした。環境、配置済みデータ、
チェックポイントを維持しながら98パーセント節約できます。

停止中の料金はレンタル前に計算できます。

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

再構築にかかる費用と比較してください。再度レンタルし、イメージを取得し、インストールして、
データを再配置する費用です。同じマシンでは、再構築に約15分の設定と43 GBの受信転送が必要で、
合計約$1.00でした。1時間あたり$0.0694と比較すると、約14時間以内に戻るなら停止が有利で、
それより長く空けるなら破棄して配置済みコピーから再構築する方が有利です。

希少なハードウェアでは、1つのリスクによって停止が安全でなくなります。停止するとGPUが解放され、
予約は維持されません。そのため、ホストに空きがある場合にだけ再起動できます。ディスクは安全でも、
GPUは確保されません。

## 関数としてのサーバーレス

マシンを管理したくない場合は、ModalとBeamのどちらもデコレーター付きPython関数をGPU上で
実行し、関数が戻るとゼロにスケールダウンします。LibreYOLO自身の夜間テストスイートはModal上で
動作し、ライブラリリポジトリ内の`tools/ci/modal_nightly.py`がコピーできる実用例です。

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # OpenCVシステムライブラリ
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
    cache.commit()                                # ボリュームを永続化


@app.local_entrypoint()
def main():
    train.remote()
```

`modal run modal_train.py`で実行します。コンテナーのファイルシステムは一時的なので、保持する
価値があるものはボリュームに置くか外部へ送信します。`timeout=`を明示的に設定してください。
停止した実行による際限のない請求を防ぐ唯一の設定です。

Beamも、`@function`デコレーター、`Volume`、`__main__`から呼び出す`train.remote()`という
同じ形式を取ります。

## ジョブあたりの費用で適正化

最適化すべき数値は1時間あたりの料金ではありません。小さなモデルでは大型GPUを半分アイドル
状態にするため、低価格で低速なGPUの方がエポックあたりの費用を抑えられる場合があります。
長時間の実行を確定する前に、レンタルしたGPUで数ステップのプロファイラーを実行してください。
判定が`dataloader`または`host / launch`なら、高速GPUを使っても効果はなく、ワーカー数または
バッチサイズを増やす方が大きな効果を得られます。
[学習パフォーマンス](/docs/train/performance)を参照してください。

## 関連項目

- 配置するアーカイブのレイアウトと、GPU課金開始前に問題を検出するdoctorコマンドについては[データセット](/docs/train/datasets)を参照してください。
- 複数GPUのマシンについては[マルチGPU学習](/docs/train/multi-gpu)を参照してください。
