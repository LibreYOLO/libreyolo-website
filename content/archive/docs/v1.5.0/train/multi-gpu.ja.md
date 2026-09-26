---
title: マルチGPU学習
seo_title: LibreYOLOのマルチGPU学習
description: >-
  device="0,1"を指定して複数GPUで学習します。ライブラリによるDDPワーカーの起動、batchがグローバルバッチになる理由、sync_bnを設定する場面、torchrun経路を解説します。
lead: >-
  LibreYOLOのマルチGPU学習はPyTorch
  DistributedDataParallelです。GPUごとに1つのプロセスが完全なモデルの複製と各バッチの一部を保持し、各ステップでランク間の勾配を平均します。
keywords:
  - PyTorch DDP 学習
  - マルチGPU 学習
  - torchrun nproc_per_node
  - DistributedDataParallel
  - SyncBatchNorm
  - グローバルバッチサイズ
  - NCCL Gloo バックエンド
  - Windows マルチGPU
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # __main__ガードは必須 各ワーカーがこのモジュールを再インポートするため
        # ガードがないと学習の起動を再帰的に繰り返す
        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # グローバルバッチ 2GPUなら各GPUに16枚
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
    - label: 起動
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
            # GPU 0で一度調査しworld sizeの倍数に調整する
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## 2台のGPUで実行

デバイスのリストを渡します。それ以外は変わりません。

<code-tabs name="train" />

複数のデバイスが指定され、torchrun環境ではない場合、モデルの`train()`は重みを一時ファイルへ保存します。要求されていればautobatchを解決し、`torch.multiprocessing.spawn`でGPUごとに1つのワーカープロセスを起動します。各ワーカーはモデルクラスを再インポートし、保存済みの重みから再構築して、通常の単一デバイス経路を実行します。起動済みワーカー内部ではtorchrunの環境変数が設定されるためです。実行が終わると、ランク0の最良チェックポイントが呼び出し元のモデルインスタンスへ読み戻されます。

`device`には`"0,1"`、`[0, 1]`、`0`、`"cuda:0"`、`"cpu"`、`"mps"`、`"auto"`を指定できます。複数のCUDAインデックスを含むリストだけが起動処理を開始します。

## `__main__`ガードは必須

起動したワーカーは元のモジュールを再インポートします。`if __name__ == "__main__":`ガードがないと、そのインポートで学習呼び出しが再実行され、各ワーカーがさらにワーカーを起動します。ライブラリはこの状態を検出し、再帰させずに例外を送出します。

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

ワーカーへ渡すものはすべてpickle化されるため、`callbacks=`はpickle化可能でなければなりません。モジュール直下のクラスは使用できますが、クロージャやラムダは使用できません。エラーにはその理由と、代替となる組み込みロガーが示されます。

## batchはグローバルバッチ

`batch`は全GPUを合計したオプティマイザーステップごとの画像数です。各ランクのデータローダーは`DistributedSampler`を使って`batch // world_size`で構築されるため、2台のGPUで`batch=32`ならGPUごとに16枚であり、32枚ではありません。

バッチがworld sizeで割り切れない場合は、暗黙に別のサイズで学習せず例外を送出します。

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

勾配はDDP自体が平均するため、損失はスケーリングせず渡されます。さらにworld sizeを掛けると、有効学習率がGPU数とほぼ同じ倍率で増えてしまいます。

## DDPでのAutobatch

`batch=-1`を使用でき、world sizeで割り切れるグローバルバッチを返します。

<code-tabs name="autobatch" />

自動起動経路では、ワーカーが存在する前に親プロセスが最初のデバイスで調査するため、各ワーカーは具体的な整数を受け取り、プロセス間の調整は不要です。torchrunではランク0が調査し、結果を1つのlongテンソルとしてブロードキャストします。

調査処理は1台のGPUの容量を測定し、world sizeを掛けます。`nbs`を設定している場合、グローバルバッチは`nbs`を上限としてworld sizeの倍数へ切り下げられます。そのためGPUを追加すると、GPUごとのバッチを小さくするのではなく、勾配累積ステップ数を減らします。調査自体の仕組みは[ハイパーパラメーター](/docs/train/hyperparameters)を参照してください。

## SyncBatchNorm

DDPでは、各ランクのBatchNormレイヤーが見るのは自身の分割だけです。`batch // world_size`による分割が小さすぎると、実行時統計の品質が低下し、単一GPU実行より収束後のモデル性能が悪化する可能性があります。

`sync_bn=True`はすべてのBatchNormをSyncBatchNormへ変換し、グローバルバッチ全体で統計を計算します。変換は分散処理が有効な場合だけ行われるため、単一GPUではフラグの有無にかかわらず影響しません。

BatchNormを多用する畳み込みファミリーでは、すでにデフォルトで有効です。対象はYOLOX、YOLOv7、YOLOv9とそのバリアント、YOLO-NAS、PicoDet、RTMDet、FOMOです。それ以外のファミリーではデフォルトで無効です。モデルにBatchNormがあり、`sync_bn`が無効で、ランクごとのバッチが16未満の場合はトレーナーが警告します。

<code-tabs name="syncbn" />

`sync_bn`に対応するCLIフラグはありません。Python引数です。

## torchrunで起動

torchrunも使用でき、クラスタースケジューラーがすでにプロセス起動を管理している場合に適しています。単一デバイス用のスクリプトを書き、torchrunにランク環境を設定させます。

<code-tabs name="torchrun" />

2つの方法を組み合わせないでください。torchrun環境が存在するとき、`device="0,1"`はプロセスを起動しません。トレーナーは`cuda:LOCAL_RANK`を使用し、プロセス数はtorchrunが管理します。

## ランクごとの動作

すべての副作用はランク0が担当します。実行ディレクトリを解決してその名前をブロードキャストすることで全ランクを一致させ、チェックポイントとアーティファクトを書き込み、ユーザーのコールバックとロガーを実行します。他のランクは学習し、勾配に寄与します。

各ランクのデータローダーとデータ拡張の乱数生成器には、設定済みの`seed`から導出した異なるシードが使われます。そのため、複数ランクが同じデータ拡張を選ぶことはありません。

## プラットフォームとバックエンド

バックエンドは自動選択されます。CUDAとNCCLの両方が利用できる場合はNCCL、それ以外はGlooです。Windows向けにはNCCLがビルドされていないため、設定なしでGlooが使用されます。プロセスグループは3時間のタイムアウトで初期化されます。

## DDPで実行されないもの

- CUDAグラフのキャプチャ。`cuda_graph=True`はログを1行出し、通常実行で学習します。[学習パフォーマンス](/docs/train/performance)を参照してください。
- 学習プロファイラー。`profile=True`は警告とともに無視されます。

すべてのファミリーが自動起動に対応するわけではありません。学習可能な検出、分類、セマンティック、復元ファミリーを含む24ファミリーが対応します。未対応のファミリーにマルチGPUデバイスを渡すと、暗黙に1台のGPUで学習せず、モデルAPIとtorchrunコマンドを示すエラーを送出します。

## 関連項目

- `batch`、`nbs`、再開については[ハイパーパラメーター](/docs/train/hyperparameters)を参照してください。
- コールバックのpickle化可能性に関する制約については[実験ロガー](/docs/train/loggers)を参照してください。
- マルチGPUマシンのレンタルについては[クラウドGPU](/docs/train/cloud-gpus)を参照してください。
