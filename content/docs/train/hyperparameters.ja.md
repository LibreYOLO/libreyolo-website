---
title: ハイパーパラメーター
seo_title: LibreYOLOの学習ハイパーパラメーター
description: >-
  重要なtrain()引数を説明します。epochs、batch、lr0、optimizer、EMA、自動バッチ、勾配累積、再開と、ファミリーごとにデフォルトが異なる理由を扱います。
lead: >-
  すべての学習引数はTrainConfigデータクラスのフィールドです。基本クラスがフィールドとデフォルト値を定義し、各モデルファミリーがサブクラスで、公開レシピに応じたデフォルトをオーバーライドします。
keywords:
  - train 引数
  - 学習率
  - バッチサイズ
  - 自動バッチ
  - 指数移動平均
  - 勾配累積
  - 学習 再開
  - 早期停止 patience
  - AMP bfloat16
  - 学習設定 YAML
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: ファミリーの解決済みデフォルトを確認
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: |
        # ファミリーのオーバーライドを含むtrain、val、predictのデフォルトを表示します。
        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # batch=-1はGPUメモリを調べ、具体的な2の累乗へ解決します。
        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # オプティマイザーの1stepごとにサイズ16のマイクロバッチを4つ処理し、有効バッチは64です。
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 中断した実行のチェックポイントを読み込み、再開を指定します。
        model = LibreYOLO("runs/train/exp/weights/last.pt")
        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # yaml内のキーはTrainConfigのフィールド名です。明示的なキーワード引数が優先されます。
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## 引数の設定

`train()`はキーワード引数を受け取り、CLIは同じ名前を`key=value`形式で受け取ります。

<code-tabs name="train" />

どちらの経路も同じ場所に到達します。キーワード引数は`TrainConfig.from_kwargs()`へ渡され、ファミリーの構成データクラスが作成されます。

## タイプミスでは例外が発生しません

`from_kwargs()`は構成のフィールドではないキーを破棄し、その名前を含む`UserWarning`を発行します。その後、学習はデフォルト値を使って開始します。

```python
# UserWarning: Unknown training config keys (ignored): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

処理は失敗せず、実行は完了しますが、学習率は呼び出し側が指定した値になりません。新しいレシピの最初のエポックでは警告を確認してください。CLIでは構成を作成する前にフラグ名を検証するため、より厳密で、スペルを誤ったCLIフラグは明示的に拒否されます。

## デフォルトはファミリーごとに異なります

`TrainConfig`はフィールドと基本のデフォルトを定義します。各ファミリーがサブクラス化し、公開レシピで変更される値をオーバーライドするため、「デフォルトの学習率」に単一の正解はありません。

基本のデフォルトは`optimizer="sgd"`、`lr0=0.01`、`momentum=0.937`、`weight_decay=5e-4`、`scheduler="yoloxwarmcos"`、`epochs=300`、`batch=16`、`imgsz=640`、`amp=True`です。ファミリーによる違いを3つの例で示します。

| フィールド | 基本 | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINEとDEIMは`amp=False`で提供されます。D-FINEのデコーダーが、float16で表現できる最大の有限値65504に活性値を制限するためです。YOLO-NASとFOMOもデフォルトで無効です。CLIの`--amp`フラグはすべてのファミリーでデフォルトが`True`なので、ユーザー指定として扱われ、ファミリーのデフォルトを上書きします。変更する意図がない限り、そのままにしてください。

推測せず、ファミリーの実際のデフォルトを確認するには次のようにします。

<code-tabs name="defaults" />

## バッチサイズ

`batch`はグローバルバッチです。マルチGPU学習では各rankが`batch // world_size`を読み込むため、指定する数値はGPUの数にかかわらず、オプティマイザーの1stepあたりの画像数です。[マルチGPU学習](/docs/train/multi-gpu)を参照してください。

`batch=-1`で自動バッチを有効にします。トレーナーは学習モードのモデルに対して、2の累乗のサイズで実際の逆伝播を実行し、メモリ曲線に直線を当てはめます。推定値のうち、VRAM総量の60%以内に収まる値を厳密に下回る、最大の2の累乗を選択します。

<code-tabs name="autobatch" />

逆伝播を含む学習モードで調査することが重要です。推論モードの調査では保持される活性値や勾配テンソルを見落とし、深いCNNではこれらが推論時の占有量の数倍になるためです。RF-DETRは目標比率を45%へ下げます。調査で使う合成の逆伝播でも、損失関数と補助デコーダー層のコストを少なく見積もるためです。

自動バッチはCUDAの機能です。CPUまたはMPSでは1行のログを出し、デフォルトバッチを維持します。

## 勾配累積

`nbs`は基準となる有効バッチサイズを設定します。トレーナーはオプティマイザーの1stepごとに`round(nbs / batch)`個のマイクロバッチを累積します。

<code-tabs name="accumulate" />

デフォルトの`None`のままにすると累積は無効で、学習は変わりません。

## 学習率とスケジュール

`lr0`は初期学習率で、`optimizer`は`sgd`、`adam`、`adamw`を受け付けます。`momentum`はSGDのmomentumまたはAdamのbeta1、`weight_decay`はL2項で、`nesterov`はSGDに適用されます。

スケジュールは`scheduler`、`warmup_epochs`、`warmup_lr_start`、`min_lr_ratio`によって形成されます。`no_aug_epochs`は強いデータ拡張なしで実行する最後のエポック数を設定します。また、複数のスケジュールが終盤の形成にも使用するため、単なるデータ拡張設定ではありません。各ファミリーがデータ拡張部分をどう扱うかは[データ拡張](/docs/train/augmentations)にあります。

一部のファミリーには独自の学習率設定があります。`backbone_lr_mult`はヘッドに対するバックボーングループの倍率、`clip_max_norm`は勾配クリッピングを設定し、SegFormerは`head_lr_mult`を使ってデコードヘッドをバックボーンの10倍の学習率で実行します。これらは基本クラスではなく、ファミリーの構成サブクラスにあります。

## EMA

`ema=True`では、学習済みの重みと並行して指数移動平均を維持します。FOMOを除き、すべてでデフォルト有効です。

`ema_decay`は目標減衰率です。減衰率は最初から目標値を使わず、徐々に増加します。更新`n`での実効値は`ema_decay * (1 - exp(-n / tau))`で、`tau`のデフォルトは2000です。このため、初期の更新ではモデルへ密接に追従し、後期の更新では平滑化します。ファミリーのデフォルトは、YOLO-NAS poseの`0.997`、YOLOXの`0.9998`、YOLOv9とDETR系の`0.9999`などです。

検証にはEMA重みが使われ、`best.pt`と`last.pt`にも格納されます。生の学習済み重みも`train_model`キーの下に保存されるため、再開時には平均からではなく、学習軌跡から継続します。

## 精度

`amp=True`はCUDA autocast下で順伝播を実行します。`amp_dtype`は`float16`（デフォルト）または`bfloat16`を選び、`fp16`と`bf16`も表記として受け付けます。

Float16には動的loss scalingが必要で、有効な`GradScaler`が作成されます。Bfloat16は指数範囲が広いため不要ですが、オプティマイザーの経路を同一に保つため、無効化されたscalerを構築します。bfloat16に対応しないCUDAデバイスで要求すると、通知なく精度を下げず、セットアップ時に例外が発生します。

## 出力、チェックポイント、停止

実行結果は`project/name`に書き込まれます。`project`のデフォルトはすべて`runs/train`ですが、`name`はファミリーごとにオーバーライドされます。基本のデフォルトは`exp`、YOLOv9は`yolo9_exp`、D-FINEは`dfine_exp`です。デフォルトの`exist_ok=False`では、既存ディレクトリを上書きせず、連番サフィックスを付けます。

`save_period`は、各エポック後の`weights/last.pt`と、追跡対象の指標が改善したときの`weights/best.pt`に加え、Nエポックごとに`weights/epoch_<N>.pt`を書き込みます。`eval_interval`は検証の実行間隔を設定し、`patience`は改善のないエポックが指定数続いたとき実行を停止します。`0`で早期停止を無効にします。

`cache`は、デコード済み画像をRAM（`True`または`"ram"`）に保持するか、ソースの隣に`.npy`ファイルとして保存（`"disk"`）し、繰り返すエポックを高速化します。キャッシュからの読み取りは新規読み取りとバイト単位で同一です。データローダーワーカーを使う場合、`"disk"`の方が安全です。

## 再開

`resume=True`は中断した実行を継続します。再開は別の引数ではなくモデルからチェックポイントを読み取るため、先にチェックポイントを読み込む必要があります。

<code-tabs name="resume" />

再開では、学習済み重み、オプティマイザー状態、EMA重みと更新回数、最良指標の追跡、`GradScaler`のscale、PyTorch、CUDA、NumPyの乱数状態を復元します。チェックポイントのエポックに1を加えた位置から開始し、スケジュールをその位置まで早送りします。

行わないことが2つあります。`resume=True`と`pretrained`は併用できず、例外が発生します。また、チェックポイントの最良指標キーが現在の実行と異なる場合は、意味の異なる値を比較せず、警告とともに最良指標の追跡をゼロへリセットします。

## ファイル内のレシピ

`cfg=`は`TrainConfig`フィールド名を持つYAMLマッピングを読み込み、明示的なキーワード引数の下に統合します。このため、キーワード引数が常にファイルより優先されます。

<code-tabs name="cfg" />

モデルインスタンスがすでに所有しているため、`size`と`num_classes`はファイルから除去されます。CLIに`--cfg`フラグはなく、ファイルパスはPython引数です。

## 関連項目

- `data=`が受け付ける内容については[データセット](/docs/train/datasets)を参照してください。
- データ拡張設定と対応するファミリーについては[データ拡張](/docs/train/augmentations)を参照してください。
- 重みの一部を学習する方法については[層の凍結](/docs/train/layer-freezing)と[LoRA](/docs/train/lora)を参照してください。
- 実行が報告する内容については[検証と指標](/docs/train/validation)を参照してください。
