---
title: ハイパーパラメータ
seo_title: "LibreYOLOの学習ハイパーパラメータ"
description: "重要なtrain()引数について説明します：epochs、batch、lr0、optimizer、EMA、自動バッチ、勾配累積、再開、およびファミリーごとにデフォルトが異なる理由。"
lead: "すべての学習引数はTrainConfig dataclassのフィールドです。基底クラスがフィールドとデフォルト値を定義し、各モデルファミリーがそれをサブクラス化して、公開レシピで異なるデフォルト値をオーバーライドします。"
keywords:
  - train 引数
  - 学習率
  - バッチサイズ
  - 自動バッチ
  - 指数移動平均
  - 勾配累積
  - 学習 再開
  - 早期終了 patience
  - amp bfloat16
  - train config yaml
last_verified: "1.5.0"
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
    - label: ファミリーの解決済みデフォルト値を確認
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
        # ファミリーによる上書きを含め、train、val、predictのデフォルト値を表示
        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # batch=-1はGPUメモリを調査し、具体的な2の累乗に解決
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

        # optimizerステップごとに16のマイクロバッチを4回、実効バッチは64
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 中断した実行のチェックポイントを読み込み、再開を要求
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

        # yamlのキーはTrainConfigフィールド名。明示的なkwargsが優先
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
---

## 引数の設定

`train()`はキーワード引数を受け取り、CLIは同じ名前を`key=value`形式で受け取ります。

<code-tabs name="train" />

どちらの経路も同じ場所に到達します。kwargsは`TrainConfig.from_kwargs()`へ渡され、
ファミリーの設定dataclassが構築されます。

## 入力ミスは例外になりません

`from_kwargs()`は設定のフィールドではないキーをすべて破棄し、その名前を示す`UserWarning`を
出します。その後、デフォルト値を使って学習が開始されます。

```python
# UserWarning: 不明な学習設定キー、無視: ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

何も失敗せず、実行は完了しますが、学習率は呼び出し元が要求した値にはなりません。新しい
レシピの最初のエポックでは警告を確認してください。CLIは設定構築前にフラグ名を検証するため、
より厳密です。入力ミスのあるCLIフラグは直ちに拒否されます。

## デフォルト値はファミリーごとに異なります

`TrainConfig`はフィールドと基本デフォルトを定義します。各ファミリーがそれをサブクラス化し、
公開レシピで異なる値をオーバーライドするため、「デフォルトの学習率は何か」に単一の正解は
ありません。

基本デフォルトは`optimizer="sgd"`、`lr0=0.01`、`momentum=0.937`、
`weight_decay=5e-4`、`scheduler="yoloxwarmcos"`、`epochs=300`、`batch=16`、
`imgsz=640`、`amp=True`です。ファミリーがそこからどれほど変わるかを3例で示します。

| フィールド | 基本 | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINEとDEIMは`amp=False`で提供されます。D-FINEデコーダーが、float16で最大の有限値である
65504にアクティベーションをクランプするためです。YOLO-NASとFOMOもデフォルトで無効です。
CLIの`--amp`フラグはすべてのファミリーでデフォルトが`True`なので、ユーザー指定として数えられ、
ファミリーのデフォルト値を上書きします。変更する意図がない限り、そのままにしてください。

推測せず、ファミリーの実際のデフォルト値を確認するには次のようにします。

<code-tabs name="defaults" />

## バッチサイズ

`batch`はグローバルバッチです。マルチGPU学習では、各rankが`batch // world_size`を読み込む
ため、渡す数値はGPU数にかかわらずoptimizerステップごとの画像数です。
[マルチGPU学習](/docs/train/multi-gpu)を参照してください。

`batch=-1`は自動バッチを有効にします。学習器は学習モードのモデルに対して、2の累乗ごとに実際の
逆伝播を使って調査し、メモリ曲線へ直線を当てはめます。その後、総VRAMの60パーセント以内に
収まると推定された値を厳密に下回る、最大の2の累乗を選びます。

<code-tabs name="autobatch" />

重要なのは逆伝播を伴う学習モードで調査することです。推論モードの調査では、保持される
アクティベーションと勾配テンソルが漏れます。深いCNNの場合、これらは推論時の占有量の数倍です。
RF-DETRは目標比率を45パーセントに下げます。調査用の合成逆伝播でも、そのcriterionと補助
デコーダー層のコストを過小評価するためです。

自動バッチはCUDAの機能です。CPUまたはMPSでは1行をログに記録し、デフォルトバッチを維持します。

## 勾配累積

`nbs`は公称、つまり実効バッチサイズを設定します。学習器はoptimizerステップごとに
`round(nbs / batch)`個のマイクロバッチを累積します。

<code-tabs name="accumulate" />

デフォルトの`None`のままなら累積は無効で、学習は変わりません。

## 学習率とスケジュール

`lr0`は初期学習率で、`optimizer`は`sgd`、`adam`、`adamw`を受け取ります。`momentum`は
SGDのmomentumまたはAdamのbeta1、`weight_decay`はL2項で、`nesterov`はSGDに適用されます。

スケジュールの形状は`scheduler`、`warmup_epochs`、`warmup_lr_start`、`min_lr_ratio`で
決まります。`no_aug_epochs`は強いデータ拡張を使わずに実行する最後のエポック数を設定します。
一部のスケジュールでは終端の形状にも使うため、純粋なデータ拡張設定ではありません。各ファミリーが
データ拡張の側でどのように扱うかは[データ拡張](/docs/train/augmentations)にあります。

一部のファミリーは独自の学習率設定を追加します。`backbone_lr_mult`はヘッドに対するバックボーン
グループの倍率を設定し、`clip_max_norm`は勾配クリッピングを設定します。SegFormerは
`head_lr_mult`を使い、デコードヘッドをバックボーンの10倍の学習率で実行します。これらは基底
クラスではなく、ファミリーの設定サブクラスにあります。

## EMA

`ema=True`は学習済みの重みと並行して、重みの指数移動平均を維持します。FOMO以外ではデフォルトで
有効です。

`ema_decay`は目標の減衰率です。減衰率は目標値から始まらず、徐々に増加します。更新`n`での実効値は
`ema_decay * (1 - exp(-n / tau))`で、`tau`のデフォルトは2000です。そのため、初期の更新は
モデルをより密に追跡し、後期の更新では平滑化します。ファミリーのデフォルト値は、YOLO-NAS姿勢の
`0.997`からYOLOXの`0.9998`、YOLOv9とDETR系の`0.9999`まで異なります。

検証され、`best.pt`と`last.pt`に含まれるのはEMAの重みです。学習された生の重みも
`train_model`キーに保存されるため、再開時は平均からではなく学習軌跡から継続します。

## 精度

`amp=True`はCUDA autocastの下で順伝播を実行します。`amp_dtype`は`float16`（デフォルト）または
`bfloat16`を選択します。`fp16`と`bf16`という綴りも受け入れられます。

Float16には動的損失スケーリングが必要で、有効な`GradScaler`を使います。Bfloat16は指数範囲が
広いため不要で、scalerは構築されますが無効です。これによりoptimizer経路が同一に保たれます。
bfloat16に対応しないCUDAデバイスで要求すると、通知なく機能を落とさず、設定時に例外を
発生させます。

## 出力、チェックポイント、停止

実行は`project/name`へ書き出されます。`project`のデフォルトはどこでも`runs/train`ですが、
`name`はファミリーごとに上書きされる項目の1つです。基本デフォルトは`exp`で、YOLOv9は
`yolo9_exp`、D-FINEは`dfine_exp`を使います。デフォルトの`exist_ok=False`では、既存の
ディレクトリが上書きされず、連番の接尾辞が付けられます。

`save_period`はNエポックごとに追加の`weights/epoch_<N>.pt`を書き出します。さらに、各エポック後の
`weights/last.pt`と、追跡対象の指標が改善するたびの`weights/best.pt`があります。
`eval_interval`は検証の実行間隔を設定し、`patience`は改善なしでそのエポック数が続くと実行を
停止します。`0`は早期終了を無効にします。

`cache`は、デコード済み画像をRAM（`True`または`"ram"`）か、ソースの隣の`.npy`ファイル
（`"disk"`）に保持して、エポックの繰り返しを高速化します。キャッシュからの読み取りは新規の
読み取りとバイト単位で同一です。データローダーのワーカーを使う場合は`"disk"`の方が安全です。

## 再開

`resume=True`は中断した実行を継続します。再開処理は別の引数ではなくモデルからチェックポイントを
読み取るため、先にチェックポイントを読み込む必要があります。

<code-tabs name="resume" />

再開時には、学習済みの重み、optimizerの状態、EMAの重みと更新回数、最良指標の追跡、
`GradScaler`のスケール、PyTorch、CUDA、NumPyの乱数状態が復元されます。チェックポイントの
エポックに1を加えた位置から開始し、スケジュールをその位置まで早送りします。

2つの処理は行いません。`resume=True`は`pretrained`と併用できず、例外を発生させます。また、
チェックポイントの最良指標キーが現在の実行と異なる場合、意味の違う値を比較せず、警告とともに
最良指標の追跡を0へリセットします。

## ファイル内のレシピ

`cfg=`は`TrainConfig`フィールド名のYAMLマッピングを読み込み、明示的なキーワード引数の下に
マージします。そのため、常にkwargがファイルより優先されます。

<code-tabs name="cfg" />

モデルインスタンスがすでに所有するため、`size`と`num_classes`はファイルから除去されます。
CLIには`--cfg`フラグがなく、ファイルパスはPython引数です。

## 関連項目

- `data=`が受け取るものについては[データセット](/docs/train/datasets)を参照してください。
- データ拡張の設定値と、それを尊重するファミリーについては[データ拡張](/docs/train/augmentations)を参照してください。
- 重みの一部を学習する方法については[層の凍結](/docs/train/layer-freezing)と[LoRA](/docs/train/lora)を参照してください。
- 実行が報告する内容については[検証と指標](/docs/train/validation)を参照してください。
