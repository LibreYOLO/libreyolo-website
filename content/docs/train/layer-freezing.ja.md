---
title: 層の凍結
seo_title: LibreYOLOで学習中に層を凍結
description: >-
  転移学習のためにモデルの一部を凍結します。ファミリーの凍結グループ数を表す整数、明示的なインデックスリスト、モジュール名・パラメータ名のselectorを使えます。
lead: >-
  凍結では、モデルのほかの部分を学習しながら、選択した重みを固定します。selectorが参照するのは、YAMLグラフの未加工の層番号ではなく、ファミリー独自の順序付き凍結グループまたはモジュール名です。
keywords:
  - layer freeze
  - 転移学習
  - backbone 凍結
  - frozen batchnorm
  - freeze groups
  - head のみ ファインチューニング
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 最初の10グループがYOLOv9のバックボーン全体
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: 名前で指定
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: 複数のselector
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: ファミリーの凍結グループを順番に一覧表示
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## 一部を凍結する

`freeze` は任意で、デフォルトでは何も凍結しません。

<code-tabs name="train" />

凍結は、モデルの構築と新しいクラス数に応じたヘッドの再構築が完了してから、optimizerを作成する前に実行されるため、optimizerは学習可能なパラメータだけを受け取ります。

## selectorに指定できるもの

| 値 | 意味 |
|---|---|
| `None`, `False`, `""`, `"none"` | すべてのパラメータを学習する |
| `10` または `"10"` | ファミリーの最初の10個の凍結グループを凍結する |
| `[0, 3, 7]` | そのゼロ始まりのグループを凍結する |
| `"backbone"` | 一致するグループ、モジュール、パラメータのprefixを凍結する |
| `["backbone", "neck"]` | 列挙した各selectorを凍結する |
| `["backbone", 3]` | 型が混在するリストも機能する |

文字列は解釈前にparseされるため、CLIとYAML設定はPythonと同じ形を受け付けます。`freeze="[0, 3, 'head']"` はliteral listとしてparseされ、`freeze="backbone,neck"` はcommaで分割され、単独の10進数文字列は個数になります。

`freeze=True` は曖昧なため拒否されます。

名前selectorは凍結グループ名、モジュール名、パラメータ名のprefixに一致し、glob文字 `*`、`?`、`[` も使えます。先頭の `model.` は柔軟に扱われるため、`backbone` と `model.backbone` はどちらもファミリー内部で使われる方の綴りに一致します。

## グループはファミリーが定義

整数が参照するのは、共有グラフ内の位置ではなく、ファミリー独自の順序付き凍結グループリストです。LibreYOLOのすべてのファミリーが1つのYAMLインデックス付きsequential modelであるわけではないため、未加工の層番号はファミリーごとに異なる意味になります。

YOLOv9のグループは入力側から順に並びます。10個のバックボーン段階、6個のネック段階、ヘッドです。そのため、`freeze=10` は正確にバックボーン全体となります。その上で `backbone`、`neck`、`head` を安定した名前selectorとして使えます。

RF-DETRのグループは `backbone.encoder`、`backbone.projector`、`decoder`、`queries`、`transformer.encoder_output`、`head` です。Transformerの構成要素は層数に対応しないため、ここでは名前がより適切です。`backbone` はprefixによって両方のバックボーングループに一致します。

意味的なグループを定義しないファミリーは、保守的なデフォルトへフォールバックします。宣言順に、1つ以上のパラメータを持つモデルの各direct childを使います。これは通常短いリストなので、大きな整数を指定すると十分なグループがありません。

```text
freeze index 10 is out of range for 3 available freeze groups.
```

推測せずに実際のリストを確認するには、次を使います。

<code-tabs name="groups" />

## 失敗は明示される

どのような指定ミスでも、要求していない内容を学習せずに例外を送出します。

何にも一致しないselectorを指定すると、一致しなかったselectorを示して例外が送出されます。

```text
freeze selector(s) matched no parameters: 'backbon'
```

学習可能なものが何も残らない凍結では、凍結時とoptimizer構築時の両方で例外が送出されます。

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

`all` はすべてのパラメータに一致するため、`freeze="all"` もこの状態になります。

凍結に成功すると、行われた処理が1行で記録されます。

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## 凍結したBatchNormは更新を停止

凍結したパラメータは、実行統計が更新され続けるモジュール内に存在します。パラメータが凍結セットに含まれるすべてのBatchNorm形式モジュールはevalモードへ切り替えられ、trainerは各エポックの `model.train()` 呼び出し後にも再適用するため、実行中ずっと統計が固定されます。

これはデフォルトで有効であり、バックボーンの凍結を実際の凍結として機能させるものです。

## LoRAとの組み合わせ

`freeze` と `lora=True` は併用できます。RF-DETR、DEIM、ConvNeXtでは、親グループが凍結されていてもadapterパラメータは学習可能なまま維持されます。これは、凍結したバックボーンの上でadapterを学習するという望ましい組み合わせです。[LoRAファインチューニング](/docs/train/lora)を参照してください。

## 範囲

これは起動時に決定する静的な凍結です。scheduled unfreezingとprogressive freezingはインターフェースに含まれません。

## 関連項目

- `train()` の残りの項目については[ハイパーパラメータ](/docs/train/hyperparameters)。
- 大規模モデルの知識を学習実行へ移すもう1つの方法については[蒸留](/docs/train/distillation)。

