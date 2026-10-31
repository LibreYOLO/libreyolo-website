---
title: 知識蒸留
seo_title: LibreYOLOの知識蒸留
description: >-
  大規模な教師モデルまたは凍結済みDINOv2バックボーンを使って小型検出器を学習します。MGD、CWD、特徴量MSEの損失、取得点、対応ファミリーを解説します。
lead: >-
  蒸留は、生徒モデルの中間特徴マップを凍結済み教師モデルの特徴マップに近づける第2の損失項を追加します。LibreYOLOは順伝播フックで特徴量を取得するため、教師自身のヘッドと損失は関与しません。
keywords:
  - 知識蒸留
  - Masked Generative Distillation
  - Channel-wise Distillation
  - 特徴蒸留
  - DINOv2 教師モデル
  - 教師生徒 学習
  - MGD loss
  - CWD loss
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 同じファミリーの大きなチェックポイントで小さなモデルを指導する
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 凍結した自己教師ありViTでバックボーンの1ステージを指導する
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: 損失を調整
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # 蒸留全体の重み
            distill_tau=1.0,   # CWDのsoftmax温度
        )
source_hash: 7210031328f6826f
---

## 大規模チェックポイントから蒸留

`distill_model`を設定すると蒸留が有効になります。値には教師のチェックポイントを指定し、他のモデルと同じファクトリで読み込みます。

<code-tabs name="detector" />

教師は`no_grad`内で順伝播し、AMPが有効な場合はautocast内でも実行されるため、凍結済みモデルが各ステップで完全精度の計算コストを負うことはありません。順伝播フックは名前付きの取得点で特徴マップを取得し、損失が生徒の特徴量と比較します。その結果は学習損失へ加算され、`distill`という名前の構成要素として報告されます。

## 凍結済み基盤バックボーンから蒸留

代わりに、自己教師ありViTで生徒の1つのバックボーンステージを指導できます。教師の特徴量はフックではなく教師自身の特徴抽出器から取得し、損失がパッチグリッドと畳み込みストライドの不一致を処理します。

<code-tabs name="foundation" />

`distill_model`はDINOv2-baseを表す`dinov2`に加え、`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`dinov2-small`、`dinov2-base`、`dinov2-large`、および`facebook/dinov2`で始まる任意の生のHub IDを認識します。それ以外は教師チェックポイントのパスとして扱われます。

この経路では`distill_loss_type`にかかわらず`feat_mse`を使用し、`transformers`のインストールが必要です。重みキーが不足した状態で読み込まれた教師は、部分的にランダムなバックボーンで蒸留せず処理を中止します。

## 対応ファミリー

蒸留への対応は生徒モデルのメソッドとして実装され、2種類あります。

`get_distill_config()`は検出器の教師が指導するマルチスケールの取得点を提供します。YOLOv9、YOLOX、RF-DETRが実装しています。

`get_backbone_distill_config()`は基盤モデルの教師が指導する1つのバックボーンステージを提供します。実装しているのはYOLOv9だけです。

それ以外では、損失なしで学習せず例外を送出します。

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## 特徴量の取得点

取得点はファミリーと役割ごとに固定されます。教師と生徒が同じアーキテクチャである必要はありませんが、特徴量のストライドは一致する必要があります。

| ファミリー | 役割 | 取得点 | ストライド |
|---|---|---|---|
| YOLOv9 | 教師または生徒 | `neck.elan_up2`、`neck.elan_down1`、`neck.elan_down2` | 8、16、32 |
| YOLOv9 | 基盤モデルの生徒 | `backbone.elan3` | 16 |
| YOLOX | 教師または生徒 | `backbone.C3_p3`、`backbone.C3_n3`、`backbone.C3_n4` | 8、16、32 |
| RF-DETR | 教師または生徒 | `model.backbone.0.projector.stages.0` | セットアップ時に調査 |

ストライドが一致しない場合は、学習開始前に例外を送出します。

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

基盤モデルの教師では、この確認を行いません。グリッドが異なること自体がこの経路の目的だからです。

## 3種類の損失

検出器の教師では、`distill_loss_type`で特徴量の損失を選択します。基盤モデルの教師は常に`feat_mse`を使用します。

`mgd`（Masked Generative Distillation）は、生徒の空間位置の一部をマスクし、小さな2層の畳み込みジェネレーターを学習して、残りの情報から教師の完全な特徴マップを再構築します。`distill_mask_ratio`はマスクする割合で、デフォルトは0.65です。

`cwd`（Channel-wise Distillation）は、各チャンネルの空間活性化を確率分布へ変換し、チャンネルごとにKLダイバージェンスを最小化します。`distill_tau`はsoftmax温度で、デフォルトは1.0です。

`feat_mse`は1×1畳み込みで生徒のチャンネルを教師に合わせ、教師のグリッドを生徒のグリッドへ双線形補間して、平均二乗誤差を取ります。`distill_normalize=True`は、最初に両方の特徴マップをチャンネル次元に沿ってL2正規化します。これにより比較は角度だけに基づき、スケールに依存しなくなります。デフォルトは`False`です。

`dis`は全体に適用する重みです。未指定の場合、各損失が公開済みのデフォルト値を使用します。MGDは2e-5、CWDは1.0、特徴量MSEは1.0です。5桁も異なるため、ある損失用に調整した重みは別の損失では意味を持ちません。

<code-tabs name="tuned" />

`distill_mask_ratio`、`distill_tau`、`distill_normalize`にはCLIフラグがありません。Python引数または`cfg=`のYAMLキーとして指定します。RF-DETRのCLI引数マッピングには蒸留キーがないため、RF-DETRの蒸留全体もPython専用です。

## アダプター、チェックポイント、マルチGPU

各損失は生徒の外側に小さな学習可能モジュールを構築します。1×1チャンネルアダプターとMGDのジェネレーターです。これらには実行時の有効学習率を持つ専用のオプティマイザーパラメーターグループが割り当てられます。

これらのモジュールはチェックポイントの`distiller`キーに書き込まれ、再開時に復元されます。そのため、再開した実行でプロジェクターが初期状態に戻ることはありません。

DDPではアダプターがラップ済みの生徒モデル外部にあるため、DDPリデューサーはその勾配を認識しません。トレーナーは各ステップで明示的にall-reduceを行い、すべてのランクで同じアダプターを学習します。

蒸留実行ではCUDAグラフのキャプチャを利用できません。`cuda_graph=True`を渡すとログを1行出し、通常実行で学習します。[学習パフォーマンス](/docs/train/performance)を参照してください。

## 関連項目

- [レイヤーの凍結](/docs/train/layer-freezing)と[LoRAファインチューニング](/docs/train/lora)は、どちらも蒸留と組み合わせることができます。
- その他の`train()`設定については[ハイパーパラメーター](/docs/train/hyperparameters)を参照してください。
