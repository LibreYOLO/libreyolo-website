---
title: PP-YOLOE
families:
  - ppyoloe
seo_title: PP-YOLOE：LibreYOLOでの推論と学習
description: PP-YOLOEは、物体検出の学習とクラスヘッドのサイズ変更に対応するアンカーフリーの物体検出モデルです。
lead: PP-YOLOEは、物体検出の学習とクラスヘッドのサイズ変更に対応するアンカーフリーの物体検出モデルです。
keywords:
  - PP-YOLOE
  - LibreYOLO
  - 物体検出 python
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibrePPYOLOE, SAMPLE_IMAGE


        model = LibrePPYOLOE(model_path=input("Path to upstream weights: "),
        device="cpu")

        result = model(SAMPLE_IMAGE)

        print(result.boxes)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibrePPYOLOE


        model = LibrePPYOLOE(model_path=input("Path to upstream weights: "),
        device="cpu")

        # データセットYAMLまたは分類フォルダーのパスを入力

        model.train(data=input("Dataset path: "), epochs=1, device="cpu",
        workers=0)
source_hash: fb96c61dcba9c0bc
---

## インストール

```bash
pip install "libreyolo"
```

## 推論

<code-tabs name="predict" />

公開済みの重みは、固定されたSHA-256による検証を行って配布元のCDNからダウンロードします。コンストラクターはローカルのアップストリームチェックポイントも受け付けます。サイズはs、m、l、xの4種類です。

## 学習

学習ではATSSによる割り当てを行った後、タスク整合型の割り当てに切り替えます。`static_assigner_epochs`のデフォルトは、指定したエポック数の30％です。クラス数を変更すると、クラス予測層を再構築します。AMPのデフォルトは`False`です。

必要な学習データについては[データセットの設定](/docs/train/datasets)を参照してください。

<code-tabs name="train" />

## エクスポート

<export-matrix />

形式ごとの依存関係とエクスポートしたファイルの読み込みについては[エクスポートの設定](/docs/export)を参照してください。

## ライセンス

<provenance-box></provenance-box>
