---
title: PP-YOLOE
families:
  - ppyoloe
seo_title: PP-YOLOEをLibreYOLOで使う
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
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPYOLOEs.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.boxes)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPYOLOEs.pt", device="cpu")
        # coco8は自動でダウンロードされる小さなYOLO形式の物体検出データセットです。
        model.train(data="coco8.yaml", epochs=1, device="cpu", workers=0)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPYOLOEs.pt", device="cpu")
        metrics = model.val(data="coco8.yaml", workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPYOLOEs.pt", device="cpu")
        model.export(format="onnx")
source_hash: 1dcf261f906a55e1
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


## 検証

<code-tabs name="val" />

タスクに対応する形式のデータセットを使います。[検証](/docs/train/validation)でデータセットの要件と返される指標を説明しています。

## エクスポート

<export-matrix />

<code-tabs name="export" />

形式ごとの依存関係とエクスポートしたファイルの読み込みについては[エクスポートの設定](/docs/export)を参照してください。

## ライセンス

<provenance-box></provenance-box>
