---
title: PP-LiteSeg
families:
  - ppliteseg
seo_title: PP-LiteSegをLibreYOLOで使う
description: PP-LiteSegは、STDCバックボーンと融合デコーダーを使って各ピクセルに意味クラスを割り当てます。
lead: PP-LiteSegは、STDCバックボーンと融合デコーダーを使って各ピクセルに意味クラスを割り当てます。
keywords:
  - PP-LiteSeg
  - LibreYOLO
  - セマンティックセグメンテーション
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPLiteSegt50-sem.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.semantic_mask)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibrePPLiteSegt50-sem.pt", device="cpu")

        # データセットYAMLまたは分類フォルダーのパスを入力

        model.train(data=input("Dataset path: "), epochs=1, device="cpu",
        workers=0)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPLiteSegt50-sem.pt", device="cpu")
        metrics = model.val(data=input("Validation dataset path: "), workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPLiteSegt50-sem.pt", device="cpu")
        model.export(format="onnx")
source_hash: 5bd90ff464ca783f
---

## インストール

```bash
pip install "libreyolo"
```

## 推論

<code-tabs name="predict" />

t50とb50は`(512, 1024)`で、t75とb75は`(768, 1536)`で評価します。寸法の順序は`(height, width)`です。

## 学習

学習のデフォルトは800エポック、バッチサイズ8、`amp=False`です。デフォルトのクロップはt50/b50が`(512, 1024)`、t75/b75が`(768, 768)`です。検証では元の長方形を維持します。学習レシピでは、補助ヘッドを使い、交差エントロピー損失、Dice損失、エッジ損失を組み合わせます。

必要な学習データについては[データセットの設定](/docs/train/datasets)を参照してください。

<code-tabs name="train" />


## 検証

<code-tabs name="val" />

タスクに対応する形式のデータセットを使います。[検証](/docs/train/validation)でデータセットの要件と返される指標を説明しています。

## エクスポート

<export-matrix />

<code-tabs name="export" />

形式ごとの依存関係とエクスポートしたファイルの読み込みについては[エクスポートの設定](/docs/export)を参照してください。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>
