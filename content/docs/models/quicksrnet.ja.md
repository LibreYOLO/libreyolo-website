---
title: QuickSRNet
families:
  - quicksrnet
seo_title: QuickSRNetをLibreYOLOで使う
description: QuickSRNetは、画像を2倍に拡大します。
lead: QuickSRNetは、画像を2倍に拡大します。
keywords:
  - QuickSRNet
  - LibreYOLO
  - 画像復元 python
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreQuickSRNetm2-restore.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        result.restored.save("upscaled.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreQuickSRNetm2-restore.pt", device="cpu")
        metrics = model.val(data=input("Validation dataset path: "), workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreQuickSRNetm2-restore.pt", device="cpu")
        model.export(format="onnx")
source_hash: 34a2e81450788353
---

## インストール

```bash
pip install "libreyolo"
```

## 推論

<code-tabs name="predict" />

Medium 2xモデルは入力画像の元の解像度を維持し、縦横それぞれの寸法を2倍にします。検証では低解像度画像と高解像度画像のペアを使い、PSNRとSSIMを報告します。ONNXは動的な空間サイズに対応し、TorchScriptは固定サイズのキャンバスを使います。学習には対応していません。


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
