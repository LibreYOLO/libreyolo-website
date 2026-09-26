---
title: QuickSRNet
families:
  - quicksrnet
seo_title: QuickSRNet：LibreYOLOでの推論と学習
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
        result.save("upscaled.png")
source_hash: 5451bb234ac635a4
---

## インストール

```bash
pip install "libreyolo"
```

## 推論

<code-tabs name="predict" />

Medium 2xモデルは入力画像の元の解像度を維持し、縦横それぞれの寸法を2倍にします。検証では低解像度画像と高解像度画像のペアを使い、PSNRとSSIMを報告します。ONNXは動的な空間サイズに対応し、TorchScriptは固定サイズのキャンバスを使います。学習には対応していません。

## エクスポート

<export-matrix />

形式ごとの依存関係とエクスポートしたファイルの読み込みについては[エクスポートの設定](/docs/export)を参照してください。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>
