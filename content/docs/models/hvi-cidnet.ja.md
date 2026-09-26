---
title: HVI-CIDNet
families:
  - hvi_cidnet
seo_title: HVI-CIDNet：LibreYOLOでの推論と学習
description: HVI-CIDNetは、色相、彩度、強度の処理によって低照度画像を復元します。
lead: HVI-CIDNetは、色相、彩度、強度の処理によって低照度画像を復元します。
keywords:
  - HVI-CIDNet
  - LibreYOLO
  - 画像復元 python
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHVICIDNett-restore.pt", device="cpu")
        result = model(SAMPLE_IMAGE, gamma=1.0, saturation=1.0, intensity=1.0)
        result.restored.save("enhanced.png")
source_hash: 443c3e4bbaf05712
---

## インストール

```bash
pip install "libreyolo"
```

## 推論

<code-tabs name="predict" />

`gamma`、`saturation`、`intensity`のデフォルトはいずれも1.0です。検証にはペア画像データを使います。学習とエクスポートには対応していません。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>
