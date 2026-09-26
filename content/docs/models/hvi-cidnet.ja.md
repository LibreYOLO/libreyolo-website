---
title: HVI-CIDNet
families:
  - hvi_cidnet
seo_title: HVI-CIDNetをLibreYOLOで使う
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
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHVICIDNett-restore.pt", device="cpu")
        # ペア画像復元データセットのYAML: inputs/に劣化画像、targets/にクリーン画像を置き、ファイル名の語幹で対応付けます。
        metrics = model.val(data="path/to/your/restore.yaml", workers=0)
        print(metrics)
source_hash: 12d3a7b809c89c08
---

## インストール

```bash
pip install "libreyolo"
```

## 推論

<code-tabs name="predict" />

`gamma`、`saturation`、`intensity`のデフォルトはいずれも1.0です。検証にはペア画像データを使います。学習とエクスポートには対応していません。


## 検証

<code-tabs name="val" />

タスクに対応する形式のデータセットを使います。[検証](/docs/train/validation)でデータセットの要件と返される指標を説明しています。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
