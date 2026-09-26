---
title: ViTMatte
families:
  - vitmatte
seo_title: ViTMatteをLibreYOLOで使う
description: ViTMatteは、画像とトライマップから前景のアルファマットを予測します。
lead: ViTMatteは、画像とトライマップから前景のアルファマットを予測します。
keywords:
  - ViTMatte
  - LibreYOLO
  - 画像 マッティング
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTMattes-matte.pt", device="cpu")
        import numpy as np
        from PIL import Image

        image = Image.open(SAMPLE_IMAGE).convert("RGB")
        # 入力形式を示すため全ピクセルを未確定に設定 # マッティングには実際のトライマップを指定
        trimap = np.full((image.height, image.width), 128, dtype=np.uint8)
        result = model(image, trimap=trimap)
        result.save("cutout.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTMattes-matte.pt", device="cpu")
        metrics = model.val(data=input("Validation dataset path: "), workers=0)
        print(metrics)
source_hash: 75d9cbe93fc6f210
---

## インストール

```bash
pip install "libreyolo"
```

## 推論

<code-tabs name="predict" />

トライマップでは、背景を0、未確定のピクセルを128、前景を255で表します。単一画像では`trimap=`で渡してください。検証では`trimap_dir`を指定するか、`trimap_radius`でガイドを生成します。学習とエクスポートには対応していません。


## 検証

<code-tabs name="val" />

タスクに対応する形式のデータセットを使います。[検証](/docs/train/validation)でデータセットの要件と返される指標を説明しています。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
