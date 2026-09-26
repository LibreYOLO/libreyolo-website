---
title: LaMa
families:
  - lama
seo_title: LaMaをLibreYOLOで使う
description: LaMaは、画像のマスクされた領域を補完します。
lead: LaMaは、画像のマスクされた領域を補完します。
keywords:
  - LaMa
  - LibreYOLO
  - 画像復元 python
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreLaMab-restore.pt", device="cpu")

        import numpy as np

        from PIL import Image


        image = Image.open(SAMPLE_IMAGE).convert("RGB")

        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        mask[image.height//3:2*image.height//3, image.width//3:2*image.width//3]
        = 255

        result = model(image, mask=mask)

        result.restored.save("inpainted.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLaMab-restore.pt", device="cpu")
        metrics = model.val(data=input("Validation dataset path: "), workers=0)
        print(metrics)
source_hash: f1bc3e47424667ad
---

## インストール

```bash
pip install "libreyolo[onnx]"
```

## 推論

<code-tabs name="predict" />

単一画像の推論では`mask=`を渡してください。ゼロ以外のピクセルが補完する領域を示します。チェックポイントにはONNXグラフが含まれ、ONNX Runtime 1.18以降が必要です。学習とエクスポートには対応していません。


## 検証

<code-tabs name="val" />

タスクに対応する形式のデータセットを使います。[検証](/docs/train/validation)でデータセットの要件と返される指標を説明しています。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
