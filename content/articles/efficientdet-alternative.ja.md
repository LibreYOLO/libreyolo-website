---
title: "EfficientDetの代替モデル：LibreYOLOでD0〜D4を推論"
description: "LibreYOLOの物体検出APIを使って、EfficientDet D0〜D4の推論、検証、エクスポートを実行できます。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "EfficientDetは開発終了していますか？"
    a: "rwightmanのPyTorchリポジトリは正式にアーカイブされていません。開発終了というより、更新が少ない状態です。"
  - q: "LibreYOLOでEfficientDetを学習できますか？"
    a: "いいえ。LibreYOLOはEfficientDetの推論、検証、エクスポートに対応していますが、学習には対応していません。"
---

広く使われている[EfficientDet PyTorchリポジトリ](https://github.com/rwightman/efficientdet-pytorch)はアーカイブされていないため、開発終了と呼ぶのは正確ではありません。更新が少ない状態です。LibreYOLOは別のデプロイ方法を提供します。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLOはEfficientDet D0からD4までの推論と検証に対応しています。ONNX、TorchScript、OpenVINO、TensorRTへのエクスポートでは、パリティ検証に対応しています。学習は実装されておらず、各サイズの入力解像度は固有の値に固定されています。

`pip install libreyolo`でインストールできます。各サイズについては[EfficientDetのドキュメント](https://www.libreyolo.com/docs/models/efficientdet)を確認するか、[Facebook DETRの代替モデル](/articles/facebook-detr-alternative)と比較してください。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [ドキュメント](https://www.libreyolo.com/docs)
