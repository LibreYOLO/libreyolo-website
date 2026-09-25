---
title: "Facebook DETR alternative 2026：LibreYOLOで実行"
description: "元のFacebook Research DETRリポジトリはアーカイブされています。LibreYOLOで公式のDETRバリアント4種類を実行できます。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Facebook DETRは開発終了していますか？"
    a: "元のfacebookresearch/detrリポジトリは2024年3月12日にアーカイブされました。"
  - q: "LibreYOLOで元のDETRを学習できますか？"
    a: "いいえ。LibreYOLOはDETRの推論、検証、エクスポートに対応していますが、500エポックの学習レシピには対応していません。"
---

元の[Facebook Research DETRリポジトリ](https://github.com/facebookresearch/detr)は2024年3月12日にアーカイブされました。LibreYOLOでは、ほかのモデルファミリーと同じAPIを使って、元の検出器を引き続き利用できます。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

ResNet-50とResNet-101のバリアントが4種類あり、推論、検証、ONNXおよびTorchScriptへのエクスポートに利用できます。学習は実装されていません。また、LibreYOLOの入力サイズは800×800に固定されており、元の縦横比を維持する評価パイプラインは再現しません。

`pip install libreyolo`の実行後に試せます。[DETRのドキュメント](https://www.libreyolo.com/docs/models/detr)に4つのチェックポイントがすべて掲載されています。CNNとの比較には、[EfficientDetの代替案](/articles/efficientdet-alternative)をご覧ください。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [ドキュメント](https://www.libreyolo.com/docs)
