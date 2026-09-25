---
title: "MiDaSの代替手段2026年版：LibreYOLOで深度推定"
description: "公式MiDaSリポジトリはアーカイブされています。SmallおよびDPT-Largeの深度モデルをLibreYOLOで実行できます。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "MiDaSは開発終了していますか？"
    a: "公式のisl-org/MiDaSリポジトリは2025年8月25日にアーカイブされました。"
  - q: "MiDaSはメートル単位の深度を返しますか？"
    a: "いいえ。MiDaSが返すのは相対逆深度で、メートル単位では表されず、画像間で固定されたスケールもありません。"
---

[公式MiDaSリポジトリ](https://github.com/isl-org/MiDaS)は2025年8月25日にアーカイブされました。LibreYOLOは、MiDaS SmallとDPT-Largeのチェックポイントを実行するための、継続的にメンテナンスされている選択肢を提供します。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

推論、ゼロショット検証、固定解像度でのエクスポートに対応していますが、学習には対応していません。出力は相対逆深度で、値が大きいほど近くを示しますが、メートル単位では表されず、画像間で固定されたスケールもありません。

基本インストールで十分です。`pip install libreyolo`を実行してください。[MiDaSのドキュメント](https://www.libreyolo.com/docs/models/midas)または[Real-ESRGANの代替手段](/articles/real-esrgan-alternative)をご覧ください。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [ドキュメント](https://www.libreyolo.com/docs)
