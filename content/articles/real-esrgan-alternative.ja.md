---
title: "Real-ESRGANの代替：LibreYOLOで画像を2倍・4倍にアップスケール（2026年）"
description: "LibreYOLOでReal-ESRGANの2xおよび4x画像アップスケーリングを実行します。大きな画像向けのタイル分割推論にも対応しています。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Real-ESRGANは開発終了していますか？"
    a: "リポジトリはアーカイブされていませんが、最新のGitHubリリースは2022年9月のバージョン0.3.0です。"
  - q: "LibreYOLOでReal-ESRGANを学習できますか？"
    a: "いいえ。LibreYOLOはReal-ESRGANの推論、検証、エクスポートに対応していますが、学習には対応していません。"
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)は正式にはアーカイブされていませんが、最新のGitHubリリースであるバージョン0.3.0は2022年9月のものです。LibreYOLOでは、通常の推論APIを通じて2xおよび4xのチェックポイントを実行できます。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLOは、タイル分割推論、PSNR/SSIM検証、エクスポートに対応した2x、4x、高速4xのチェックポイントをサポートしています。学習は実装されていません。アップストリームのノイズ除去強度ブレンドも、この移植版では利用できません。

試すには`pip install libreyolo`を実行してください。復元に関するすべてのオプションは[Real-ESRGANのドキュメント](https://www.libreyolo.com/docs/models/real-esrgan)に記載されています。深度推定については[MiDaSの代替](/articles/midas-alternative)をご覧ください。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [ドキュメント](https://www.libreyolo.com/docs)
