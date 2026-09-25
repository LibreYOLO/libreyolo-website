---
title: "YOLOXの代替を2026年に探すなら：LibreYOLOで実行・学習"
description: "YOLOXの最新リリースは2022年です。LibreYOLO APIを使ってYOLOXの推論、学習、検証、エクスポートを実行できます。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "YOLOXは開発終了していますか？"
    a: "公式リポジトリはアーカイブされていませんが、最新リリースは2022年4月のYOLOX 0.3.0です。正式に開発終了とされているわけではなく、活動が低調という表現がより正確です。"
  - q: "LibreYOLOでYOLOXを学習できますか？"
    a: "はい。LibreYOLOはYOLOXの推論、学習、検証、エクスポートに対応しています。"
---

YOLOXは今も有用なApache-2.0の物体検出モデルです。[公式リポジトリ](https://github.com/Megvii-BaseDetection/YOLOX)はアーカイブされていませんが、最新リリースの0.3.0は2022年4月のものです。メンテナンス上の問題は実際にあります。それでもモデルは動作します。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLOはNanoからXまで6つのYOLOXサイズに対応し、推論、学習、検証、エクスポートを1つのAPIから実行できます。元のプロジェクト固有の`Exp`設定ワークフローが必要な場合は、元のプロジェクトを使ってください。

まず`pip install libreyolo`を実行してください。[YOLOXのドキュメント](https://www.libreyolo.com/docs/models/yolox)でAPI全体を確認できます。また、[YOLO-NASが現在もメンテナンスされている](/articles/yolo-nas-with-libreyolo)ことを短く紹介した記事もあります。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [ドキュメント](https://www.libreyolo.com/docs)
