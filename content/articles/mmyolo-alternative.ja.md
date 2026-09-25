---
title: "MMYOLOの代替を探すなら？LibreYOLOでRTMDetを実行"
description: "MMYOLOの最新リリースは2023年です。LibreYOLOの直接APIを使って、RTMDetなどのサポート対象検出器を実行できます。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "MMYOLOは開発終了していますか？"
    a: "MMYOLOは正式にアーカイブされていませんが、最新のGitHubリリースは2023年8月のバージョン0.6.0です。"
  - q: "LibreYOLOで任意のMMYOLOチェックポイントを読み込めますか？"
    a: "いいえ。LibreYOLOがサポートするのは名前が明示されたモデルファミリーであり、任意のMMYOLOチェックポイントをそのまま読み込めるとは限りません。"
---

[MMYOLO](https://github.com/open-mmlab/mmyolo)はアーカイブされていませんが、最新リリースのバージョン0.6.0は2023年8月のものです。RTMDetなどのサポート対象検出器だけが必要なら、OpenMMLabのスタック全体は不要かもしれません。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLOはMMYOLOを完全に置き換えるものではありません。すべての設定、レシピ、チェックポイントを再現するわけではありません。一方、RTMDetやYOLOXなどのサポート対象ファミリーに、推論、学習、検証、エクスポート用の直接APIを提供します。

まず`pip install libreyolo`を実行してください。[RTMDetのドキュメント](https://www.libreyolo.com/docs/models/rtmdet)にサポート対象の操作が記載されています。[RTMDetの簡潔なガイド](/articles/rtmdet-without-mmdetection)では移行について説明しています。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [ドキュメント](https://www.libreyolo.com/docs)
