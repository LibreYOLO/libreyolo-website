---
title: Depth Anything V2の深度推定をLibreYOLOで簡単に実行
description: Depth Anything V2は、単眼深度推定で最先端の性能を発揮します。LibreYOLOなら2行で実行できます。
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "Depth Anything V2を最も簡単に実行する方法は何ですか？"
    a: "LibreYOLOなら2行で実行できます。LibreDepthAnythingV2l-depth.ptを名前で読み込み、save=Trueを指定してpredictを呼び出します。重みは初回使用時に自動ダウンロードされ、正規化、カラーマップ、デバイスへの配置も、CUDA、Apple Silicon、通常のCPU上で自動的に処理されます。"
  - q: "Depth Anything V2は商用利用できますか？"
    a: "Apache-2.0なのはSmallエンコーダーだけです。Base、Large、GiantはCC-BY-NC-4.0なので、最も高性能なチェックポイントは非商用利用に限られます。どのライブラリで重みを読み込んでも、アップストリームのライセンスが適用されます。"
  - q: "LibreYOLOはDepth Anything V2の学習に対応していますか？"
    a: "いいえ。学習は元のリポジトリで行います。LibreYOLOは深度推定タスクの推論、動画、検証に対応しています。"
---

![ビルバオのグッゲンハイム美術館とその深度マップ](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2は、現在利用できる単眼深度マップの中でも特に優れたものを生成します。

YOLOの開発経験がある場合、公式リポジトリはすぐに使い始めるには少し手間がかかります。深度マップを1枚生成するだけでも、いくつかの手作業が必要です。

* 適切なチェックポイントを手動でダウンロードし、正しいフォルダーに配置する

* エンコーダーと設定（`encoder="vitl"`、`features=256`、`out_channels=...`）を一致させる

* 正規化と色付けの処理を自分で記述する

* CUDAだけでなくMacやCPUでも動くようにデバイスへの配置を処理する

* スタックのほかの部分とはまったく異なる推論APIを覚える

どれも難しい作業ではありません。ただ手間がかかるだけで、すべて省けます。

LibreYOLOは同じDepth Anything V2の重みを読み込み、`predict()`を呼び出すだけで深度推定を実行できます。モデル名を指定すれば、初回使用時にダウンロードされます。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # 初回実行時に自動ダウンロード
model.predict("image.jpg", save=True)     # 色付き深度マップをディスクに保存
```

これだけです。標準的なYOLO APIを使ったことがあれば、見慣れた形です。モデル名で読み込み、`predict`を呼び出し、`save=True`で可視化結果を保存します。物体検出で使うものとまったく同じ呼び出し方ですが、結果にはボックスではなく深度マップが含まれます。カラーマップ、正規化、デバイスへの配置は自動で処理されます。CUDAを使うLinuxでも、Apple Silicon搭載のMacでも、通常のCPUでも同じように動作します。コードを変更する必要はありません。

さらに、同じ呼び出しで生の深度値を取得し、そのまま扱うこともできます。Depth Anything V2自体の学習は元のリポジトリで行います。LibreYOLOは推論、動画、検証に対応しています。

まったく同じ1行の呼び出しで、さまざまなシーンを処理できます。

![パルクールのサンプル画像と深度マップ。手前のジャンパーが背後のコンクリート壁より際立っています。](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![ドノスティアのラ・コンチャ湾を上空から見た画像と深度マップ。ボートと海岸線が近く、沖の海が遠くに見えます。](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![ゲルニカの議事堂の列柱がある中庭と深度マップ。奥へ続く建築の様子がわかります。](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![ドノスティアのコンスティトゥシオン広場で夜に開かれた祭りの群衆と深度マップ。手前の人々が、背後の照らされたファサードから際立っています。](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

重みについて補足します。LibreYOLOは変換済みのDepth Anything V2チェックポイントをホストしており、初回使用時に取得するため、手動でダウンロードする必要はありません。アップストリームのライセンスは引き続き適用されます。SmallエンコーダーはApache-2.0ですが、Base、Large、GiantはCC-BY-NC-4.0（非商用）なので、高性能なチェックポイントは非商用利用に限られます。オフラインで使いたい場合や、自分で変換したい場合は、1回限りの変換スクリプトも利用できます。

```bash
# 自動ダウンロードの代わりに公式チェックポイントを自分で変換する場合
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## 試してみる

```bash
pip install libreyolo
```

LibreYOLOは、pipでインストールできる最も包括的なコンピュータビジョンライブラリです。使い慣れた1つのAPIで、物体検出（RF-DETR、D-FINE、DEIM）、セグメンテーション、姿勢推定、回転バウンディングボックス、そして新たにDepth Anything V2による単眼深度推定まで、拡大を続ける最先端モデル群を利用できます。対応するタスクの学習と検証も行えます。主要なすべてのOSで、GPUまたはCPU上で動作し、完全なMITライセンスなので、制約なく商用製品に組み込めます。

GitHubでスターを付けてください：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | ドキュメント：[libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
