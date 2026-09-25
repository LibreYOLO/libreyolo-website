---
title: "YOLO-NASはまだ使える？LibreYOLOで学習・推論・エクスポート"
description: "LibreYOLOでYOLO-NASの推論、学習、検証、エクスポートを実行できます。SuperGradientsの最終リリースは2024年4月です。Deciのライセンスを解消するため、新しい重みをスクラッチから学習する予定です。"
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "YOLO-NASは開発終了していますか？"
    a: "SuperGradientsの最終リリースは2024年4月の3.7.1です。LibreYOLOは最新のPyTorchでYOLO-NASの推論、学習、検証、エクスポートをサポートしています。"
  - q: "UltralyticsはYOLO-NASをメンテナンスしていますか？"
    a: "Ultralyticsは推論、検証、エクスポートをサポートしています。学習はサポートしていません。"
  - q: "学習済みYOLO-NASの重みを商用利用できますか？"
    a: "Deciの学習済み重みには、どのライブラリで読み込んでもアップストリームの非商用条件が適用されます。ランダム初期化から学習すれば、これらのチェックポイントを使わずに済みます。スクラッチから学習したCOCOの重みも公開する予定です。"
---

YOLO-NASは今も有用な検出モデルです。元の提供元である[SuperGradients](https://github.com/Deci-AI/super-gradients)の最終リリースは、2024年4月の3.7.1でした。LibreYOLOでは、YOLO-NASの推論、学習、検証、エクスポートをメンテナンスしています。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLOは検出タスクでS、M、Lをサポートし、姿勢推定にも対応しています。チェックポイントは`LibreYOLONASs.pt`のような名前でDeciの公開CDNからダウンロードされます。LibreYOLOはチェックポイントをホストしておらず、Deciの非商用条件は引き続き適用されます。SuperGradientsが対応するPyTorchのバージョンは1.9から1.13までですが、LibreYOLOには2.4以降が必要です。検出モデルのエクスポート形式はONNX、TorchScript、OpenVINO、NCNN、TFLite、Paddle、MNN、ExecuTorch、Core AIに対応しています。CoreMLには対応していません。

Deciのチェックポイントを使わずに学習するには、次のようにします。

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

YOLO-NASをCOCOでスクラッチから学習し、その重みを公開する予定もあります。

`pip install libreyolo`でインストールできます。[YOLO-NASのドキュメント](https://www.libreyolo.com/docs/models/yolo-nas)にモデルの詳細があります。[SuperGradientsの代替](/articles/supergradients-alternative)に関する、より詳しい記事もあります。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [ドキュメント](https://www.libreyolo.com/docs)
