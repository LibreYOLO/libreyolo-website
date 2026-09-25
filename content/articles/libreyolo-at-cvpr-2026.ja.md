---
title: "CVPR 2026でLibreYOLOXsをエッジAIに活用、Hailo-8LとSnapdragonで動作"
description: "デンバーで開催されたCVPR 2026で、JabraとコペンハーゲンIT大学のチームが「Edge AI in Action」チュートリアルのモデル例としてLibreYOLOXsを使用し、Hailo-8LとSnapdragonで実行しました。"
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "エッジハードウェアでLibreYOLOXsはどのくらいの速度で動作しますか？"
    a: "CVPR 2026のチュートリアルでは、LibreYOLOXsはHailo-8Lを搭載したRaspberry Pi 5で1フレームあたり19.0 ms（52.6 FPS）、INT8量子化後のQualcomm QCS6490のHTP/DSPで6.69 msで動作しました。"
  - q: "LibreYOLOモデルをエッジアクセラレーターにデプロイするにはどうすればよいですか？"
    a: "LibreYOLOのエクスポート呼び出しでONNXにエクスポートし、ターゲット向けにコンパイルします。HailoチップではHailo Dataflow CompilerがHEFを生成し、Qualcomm SnapdragonではSNPE / QAIRT / AI Hubスタックを使用します。これがCVPRのチュートリアルで最初から最後まで実演されたパイプラインです。"
---

![CVPR 2026、コロラド州デンバー、6月3日〜7日](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLOがCVPR 2026に登場しました。

CVPRは、世界で最も権威あるコンピュータビジョン学会として広く知られています。今年はデンバーで開催され、JabraとコペンハーゲンIT大学のチームが「Edge AI in Action: Mastering On-Device Inference」というチュートリアルを実施しました。エッジチップで物体検出を行うモデル例として、LibreYOLOXsを選びました。

## 構築したもの

このチュートリアルでは、エッジ向けのパイプライン全体を最初から最後まで紹介しました。まずLibreYOLOXsモデルを用意し、ONNXにエクスポートします。次に、そのONNXを性質の大きく異なる2種類のハードウェア、Hailo-8LアクセラレーターとQualcomm Snapdragon向けにコンパイルします。

## Hailo-8Lでリアルタイム動作

ONNXをHailo Dataflow CompilerでHEFにコンパイルし、Hailo-8L AI HATを搭載したRaspberry Pi 5で実行しました。

![Hailo-8Lを搭載したRaspberry Pi 5でLibreYOLOXsが動作し、にぎやかな通りで人とハンドバッグを検出している様子](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Raspberry Piで1フレームあたり19.0 ms、52.6 FPSで動作しました。

## Snapdragonでリアルタイム動作

Qualcomm側では、LibreYOLOXsをINT8に量子化し、SNPE、QAIRT、AI Hubのスタックで実行しました。ベンチマークはQCS6490で測定しました。

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="Snapdragonスマートフォン上のEdgeVision AIアプリで、LibreYOLOXsがテレビ、ノートPC、キーボード、マウス、携帯電話をリアルタイム検出している様子" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

HTP/DSPで6.69 msでした。

## 謝辞

プロジェクトを取り上げてくださったSai Narsi Reddy Donthi Reddy、Fabricio Batista Narcizo、Elizabete Munzlingera、Shan Ahmed Shaffiに感謝します。

- チュートリアルとスライド：[Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Qualcomm向けに量子化されたLibreYOLOXsモデル：[Hugging Face上のモデル](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## 試してみる

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # エッジターゲット向けにコンパイル
```

LibreYOLOはMITライセンスで、Linux、Mac、Windowsで動作します。コードを変更せずにGPU、Apple Silicon、通常のCPUで使えます。1つのAPIでYOLOX、RF-DETR、D-FINE、DEIM、YOLO-NAS、セグメンテーション、姿勢推定、深度推定などに対応します。

GitHubでスターを付けてください：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | ドキュメント：[libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
