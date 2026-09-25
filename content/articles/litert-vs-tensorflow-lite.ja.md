---
title: "LiteRT vs TensorFlow Lite：名称変更であり、新しい形式ではない"
description: "Googleは2024年9月にTensorFlow LiteをLiteRTへ改称しました。同じランタイムで、同じ`.tflite`ファイルが使えるため、既存の環境に影響はありません。改称の理由、実際に変わった点、LibreYOLOモデルをエクスポートする方法を解説します。"
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "TensorFlow Liteは非推奨ですか？"
    a: "廃止されるのは名前であり、技術ではありません。ランタイムは同じAPIを備えたLiteRTとして引き続き提供され、既存のTFLiteモデルとコードもそのまま動作します。CompiledModel APIやNPUアクセラレーションなどの新機能は、LiteRTの名前で提供されます。"
  - q: "LiteRT用にモデルを再エクスポートする必要がありますか？"
    a: "いいえ。TensorFlow Lite用にエクスポートした`.tflite`ファイルはLiteRTモデルです。ファイルに変更はないため、再エクスポートも移行も必要ありません。"
---

**LiteRTはTensorFlow Liteの新しい名前です。新しいランタイムでも、新しいファイル形式でもありません。モデルは引き続き`.tflite`ファイルであり、TensorFlow Liteで動作していたすべての`.tflite`ファイルが、LiteRTでも変更なしで動作します。**

「LiteRT vs TensorFlow Lite」「TensorFlow Liteは非推奨か」「LiteRTとは何か」と検索した場合、答えはこの段落のとおりです。このページでは、改称の理由と、LibreYOLOでモデルをエクスポートする方法を簡潔に説明します。

## Googleが改称した理由

TensorFlow Liteは2017年に、スマートフォンや組み込みボードでTensorFlowモデルを実行する方法として登場しました。その後、用途は大きく広がりました。GoogleがPyTorch、JAX、Kerasからの変換経路を構築したため、2024年までには、TensorFlowを一度も使ったことのないモデルが実行モデルのかなりの割合を占めていました。

その時点で、この名前は実態と合わず、誤解を招くものになっていました。TensorFlow専用のツールに聞こえるため、PyTorchユーザーは使わずにいました。そこでGoogleは2024年9月に[LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/)へ改称しました。これは「Lite Runtime」の略です。理由はそれだけです。同じチーム、同じコードで、フレームワークに依存しない名前になりました。

## 実際に変わったこと

変更はごくわずかで、既存のものに影響する変更はありません。

* コードの公開場所が新しくなりました：[github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert)。
* ドキュメントの場所が[ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)に変わりました。
* モデル実行用のPythonパッケージは`ai-edge-litert`になりました。以前の`tflite-runtime`パッケージは古く、新しいパッケージでも同じ`Interpreter`クラスを使います。
* 改称後、新機能はLiteRTの名前で提供されています。よりシンプルな`CompiledModel` APIと、Googleが[2026年1月に本番環境向けと宣言した](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/)GPU/NPUアクセラレーションです。

従来のInterpreter APIもこれまでどおり動作します。実際に新しく追加された拡張機能は`.litertlm`だけですが、これはオンデバイスLLM専用のパッケージ形式です。コンピュータビジョンモデルを扱う場合は関係ありません。そのため、あるドキュメントに「TFLite」、別のドキュメントに「LiteRT」と書かれていても、同じファイルを指しています。

## LibreYOLOモデルをLiteRTにエクスポートする

LibreYOLOはv1.3.0でTFLite/LiteRTへのエクスポートに対応しました。Python 3.12以降と追加パッケージが必要です。

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

CLIからも実行できます。

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

現在検証済みの経路は、YOLO9の物体検出と、RF-DETRの物体検出、セグメンテーション、姿勢推定です。いずれも実験的機能としてマークされています。対応状況の全体は[ドキュメント](/docs/v1.3.0)にあります。

エクスポートしたファイルを実行するには、対象デバイスにGoogleの`ai-edge-litert`パッケージをインストールします。

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, your preprocessed image here
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

入力はNHWC（チャンネルラスト形式）です。ONNXからの変換時にレイアウトが転置されますが、この形式では通常の動作です。

## LibreYOLOで変更したこと

変更はほとんどありません。変更が必要な箇所もほとんどなかったためです。ドキュメントでは形式名を「TFLite（LiteRT）」と表記し、両方の名前で検索できるようにしました。APIでは引き続き`format="tflite"`を形式名として使います。

## FAQ

**TensorFlow Liteは非推奨ですか？**
廃止されるのは名前であり、技術ではありません。ランタイムは同じAPIを備えたLiteRTとして引き続き提供され、既存のTFLiteモデルとコードもそのまま動作します。CompiledModel APIやNPUアクセラレーションなどの新機能は、LiteRTの名前で提供されます。

**LiteRT用にモデルを再エクスポートする必要がありますか？**
いいえ。TensorFlow Lite用にエクスポートした`.tflite`ファイルはLiteRTモデルです。ファイルに変更はないため、再エクスポートも移行も必要ありません。
