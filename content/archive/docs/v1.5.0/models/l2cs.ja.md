---
title: L2CS-Net
families:
  - l2cs
seo_title: L2CS-Net：LibreYOLOで視線推定
description: >-
  LibreYOLOのL2CS-Netで2段階の視線pitch・yaw推定を行います。インストール、推論、エクスポートを説明します。Gaze360チェックポイントは研究用途専用です。
lead: >-
  L2CS-Netは2段階の視線推定器です。顔検出器が顔を見つけ、2個のangle-bin分類ヘッドを持つResNet
  trunkが顔ごとにpitchとyawを予測します。LibreYOLOは推論専用でラップします。
keywords:
  - L2CS-Net 使い方
  - 視線推定
  - eye tracking
  - pitch yaw
  - Gaze360
  - 顔検出
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # face_detector未指定時はOpenCV同梱の顔検出器へフォールバック
        # OpenCV 4ではHaar、OpenCV 5ではYuNet。L2CSチェックポイント
        # 自体以外の追加ダウンロードなしで実行
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 顔の入力ソース
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # 実行済みの検出器からL2CSへボックスを渡す
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # または同梱された特定の顔検出器を名前で指定
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # エクスポートしたグラフはResNet trunkと2個のangle-binヘッドのみ
        # 前処理済み448x448顔クロップを受け取り、デコード済み角度ではなく生の
        # (yaw_logits, pitch_logits)を返す。softmax、bin期待値、度への変換は
        # Python側に残る。libreyolo.models.l2cs.utils.bin_logits_to_anglesを参照
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## インストール

すでにチェックポイントがあるモデルの構築、推論、エクスポートでは、L2CS-Netに追加パッケージは
必要ありません。

```bash
pip install libreyolo
```

LibreYOLOが自動取得できる唯一のチェックポイントは、Gaze360学習済みResNet-50です。LibreYOLO
組織ではなく作者のGoogle Driveにあるため、通常のHTTPミラーではなく`gdown`でダウンロードします。
この経路には`gaze`追加パッケージが必要です。

```bash
pip install "libreyolo[gaze]"
```

追加パッケージがない場合、LibreYOLOは黙って失敗せず、手動ダウンロードの手順を表示します。

## 推論

<code-tabs name="predict" />

L2CS-Netは2段階の推定器です。最初に顔検出器を実行し、返された各顔クロップから視線ヘッドが
pitchとyawを読み取ります。何も指定しない場合、推論はOpenCV同梱の検出器へフォールバックします。
そのため、L2CSチェックポイント自体を取得した後は、追加ダウンロードなしで通常の呼び出しが動作します。
`face_boxes`は実行済みの検出器からのボックスを受け付けます。`face_detector`は`"auto"`、`"haar"`、
`"yunet"`、LibreYOLO検出モデル、または通常のcallableを受け付けます。`result.gaze`にはラジアン単位の
pitchとyawが入り、検出された顔ボックスの`result.boxes`と行単位で揃います。入力ソース、
ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

5種類のバックボーン深度が1つの入力解像度を共有し、同じ引数を受け取ります。唯一の公開
チェックポイントの基になったデータセットGaze360はResNet-50を学習しました。ほかの4種類の
深度はアーキテクチャとして対応しますが、読み込める公開済みの重みはありません。

## エクスポート

<export-matrix />

<code-tabs name="export" />

## ライセンス

<provenance-box>

LibreYOLOはL2CSチェックポイントをホストもミラーもしません。このサイトの多くのファミリーと異なり、
LibreYOLOのHugging Face組織にこのファミリーのファイルはありません。ライブラリが自動取得できる唯一の
チェックポイントは作者自身のGoogle Drive配布元から直接取得されます。転送開始前に表示される
Gaze360ライセンス通知による制限があり、上記の概要が示唆する「huggingface.co/LibreYOLOで再公開」
されたコピーではありません。

</provenance-box>

