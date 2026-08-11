---
title: L2CS-Net
families:
  - l2cs
seo_title: L2CS-Net：LibreYOLOでの視線推定
description: >-
  LibreYOLOのL2CS-Netで2段階の視線pitch/yaw推定を行います。インストール、推論、エクスポートに対応します。Gaze360チェックポイントは研究利用専用です。
lead: >-
  L2CS-Netは2段階の視線推定器です。顔検出器が顔を特定し、2つの角度ビン分類ヘッドを持つResNetトランクが顔ごとにpitchとyawを予測します。LibreYOLOは推論専用としてこれをラップします。
keywords:
  - L2CS-Net
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

        # face_detector未指定ではOpenCV同梱の顔検出器へフォールバック
        # OpenCV 4ではHaarでOpenCV 5ではYuNetを使用するため
        # L2CSチェックポイント以外に追加ダウンロードは不要
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 顔のソース
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # すでに実行した検出器のボックスをL2CSに渡す
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # または同梱された特定の顔検出器を指定
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

        # エクスポート済みグラフはResNetトランクと2つの角度ビンヘッドのみ
        # 前処理済み448x448顔クロップを受け取りデコード済み角度ではなく未処理の
        # (yaw_logits, pitch_logits)を返す softmaxとビン期待値と度への変換は
        # Python側に残る 詳細はlibreyolo.models.l2cs.utils.bin_logits_to_angles
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## インストール

手元にチェックポイントがある場合、L2CS-Netの構築、推論、エクスポートに追加パッケージは不要です。

```bash
pip install libreyolo
```

LibreYOLOが自動取得できる唯一のチェックポイントは、Gaze360で学習したResNet-50です。LibreYOLO組織ではなく著者のGoogle Driveにあるため、通常のHTTPミラーではなく `gdown` でダウンロードします。この経路には `gaze` 追加パッケージが必要です。

```bash
pip install "libreyolo[gaze]"
```

このパッケージがない場合、LibreYOLOは暗黙に失敗する代わりに手動ダウンロード手順を表示します。

## 推論

<code-tabs name="predict" />

L2CS-Netは2段階の推定器です。最初に顔検出器を実行し、視線ヘッドが返された各顔クロップからpitchとyawを読み取ります。指定を変えなければ、推論はOpenCV同梱の検出器へフォールバックするため、L2CSチェックポイントを用意した後は、追加のダウンロードなしで通常の呼び出しが動作します。`face_boxes` はすでに実行した検出器のボックスを受け付けます。`face_detector` は `"auto"`、`"haar"`、`"yunet"`、LibreYOLO検出モデル、または通常の呼び出し可能オブジェクトを受け付けます。`result.gaze` はラジアン単位のpitchとyawを保持し、検出された顔ボックスである `result.boxes` と行単位で対応します。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

5つのバックボーン深度が1つの入力解像度を共有し、同じ引数を受け取ります。唯一の公開チェックポイントの基になったデータセットであるGaze360はResNet-50を学習しています。その他の4つの深度はアーキテクチャとして対応していますが、読み込める公開済みの重みはありません。

## エクスポート

<export-matrix />

<code-tabs name="export" />

## ライセンス

<provenance-box>

LibreYOLOはL2CSチェックポイントをホストもミラーもしません。このサイトの他の多くのファミリーとは異なり、LibreYOLOのHugging Face組織にこのファミリーのものは存在しません。ライブラリが自動取得できる唯一のチェックポイントは、著者独自のGoogle Drive配布から直接取得します。転送開始前に表示されるGaze360ライセンス通知の対象であり、上の概要が示唆する「huggingface.co/LibreYOLOで再公開」されたコピーではありません。

</provenance-box>
