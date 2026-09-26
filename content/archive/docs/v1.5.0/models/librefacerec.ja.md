---
title: LibreFaceRec
families:
  - facerec
seo_title: LibreFaceRec：顔認識と照合
description: >-
  LibreYOLOのLibreFaceRecで顔検出、埋め込みベクトル、照合を行います。インストールして推論できます。埋め込みベクトルの重みはApache-2.0です。
lead: >-
  LibreFaceRecはLibreYOLOの顔埋め込みタスクです。顔検出器が顔を特定して位置合わせし、認識ヘッドが照合または検索用のL2正規化済み識別埋め込みベクトルを生成します。
keywords:
  - LibreFaceRec
  - 顔認識
  - 顔 エンベディング
  - 顔照合
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-*名はファイル接尾辞に関係なくこのファミリーへ振り分け
        # 初回使用時にデフォルトの顔検出器とともにLibreYOLOの
        # Hugging Face組織からダウンロード
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (N, D)でL2正規化済み
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: 照合
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # 各画像で最も目立つ顔をL2正規化済み埋め込みベクトルの
        # コサイン類似度によって比較
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: ギャラリー検索
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # この画像内の顔
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # (query_faces, N_total)のコサイン類似度
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## インストール

LibreFaceRecの認識ヘッドは、基本インストールに含まれない `onnxruntime` を通じて実行されます。

```bash
pip install "libreyolo[onnx]"
```

## 推論

<code-tabs name="predict" />

1回の呼び出しの背後で、検出と認識を別々の2つのONNXグラフが処理します。顔検出器が各顔を特定して標準的なクロップに位置合わせし、認識ヘッドが顔ごとにL2正規化済み埋め込みベクトルを返します。指定を変えなければ、`predict()` は同梱されたデフォルト検出器を自動的にダウンロードして組み合わせます。`face_detector` は呼び出し可能オブジェクト、LibreYOLO検出モデル、または `FaceDetector` インスタンスを受け付けます。`face_boxes` にはすでに用意したボックスを渡し、検出を完全に省略できます。`result.embeddings` は検出した顔ごとに1行を持ち、`result.boxes` と対応しています。その `.similarity()` メソッドは、別の埋め込みベクトルまたはギャラリー全体に対するコサイン類似度を1回の呼び出しで計算します。計算済みの2つの埋め込みベクトルではなく2枚の画像を直接比較するには、`model.verify(image_a, image_b)` が両方に対して検出と埋め込みを実行し、最も信頼度の高い顔を比較します。ファイルパスを `librefacerec-*` 名の代わりに渡せば、ArcFace規約に従う他のONNX認識モデル（位置合わせ済みクロップを入力し、`(N, D)` の埋め込みベクトルを出力）へ置き換えることもできます。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## エクスポート

<export-matrix />

LibreFaceRecはエクスポート済みONNXグラフをすでにラップしています。別の形式への再エクスポートは実装されていません。

## ライセンス

<provenance-box>

同梱のデフォルト顔検出器は、別のライセンスを持つ2つ目の成果物です。OpenCV ZooのYuNetで、MITライセンス、著作権者はShiqi Yuです。どちらのプロジェクトからもアーキテクチャのコードは移植されていません。両方のグラフは `onnxruntime` を通じて不透明なものとして利用されるため、LibreYOLO独自のラッパーにサードパーティーのコードは含まれず、全体がMITです。

</provenance-box>

## 引用

<citation-block />
