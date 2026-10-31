---
title: 顔認識
seo_title: LibreYOLOの顔認識
description: >-
  LibreYOLOで顔を検出し、埋め込み、識別します。PythonまたはCLIからgalleryを登録し、2枚の画像を比較して、cosine類似度で照合します。
lead: >-
  顔認識は顔に適用するembedタスクです。検出器がすべての顔を特定して位置合わせし、認識ヘッドが顔ごとにL2正規化されたベクトルを返します。IDは固定クラスリストではなく、登録済みreferenceとのcosine類似度で決まります。
keywords:
  - 顔認識 python
  - 顔 エンベディング
  - 顔認証
  - 顔 gallery
  - arcface onnx
  - libreyolo embed task
  - cosine 類似度 顔
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-*名はファイル接尾辞に関係なくface-embeddingファミリーへ
        # 経路を選び、初回使用時にデフォルトの顔検出器とともに
        # LibreYOLO Hugging Face orgからダウンロード
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # (N, 4)顔ボックス
        print(result.embeddings.data.shape)  # (N, D)、顔ごとに1行
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: 2枚の画像を比較
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # 両方の画像で検出と埋め込みを実行し、もっとも信頼度の高い顔を比較
        # Cosine similarityは[-1, 1]
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: galleryへ登録して識別
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # しきい値未満ではnameはNone
    - label: CLIから登録して識別
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: 独自の顔ボックスを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxesは検出を完全に省略。face_detectorはcallable
        # LibreYOLO検出モデル、FaceDetector instanceを受け付ける
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## 定義

顔認識はラベルではなく、顔ごとにベクトルを返します。推論は2段階で行われます。顔検出器が各顔と5つのlandmarkを特定し、cropをcanonicalな112x112配置へwarpし、認識ヘッドがL2正規化された埋め込みベクトルを出力します。

`result.embeddings` はshape `(N, D)` の `Embeddings` ペイロードで、`result.boxes` と行単位で対応します。そのため、行 `i` はボックス `i` の顔を表します。各行はunit vectorなので、cosine類似度は内積となり、`embeddings.similarity()` により別の `Embeddings` または行列全体に対して1回で計算できます。

顔への命名は別のステップです。`Gallery` は名前付きreference vectorを保持します。`predict()` へ `gallery=` を渡すと、埋め込みベクトルと行単位で対応する `result.identities` が追加され、顔ごとの名前と最良のcosine scoreを保持します。照合しきい値未満の顔は名前として `None` を維持し、しきい値未満でもっとも近い名前を代わりに設定することはありません。

ライブラリのcanonical task keyは `embed` です。`face-recognition`、`facial-recognition`、`reid`、`face` はすべてこの値へ正規化されるため、`task="face-recognition"` と `task="embed"` は同じものを選択します。顔は、より広いタスクにおけるregion shapeです。[埋め込みベクトル](/docs/tasks/embeddings)では、画像全体とテキストのshape、共有される `Embeddings`、`Identities`、`Gallery` API、何も検出せずにベクトルを生成するモデルを説明しています。

## モデル

[LibreFaceRec](/docs/models/librefacerec)はこのタスクのファミリーです。1回の呼び出しの背後に2つのONNX成果物があります。`librefacerec-l.onnx` は512次元の埋め込みベクトルを生成するiResNet100認識ヘッド、`librefacerec-det.onnx` はOpenCV zooから取得した5つのlandmarkを持つデフォルトの顔検出器です。どちらも初回使用時にLibreYOLOのHugging Face orgからダウンロードされます。ほかのArcFace規約のONNXファイル（位置合わせ済み112x112を入力し、`(N, D)` を出力）も、`librefacerec-*` 名の代わりにパスを渡せば、認識ヘッドとして利用できます。

`embed` タスクキーの対象は顔より広くなります。[CLIP](/docs/models/clip)、[SigLIP2](/docs/models/siglip2)、[DINOv2](/docs/models/dinov2)も `task="embed"` に対応し、画像全体について1つのベクトルを返します。これは顔のIDではなく、画像検索です。`Gallery` と `Embeddings` APIを共有するため、以下の登録・照合ワークフローを転用できますが、顔の検出や位置合わせは行いません。

認識ヘッドは、基本インストールに含まれない `onnxruntime` 上で動作します。

```bash
pip install "libreyolo[onnx]"
```

## 推論

<code-tabs name="predict" />

設定を変更しなければ、`predict()` はデフォルトの検出器をダウンロードして組み合わせます。`face_detector` により、callable、LibreYOLO検出モデル、`FaceDetector` instanceで上書きでき、constructorまたは呼び出しごとに設定できます。`face_boxes` はすでに持っているボックスを使い、検出を迂回します。CLIでは、`face_detector=` に顔検出器の `.onnx` パスまたはLibreYOLO検出器の名前を指定できます。

`model.verify(image_a, image_b)` は2枚画像用のshortcutです。それぞれでもっとも信頼度の高い顔を埋め込み、`{"similarity", "same_person", "threshold"}` を返します。`model.embed(sources)` は1枚以上の画像に含まれるすべての顔の行を単一の `(N_total, D)` テンソルへ積み重ねて返します。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

登録では、IDごとのフォルダーを読み取ります。フォルダー名がIDとなり、その中の各画像がその名前のreferenceになります。

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` はこのツリーを走査し、`.npz` galleryを書き込みます。既存のgalleryファイルは置き換えられず、その場で拡張されるため、時間をかけてIDを追加できます。galleryは、埋め込みベクトルの次元とファイルfingerprintにより、それを生成した重みに結び付けられます。別のモデルで照合しようとすると、互換性のないベクトル空間を比較せずに例外を送出します。

デフォルトでは各ソース画像から1つ、もっとも信頼度の高い顔のreference行だけが追加されるため、ほかの人が写るportraitでも被写体だけを登録します。返されたすべての行を保存するには、`Gallery.enroll` へ `select="all"` を渡してください。

## 学習

このタスクのどのファミリーもLibreYOLO内では学習できません。`LibreFaceEmbedder.train()` は例外を送出します。アップストリームで認識ヘッドを学習し、ArcFace規約のONNXへエクスポートして、ファイルをパスで読み込んでください。

## 検証

このタスクにはデータセット検証機能がなく、`val()` は対応しているように見せかけずに例外を送出します。認証精度は、ラベル付き画像ペアに対して `model.verify()` を使い、`threshold` を走査して目的の動作点を選ぶことで測定します。識別精度は、galleryを登録してhold-out画像の `result.identities.name` と `result.identities.score` を読み、`None` の名前を拒否として数えることで測定します。

## エクスポート

認識ヘッドはすでにONNXグラフなので、変換するものはありません。`LibreFaceEmbedder.export()` は例外を送出します。`.onnx` ファイルを直接デプロイするか、LibreYOLOへ渡して、ファミリーに検出、位置合わせ、正規化を処理させてください。

