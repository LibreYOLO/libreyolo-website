---
title: 顔認識
seo_title: LibreYOLOによる顔認識
description: LibreYOLOで顔の検出、埋め込み、識別を実行します。PythonまたはCLIからギャラリーを登録し、2枚の画像を比較して、コサイン類似度で照合します。
lead: >-
  顔認識は顔に適用するembedタスクです。検出器が各顔を特定して整列し、認識ヘッドが顔ごとにL2正規化済みベクトルを返します。固定クラスリストではなく、登録済みの参照とのコサイン類似度によって人物を識別します。
keywords:
  - Python 顔認識
  - 顔埋め込み
  - 顔照合
  - 顔ギャラリー
  - ArcFace ONNX
  - LibreYOLO embedタスク
  - 顔 コサイン類似度
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # librefacerec-*の名前はファイル拡張子にかかわらず顔埋め込みファミリーへ
        # 振り分けられ 初回使用時にデフォルト顔検出器とともに
        # LibreYOLOのHugging Face組織からダウンロードされる
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # (N, 4) 顔のボックス
        print(result.embeddings.data.shape)  # (N, D) 顔ごとに1行
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

        # 両画像で検出と埋め込みを実行し最も信頼度の高い顔を比較する
        # コサイン類似度の範囲は[-1, 1]
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: ギャラリーへ登録して識別
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
    - label: CLIで登録して識別
      language: bash
      code: >
        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=faces.npz

        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg
        gallery=faces.npz
    - label: 独自の顔ボックスを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxesは検出を完全に省略する face_detectorには
        # callable LibreYOLO検出モデル FaceDetectorインスタンスを指定できる
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## 定義

顔認識はラベルではなく、顔ごとにベクトルを返します。推論は2段階で実行されます。まず顔検出器が各顔と5つのランドマークを特定し、クロップを標準化された112×112の配置へ変形します。次に認識ヘッドがL2正規化済みの埋め込みベクトルを出力します。

`result.embeddings`は`(N, D)`形状の`Embeddings`ペイロードで、`result.boxes`と行が対応します。そのため、行`i`はボックス`i`内の顔を表します。各行は単位ベクトルなので、コサイン類似度は内積になり、`embeddings.similarity()`を使うと、別の`Embeddings`または行列全体との値を1回で計算できます。

顔への名前付けは別の処理です。`Gallery`は名前付きの参照ベクトルを保持します。`predict()`へ`gallery=`を渡すと、埋め込みベクトルと行が対応する`result.identities`が追加され、顔ごとに名前と最良のコサインスコアを保持します。一致しきい値未満の顔の名前は`None`のままで、しきい値未満で最も近い名前へ置き換わることはありません。

ライブラリの標準タスクキーは`embed`です。`face-recognition`、`facial-recognition`、`reid`、`face`はすべてこのキーへ正規化されるため、`task="face-recognition"`と`task="embed"`はまったく同じものを選択します。顔はこの広いタスクにおける領域単位の形です。[埋め込み](/docs/tasks/embeddings)では、画像全体とテキストの形、共通の`Embeddings`、`Identities`、`Gallery` API、何も検出せずベクトルを生成するモデルを解説しています。

## モデル

[LibreFaceRec](/docs/models/librefacerec)はこのタスク用のファミリーです。1つの呼び出しの背後に2つのONNXアーティファクトがあります。`librefacerec-l.onnx`は512次元の埋め込みベクトルを生成するiResNet100認識ヘッド、`librefacerec-det.onnx`はOpenCV zooから取得した5点ランドマーク対応のデフォルト顔検出器です。どちらも初回使用時にLibreYOLOのHugging Face組織からダウンロードされます。他のArcFace規約のONNXファイル（整列済み112×112入力、`(N, D)`出力）も、`librefacerec-*`の名前の代わりにパスを渡すことで認識ヘッドとして使用できます。

`embed`タスクキーは顔だけを対象とするものではありません。[CLIP](/docs/models/clip)、[SigLIP2](/docs/models/siglip2)、[DINOv2](/docs/models/dinov2)も`task="embed"`に対応し、画像全体のベクトルを1つ返します。これは顔の識別ではなく画像検索です。これらも`Gallery`と`Embeddings` APIを共有するため、後述の登録と照合の流れを適用できますが、顔の検出や整列は行いません。

認識ヘッドは基本インストールに含まれない`onnxruntime`で実行されます。

```bash
pip install "libreyolo[onnx]"
```

## 推論

<code-tabs name="predict" />

指定しない場合、`predict()`はデフォルト検出器をダウンロードして組み合わせます。`face_detector`を指定すると、callable、LibreYOLO検出モデル、`FaceDetector`インスタンスのいずれかで置き換えられ、コンストラクターまたは呼び出しごとに設定できます。`face_boxes`は保持しているボックスを使って検出を省略します。CLIでは`face_detector=`に顔検出器の`.onnx`パスまたはLibreYOLO検出器の名前を指定できます。

`model.verify(image_a, image_b)`は2枚の画像用の短縮処理です。各画像で最も信頼度の高い顔を埋め込み、`{"similarity", "same_person", "threshold"}`を返します。`model.embed(sources)`は1枚以上の画像に含まれるすべての顔の行を、1つの`(N_total, D)`テンソルへ積み重ねて返します。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

登録では人物ごとに1つのフォルダーを読み取ります。フォルダー名が人物名となり、中の各画像がその人物の参照として追加されます。

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll`はこのツリーをたどり、`.npz`ギャラリーを書き込みます。既存のギャラリーファイルは置き換えず、その場で拡張されるため、後から人物を追加できます。ギャラリーは埋め込み次元とファイルのフィンガープリントにより、ベクトルを生成した重みに結び付けられます。異なるモデルで照合しようとすると、互換性のないベクトル空間を比較せず例外を送出します。

デフォルトでは各ソース画像から最も信頼度の高い1つの顔を参照行として追加するため、周囲の人が写ったポートレートでも被写体だけが登録されます。`Gallery.enroll`へ`select="all"`を渡すと、返されたすべての行を保存します。

## 学習

このタスクのファミリーはLibreYOLO内部では学習できません。`LibreFaceEmbedder.train()`は例外を送出します。アップストリームで認識ヘッドを学習し、ArcFace規約のONNXへエクスポートして、ファイルパスから読み込んでください。

## 検証

このタスクにはデータセット検証器がなく、`val()`は存在するように装わず例外を送出します。照合精度はラベル付きの画像ペアに対して`model.verify()`を実行し、`threshold`を変えて目的の運用点を選ぶことで測定します。識別精度はギャラリーを登録し、保留しておいた画像に対する`result.identities.name`と`result.identities.score`を読み取って測定します。名前が`None`なら拒否として数えます。

## エクスポート

認識ヘッドはすでにONNXグラフなので、変換するものはなく、`LibreFaceEmbedder.export()`は例外を送出します。`.onnx`ファイルを直接デプロイするか、LibreYOLOにそのパスを渡して検出、整列、正規化をファミリーに処理させてください。
