---
title: 埋め込みベクトル
seo_title: LibreYOLOの画像および領域の埋め込みベクトル
description: >-
  embedタスクは、画像全体、検出領域ごと、またはテキストに対するL2正規化済みfloat32ベクトルを返します。ギャラリーへの登録、コサイン類似度による照合、PythonまたはCLIからの検索を行えます。
lead: >-
  LibreYOLOが生成するすべてのベクトルを1つのタスクで扱います。行が画像全体、検出された1つの顔、1行のテキストのどれを表す場合でも、embedはドット積が類似度スコアになる単位長のfloat32行を返し、同じGalleryですべてを照合できます。
keywords:
  - 画像 エンベディング python
  - l2 正規化 エンベディング
  - コサイン類似度 検索
  - libreyolo embed タスク
  - 類似画像検索
  - gallery 登録
  - clip エンベディング
  - dinov2 エンベディング
  - reid エンベディング
last_verified: 1.5.0
verification: >-
  libreyolo/tasks.pyからタスクキーと別名を確認しました。libreyolo/utils/results.pyのEmbeddingsクラスとIdentitiesクラスから結果ペイロードを確認しました。libreyolo/utils/gallery.pyからGallery
  APIを確認しました。libreyolo/models/base/model.pyからembedと_postprocess_embeddingsを確認しました。libreyolo/models/**/model.pyのSUPPORTED_TASKSにあるembedを検索して、対応ファミリーを特定しました。libreyolo/cli/__init__.py、libreyolo/cli/commands/special.py、libreyolo/cli/commands/predict.pyからCLIインターフェースを確認しました。docs/adr/0015-embed-generalization.mdから設計意図を確認しました。
meta:
  - label: タスクキー
    value: embed
    mono: true
  - label: 別名
    value: 'face-recognition, reid, face'
    mono: true
  - label: 結果ペイロード
    value: 'Embeddings, Identities'
    mono: true
  - label: 行のdtype
    value: float32、単位長
snippets:
  predict:
    - label: 画像全体
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # CLIPのデフォルトはclassifyなので、ベクトルを明示的に要求
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)  # (1, 512)、画像ごとに1行
        print(result.boxes)                  # None、位置特定はなし
    - label: 領域ごと
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # 行iはボックスiの領域を表す
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: 多数の画像を一度に処理
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # すべての結果の全行を1つのテンソルへ連結
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: テキスト
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        # テキストはメソッドであり、推論ソースにはならない
        # model(...)に渡した文字列は引き続きパスまたはURL
        text = model.embed_text(["a photo of a cat", "a photo of a dog"])
        print(text.shape)  # (2, 512)
  similarity:
    - label: 2組の行を比較
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # 行は単位長なので、コサイン類似度はドット積
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: 画像とテキストを比較
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: 登録して識別
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # しきい値未満ではnameはNone
    - label: Top-k検索
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # 先頭行の[(name, score), ...]
    - label: 保持済みベクトルを登録
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # 入力時に正規化
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: フォルダーツリーを登録
      language: bash
      code: >
        # source/<identity>/*.jpg。既存ギャラリーはその場で拡張

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: 推論しながら識別
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: 2枚の画像を比較
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verifyは別名で登録された同じコマンド

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## 定義

`embed`は画像、画像内の領域、または文字列を、長さが1の固定幅float32行へ変換します。
すべての行が単位ベクトルなので、2つの比較はドット積となり、2組の比較は1回の行列乗算に
なります。タスクの他の部分はモデルに固有ではありません。検索、重複検出、再識別、顔認識は、
いずれも異なる行に対する同じ演算です。

ベクトルが出力です。クラスリストはないため、名前はネットワークが学習時に推論するものではなく、
後から指定した参照との比較によって付けられます。

### 3つの形状

| 形状 | `Results.embeddings` | `Results.boxes` | 生成方法 |
|---|---|---|---|
| 画像全体 | `(1, D)` | `None` | 画像全体を扱うファミリーに画像を渡す |
| 領域 | `(N, D)` | `(N, 4)`、行が対応 | 顔認識など、最初に位置特定を行うファミリー |
| テキスト | `Results`ではない | | `(M, D)`を返す`model.embed_text(texts)` |

画像全体の結果は、画像が1枚でも2次元のままです。`(D,)`は許可された戻り値の形状ではないため、
利用側は1行だけの場合を特別扱いする必要がありません。テキストは`Results`ではなく通常の
テンソルを返します。文字列は画像ソースではないためです。`model(...)`に文字列を渡した場合は
引き続きパスまたはURLを意味し、ライブラリが文字列を文章だと推測することはありません。

正規のタスクキーは`embed`です。`embedding`、`embeddings`、`face-recognition`、
`facial-recognition`、`recognition`、`face`、`faceid`、`reid`はすべて`embed`に正規化されるため、
`task="reid"`と`task="embed"`は完全に同じものを選択します。

## モデル

4つのファミリーがこのタスクを提供し、最初に何かを位置特定するかどうかで明確に分かれます。

| ファミリー | 形状 | 次元 | その他の対応タスク |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | 領域。検出された顔ごとに1行 | 512 | なし。`embed`が唯一のタスク |
| [CLIP](/docs/models/clip) | 画像全体。対応するテキストタワーあり | `b32`と`b16`は512、`l14`は768 | デフォルトのままの`classify` |
| [SigLIP 2](/docs/models/siglip2) | 画像全体。対応するテキストタワーあり | `b16`は768、`so400m`は1152 | デフォルトのままの`classify` |
| [DINOv2](/docs/models/dinov2) | 画像全体。画像のみ | 384 | `semantic`、`classify` |

CLIPとSigLIP 2は`classify`をデフォルトタスクとして維持するため、`task="embed"`を明示する
必要があります。既存の`-cls`チェックポイントは共有の2タワー成果物です。同一の重みに対して
重複する`-embed`チェックポイントは公開されません。

`embed_text`があるのは、テキストタワーを持つ2ファミリーのCLIPとSigLIP 2だけです。
DINOv2にはありません。DINOv2の埋め込み処理は、セマンティックおよび分類ヘッドを迂回し、
224ピクセルで最終正規化済みCLSトークンを読み取ります。`n`、`s`、`m`、`l`の各バリアントは
すべてDINOv2-Sエンコーダーを共有するため、4つとも`D = 384`を返します。

このリリースで追加された分類専用バックボーンの[ViT](/docs/models/vit)、
[Swin](/docs/models/swin)、[DeiT](/docs/models/deit)は`classify`だけを宣言し、このタスクは
提供しません。

<code-tabs name="predict" />

`model.embed(source, **kwargs)`はバッチ処理用のショートカットです。`predict`を実行し、
全結果のすべての行を1つの`(N_total, D)` CPU float32テンソルへ連結します。行の次元が混在する
場合は例外を発生させます。対応タスクに`embed`がないファミリーは`NotImplementedError`を
発生させます。

## 結果ペイロード

`result.embeddings`は`Embeddings`ペイロードです。その`data`は常に`(N, D)` float32で、
推論経路によってすでにL2正規化されています。2次元でない入力は通知なく形状変更されず、
例外を発生させます。

| メンバー | 意味 |
|---|---|
| `.data` | `(N, D)`行列 |
| `.dim` | `D` |
| `.normalized` | 念のため再正規化した同じ行 |
| `.similarity(other)` | 別の集合に対しては`(N, M)`、単一の`(D,)`ベクトルに対しては`(N,)` |
| `.verify(i, j, threshold=0.4)` | 行`i`と`j`が同じ対象かどうか |

`result.identities`は`Identities`ペイロードで、ギャラリーを渡した場合だけ存在します。
これはテンソルではなく通常のコンテナーなので、`Results`をデバイス間で移動しても変更されません。

| メンバー | 意味 |
|---|---|
| `.name` | 名前のリスト。しきい値を超えたものがない箇所は`None` |
| `.score` | 最良のコサインスコアを持つ`(N,)` float32。名前が`None`でも保持 |
| `.data` | `(name, score)`タプルのリスト |

<code-tabs name="similarity" />

512個の浮動小数点数を持つ行は対象ごとに約2キロバイトになるため、デフォルトではベクトルを
`summary()`と`to_json()`に含めません。代わりに各行が`embedding_dim`を報告し、ギャラリーを
使用した場合は`identity`と`identity_score`も報告します。数値を含めるには
`summary(embeddings=True)`を渡します。

## ギャラリー

`Gallery`は名前付きの参照行の集合です。参照を平均せず個別に保存するため、名前のスコアには
単一の最良一致参照が使われます。品質の悪い写真を追加しても人物識別の重心がずれることは
ありません。

<code-tabs name="gallery" />

`Gallery(model)`は、ベクトルを生成する重みに関連付けられます。
`enroll(name, sources, select="best")`は各ソースで推論を実行し、結果ごとに最も信頼度が高い行を
保持します。`select="all"`では代わりにすべての行を保持します。参照画像に複数の対象が正当に
含まれる場合に適した設定です。`enroll_embedding(name, vector)`は推論を省略してベクトルを
直接受け取り、正規化し、全要素が0の行を拒否します。

`FaceGallery`は同じクラスの恒久的な別名で、以前の顔専用リリースで書き出したアーカイブも
引き続き読み込めます。

### 照合としきい値

照合では、保存済みの全参照に対して密な行列乗算を行い、最大値を取ることで名前ごとに1つの
スコアへ集約します。近似インデックスはありません。これにより数値は正確に保たれますが、
ギャラリーのサイズには実用上の上限があります。

2つのエントリーポイントは、しきい値未満の処理が異なります。`match()`は行ごとに
`[(name, score), ...]`を返し、しきい値未満をすべて破棄するため、一致がない行は空のリストです。
`identify()`は常に最良スコアを維持し、しきい値未満の場合は名前を`None`に設定した
`Identities`ペイロードを返します。どちらも、しきい値未満で最も近い名前を代用しません。

デフォルトのしきい値は全体で`0.4`です。これは確率ではなくコサイン値です。適切な動作点は、
データと誤一致への許容度によって決まるため、デフォルトをそのまま受け入れず、ラベル付きペアで
値を変えながら評価してください。`libreyolo enroll`と`gallery=`推論引数は同じ値を使います。

### 永続化

`save(path)`は、ベクトル、名前、形式バージョン、埋め込み次元、行を生成した重みのフィンガー
プリントを含むメタデータブロックを格納した圧縮`.npz`を書き出します。
`Gallery.load(path, model=...)`は比較前に両方を確認するため、ギャラリーを異なるモデルに向けると、
関連のない2つの空間から得たベクトルを通知なく採点せず、例外を発生させます。空のギャラリーは
保存できません。

## コマンドライン

| コマンド | 用途 |
|---|---|
| `libreyolo enroll` | 人物識別ごとのフォルダーツリーを走査し、`.npz`ギャラリーを書き出すか拡張 |
| `libreyolo compare` | 2枚の画像の主要対象を埋め込み、コサイン類似度を報告 |
| `libreyolo verify` | 別名で登録された同じコマンド |
| `libreyolo predict gallery=...` | 通常の推論実行に人物識別を付加 |

<code-tabs name="cli" />

すべてのLibreYOLOコマンドは`key=value`と`--key value`の両方を受け付けるため、
`gallery=refs.npz`と`--gallery refs.npz`は同じ引数です。

`enroll`は`model`、`source`、`gallery`に加えて、オプションの`face-detector`、`device`、
`--json`、`--quiet`を受け取ります。人物識別ごとに1つのフォルダーを読み取り、フォルダー名を
人物識別として、中の各画像から参照を追加します。

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

何も生成しない画像は実行を中断せず、stderrへの1行とともにスキップされます。概要では名前ごとに
保存された参照数を報告します。既存のギャラリーファイルはその場で拡張されるため、時間をかけて
人物識別を追加できます。

`compare`と`verify`は、2回登録された1つの関数です。`model`、`source`、`source2`、および
オプションの`threshold`を受け取り、コサイン類似度、同一か別かの判定、その判定に使った
しきい値を出力します。`--json`は同じ3フィールドをオブジェクトとして出力します。

`predict`では、`gallery`が保存済み`.npz`を指し、`gallery_threshold`がデフォルトの`0.4`を
上書きします。タスクが`embed`ではないモデルにギャラリーを渡すと、通知なく何もしないのでは
なくエラーになります。ギャラリーファイルがない場合は、作成に使える`libreyolo enroll`
コマンドが提案されます。

## 顔

顔認識はこのタスクの領域形式であり、同形式として提供される唯一の実装です。埋め込みヘッドの
前段に検出と位置合わせのステージを追加し、さらに`verify()`メソッド、独自ボックスを渡す引数、
公開済みの精度値、しきい値のキャリブレーション手順を提供します。それらはすべて
[顔認識](/docs/tasks/face-recognition)にあります。対象が顔の場合に従う手順です。このページの
内容はすべて変更なく適用されます。

## 学習、検証、エクスポート

このタスクはLibreYOLO内で何も学習しません。顔埋め込みヘッドはONNX成果物で、その`train()`、
`val()`、`export()`はすべて例外を発生させます。ヘッドはアップストリームで学習し、ファイルを
パスで読み込んでください。CLIP、SigLIP 2、DINOv2は、`embed`ではなく分類とセグメンテーションの
タスクを通じて学習およびエクスポートします。

検索用のバリデーターはありません。ラベル付きペアで`threshold`を変えながら検証精度を測定し、
ギャラリーを登録して、保留画像の`identities.name`と`identities.score`を読み取ることで識別精度を
測定してください。名前が`None`の場合は拒否として数えます。
