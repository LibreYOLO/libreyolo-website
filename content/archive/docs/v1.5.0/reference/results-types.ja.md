---
title: Resultsの型
seo_title: LibreYOLO Resultsオブジェクトリファレンス
description: >-
  LibreYOLOのResultsオブジェクトが保持できる全ペイロードを、タスク形状ごとに1スロットずつ説明します：ボックス、マスク、キーポイント、probs、obb、深度、OCR、埋め込みベクトル、その他10種。
lead: >-
  Resultsは、すべてのLibreYOLOモデルが画像ごとに返す単一の型です。タスク形状ごとに1つ、合計18個のオプションのペイロードスロットを持ち、モデルが生成したものだけを格納します。
keywords:
  - libreyolo results オブジェクト
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - Results.summary
  - libreyolo results json 変換
last_verified: 1.5.0
verification: >-
  v1.5.0のlibreyolo/utils/results.pyからスロット名、形状、プロパティ、デフォルト値を確認しました。意味はペイロードクラスのdocstringから引用しました。
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # すべてのペイロードをまとめて移動
        result = result.cpu().numpy()

        # 行を通常のdictで取得し、次にJSONへ変換
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## Resultsオブジェクト

1つの`Results`が1枚の画像を表します。単一画像のソースは1つ、リストのソースまたは
ディレクトリはリストを返し、`stream=True`はそれらを生成するジェネレーターを返します。

| 属性 | 型 | 意味 |
|---|---|---|
| `orig_shape` | `(int, int)` | 元画像の高さと幅 |
| `path` | `str` | 入力がディスク由来の場合のソースパス |
| `names` | `dict[int, str]` | クラスインデックスからクラス名への対応 |
| `speed` | `dict[str, float]` | ステージごとのミリ秒 |
| `track_id` | tensor | 結果が`track()`由来の場合のトラックID |
| `frame_idx` | `int` | 動画およびストリームソースのフレームインデックス |
| `restore_scale` | `int` | 復元結果の出力対入力アップスケール係数。他ではすべて`1` |

<code-tabs name="usage" />

## ペイロードスロット

モデルが生成しない限り、各スロットは`None`です。ファミリーが埋めるスロットは、そのタスクに
よって決まります。

| スロット | クラス | タスク |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | galleryを伴うembed |
| `meshes` | `Meshes` | mesh |

`result.normals`は`result.normal_map`の読み書き可能な別名です。

複数のスロットを同時に設定できます。セグメンテーションモデルは`boxes`と`masks`の両方を
埋めます。視線モデルは顔のボックスを`boxes`に、角度を`gaze`に格納します。メッシュモデルは
人物ボックスを`boxes`に格納し、それと行を対応させて`meshes`を埋めます。

## Boxes

1枚の画像に対する検出ボックス。

| メンバー | 戻り値 |
|---|---|
| `xyxy` | 元画像のピクセル単位の隅座標 |
| `xywh` | ピクセル単位の中心とサイズ |
| `xyxyn` | `[0, 1]`に正規化された隅座標 |
| `xywhn` | `[0, 1]`に正規化された中心とサイズ |
| `conf` | ボックスごとの信頼度 |
| `cls` | ボックスごとのクラスインデックス |
| `id` | ボックスごとのトラックID、または`None` |
| `is_track` | トラックIDがある場合は`True` |
| `data` | パック済みテンソル |

`with_id(id)`と`with_orig_shape(orig_shape)`は、該当フィールドを置き換えた新しい`Boxes`を
返します。

## Masks

1枚の画像に対するインスタンスマスク。`data`はマスクテンソルです。`xy`はインスタンスごとの
輪郭をピクセル単位で返し、`xyn`は正規化して返します。

## Keypoints

`boxes`と行が対応する姿勢キーポイント。`xy`はキーポイントごとの座標ペア、`xyn`は正規化済み
ペアです。データに第3チャンネルがあれば`conf`はその値、なければ`None`です。`has_visible`は
ブール配列で、`conf > 0`の位置がtrueになり、信頼度チャンネルがない場合はすべてtrueです。

## Points

1枚の画像に対する点の位置特定。`data`の形状は`(N, 4)`で、各行は`x, y, class, confidence`です。
座標は絶対ピクセルです。`xy`、`cls`、`conf`が各列を分け、`xyn`が座標を正規化します。

## Probs

分類スコア。`top1`は最上位のインデックス、`top5`は上位5件のインデックス、`top1conf`と
`top5conf`はそれぞれのスコアです。

## OBB

回転ボックス。`data`は行ごとに7個または8個の値を保持します：`xywhr`、オプションの
トラックID、信頼度、クラスの順です。

| メンバー | 戻り値 |
|---|---|
| `xywhr` | 中心、サイズ、ラジアン単位の回転 |
| `xyxyxyxy` | ピクセル単位の4つの隅 |
| `xyxyxyxyn` | 正規化された4つの隅 |
| `xyxy` | ピクセル単位の軸平行外接矩形 |
| `conf`, `cls`, `id`, `is_track` | `Boxes`と同じ |

## Gaze

顔ごとのラジアン単位の視線角度で、形状は`(N, 2)`、`boxes`の顔ボックスと行が対応します。
L2CSの規約に従い、列0はピッチ、列1はヨーです。正のヨーは被写体から見て左へ視線を回転させ、
正のピッチは下へ回転させます。`pitch_deg`と`yaw_deg`は度へ変換し、`direction_3d`は単位方向
ベクトルを返します。

## SemanticMask

元画像のキャンバス上にある、整数クラスIDの密なセマンティックマップで、形状は`(H, W)`です。
`255`は無視する値であり、クラスとして数えられることはありません
（`SemanticMask.IGNORE_INDEX`）。`classes`は存在するクラスIDを一覧化し、
`class_mask(class_id)`は1つのクラスに対するブールマスクを返します。

## PanopticSegmentation

すべてのピクセルに重複しないセグメントをちょうど1つ割り当て、stuff領域とthingインスタンスを
統合します。`data`は`(H, W)`の整数セグメントIDマップです。セグメントID`0`はラベルなしを
表します（`PanopticSegmentation.IGNORE_INDEX`）。`segments_info`はセグメントごとに1つの
dictを持つリストで、それぞれ少なくとも`{"id": int, "category_id": int}`を含みます。
ここで`id`はマップ内の値に一致し、`category_id`は`names`のインデックスです。
`segment_ids`は存在するIDを一覧化し、`segment_mask(segment_id)`は1つのセグメントの
ブールマスクを返します。

thingとstuffの区別はセグメントではなくカテゴリーのプロパティです。ペイロードはそれを
`"isthing": bool`として各セグメントへ非正規化する場合があり、その場合、値はカテゴリー
レベルのマップと一致する必要があります。

## DepthMap

元画像のキャンバス上にある、浮動小数点数の密な相対逆深度マップで、形状は`(H, W)`です。
値が大きいほどカメラに近いことを意味します。値は相対値であり、メートル単位の実測値では
ありません。`min`、`max`、`mean`は有限値に対して計算され、`normalized()`はマップを
`[0, 1]`に再スケーリングします。

## NormalMap

元画像のキャンバス上にある、float32の密な表面法線フィールドです。形状は`(H, W, 3)`で、
OpenCVのカメラ座標系を使います：`+x`は右、`+y`は下、`+z`はシーンの奥です。法線はカメラを
向くため、正面に平行な面は`(0, 0, -1)`です。すべてのピクセルは単位ベクトルです。
`assert_normalized(atol=1e-4)`はこの不変条件を確認します。

## EdgeMap

元画像のキャンバス上にある、float32の密なエッジ確率マップで、形状は`(H, W)`です。`0`は
非エッジ、`1`はエッジです。しきい値を呼び出し元が選べるよう、連続値マップが保持されます。
`binary(threshold=0.5)`がしきい値を適用し、`array`はnumpyビューを返します。

## RestoredImage

復元されたRGB画像で、形式は`(H, W, 3)`のuint8です。超解像では、キャンバスは入力の
`Results.restore_scale`倍になります。`array`はnumpyビューを返し、`save(path)`は画像を
書き出します。

## Matte

元画像のキャンバス上にある、`[0, 1]`範囲のfloat32によるソフト不透明度マットで、形状は
`(H, W)`です。`1`は完全な前景、`0`は完全な背景です。ソフトマットは0.5でしきい値処理した
強制的な背景除去マスクを包含し、バイナリマスクでは失われるアンチエイリアス処理済みの境界を
保持します。`array`はnumpyビューを返します。

マットの結果では、`Results.cutout(image=None)`が第4チャンネルをマットとするRGBAの
`(H, W, 4)` uint8配列を返し、`Results.save(path, image=None)`がその切り抜きを透明背景の
PNGとして書き出します。どちらも`image`が指定されていればそこからRGBを取得し、それ以外は
`Results.path`から再読み込みします。

## OCRRegions

位置が特定されたテキストと転記。`data`は元画像のピクセル単位による`(N, 4, 2)`の浮動小数点
ポリゴンで、左上、右上、右下、左下の順です。領域は上から下、次に左から右という読み順で
並びます。`texts`はN個の転記のリストです。`conf`は領域ごとの認識スコア、`det_conf`は
検出スコアで、どちらも形状は`(N,)`です。

検出四角形は実際のポリゴンであるため、`Results.boxes`には格納されません。`xyxy`は軸平行の
外接矩形を返します。

## Embeddings

`embed`タスクのL2正規化済みベクトルで、形状は常に`(N, D)`です。画像全体の結果は1行を持ち、
ボックスはありません。領域の埋め込みベクトルは`boxes`と行が対応します。各行は正規化されて
いるため、コサイン類似度はドット積になります。

| メンバー | 戻り値 |
|---|---|
| `dim` | `D` |
| `normalized` | 再正規化された行 |
| `similarity(other)` | 別の`Embeddings`またはテンソルに対するペア単位のコサイン類似度 |
| `verify(i, j, threshold=0.4)` | 行`i`と`j`が一致する場合は`True` |

## Identities

`embeddings`と行が対応する、名前付きギャラリーとの一致結果。`Gallery`を`embed`推論に渡すと
生成されます。`name`はリストで、一致のしきい値を下回る項目は`None`です。しきい値を下回る
最も近い名前が推測されることはありません。`score`は一致スコアの配列で、`data`は名前と
スコアをペアにします。

## Meshes

`boxes`の人物ボックスと行が対応する、パラメトリックな人体メッシュ。すべて元画像のカメラ
座標系にあります。`transl`はメートル単位の実測値で、`+z`はカメラから遠ざかる方向です。
`vertices`と`joints3d`は実測値で、すでに`transl`を含みます。`joints2d`はネットワークが
入力した切り抜きではなく、元画像のキャンバス上のピクセル単位です。ワールド座標系や重力
座標系を持つフィールドはありません。

パラメータの配置は人体モデルごとに異なるため、形状についてハードコードされているものは
ありません。`body_model`がパラメータ化方式を示し、`num_vertices`、`num_joints`、
`num_betas`、`has_vertices`の数値はテンソルから読み取られます。`params`はパラメータの
dictを返し、`save_obj(path, index=0)`は1つのメッシュを書き出します。フィールドは
`global_orient`、`body_pose`、`betas`、`transl`、`vertices`、`faces`、`joints3d`、
`joints2d`、`conf`、`focal_length`、`extras`です。

`body_model="mhr"`の場合、回転は軸角ではなくラジアン単位のオイラー角です。`body_pose`は
関節ごとの3要素ではなく、関節ごとの平坦なパラメータベクトルで、`betas`は人物識別用の
ブレンドシェイプ係数です。骨格スケール、手の姿勢、顔の表情は`extras`に格納されます。

## 変換と選択

すべてのペイロードは`to(*args, **kwargs)`、`cpu()`、`cuda()`、`numpy()`を持ちます。
`Results`でいずれかを呼び出すと、格納済みのすべてのスロットへ一度に適用されます。

<code-tabs name="convert" />

`result[idx]`は、行が対応するペイロード全体から行を選択します。`len(result)`は検出数で、
ボックスがなければ点の数です。`result.update(...)`は指定したスロットを置き換えたコピーを
返します。すべてのスロットに加えて、`track_id`と`restore_scale`を受け入れます。

## summaryとto_json

`summary(normalize=False, decimals=5, embeddings=False)`は通常のdictのリストを返します。
どのスロットが設定されているかに応じて、検出、セグメント、点、または領域ごとに1行です。
`to_json(**kwargs)`は引数を`summary`へ渡し、JSON文字列を返します。

`plot()`は密な法線またはエッジの結果を標準の可視化形式でレンダリングします。他の結果型では
例外を発生させます。その他のタスクのアノテーション付き画像は`predict(save=True)`で
生成されます。
