---
title: 結果の操作
seo_title: LibreYOLOのResultsオブジェクト
description: >-
  画像ごとに1つのResultsオブジェクトを返し、ボックス、マスク、キーポイント、確率、深度、パノプティック、OCRなどのペイロード型ごとにスロットを持ちます。描画、保存、JSONにも対応します。
lead: >-
  各推論は画像ごとにResultsオブジェクトを返します。ペイロードの種類ごとに名前付きスロットが1つあり、モデルが生成するもの以外は空です。エクスポートした成果物でも同じスロットを使います。
keywords:
  - yolo results オブジェクト python
  - results.boxes xyxy
  - results json 変換
  - アノテーション画像 保存
  - セグメンテーション マスク python
  - キーポイント results
  - 深度マップ results
  - results summary
  - onnx 同じ results
last_verified: 1.5.0
verification: >-
  ペイロードクラス、スロット、移動のセマンティクス、summary()、to_json()、plot()、save()、cutout()はlibreyolo/utils/results.pyで確認しました。アノテーションとディスク書き込みの動作はlibreyolo/models/base/inference.pyのInferenceRunner._save_annotated_imageとlibreyolo/utils/general.pyのresolve_save_pathで確認しました。サフィックスによる振り分けはlibreyolo/models/__init__.pyのLibreYOLO()で確認しました。
snippets:
  basic:
    - label: ボックス
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape)   # ソース画像の (height, width)
        print(result.path)         # ソースパス メモリ内入力では None

        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: 正規化座標
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy[:1])    # ピクセル x1 y1 x2 y2
        print(result.boxes.xywh[:1])    # ピクセル 中心 x 中心 y w h
        print(result.boxes.xyxyn[:1])   # 同じボックスを幅と高さで除算
        print(result.boxes.xywhn[:1])
    - label: NumPyとデバイス
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # それぞれ新しい Results を返し元のオブジェクトは変更しない
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summaryとto_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # 同じキーワード引数を使い同じ内容を文字列として返す
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: アノテーション済み画像
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True でペイロードを描画し runs/detect/predict* の下に書き込む
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: エクスポート用追加パッケージをインストール
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: エクスポート成果物から同じResultsを取得
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # 書き込んだパスを返す

        # LibreYOLO() はファイルサフィックスで振り分け
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## 1オブジェクト、ペイロードごとに1スロット

1枚の画像に対する推論は1つの`Results`を返します。18個のペイロードスロットがあり、モデルは自身のタスクが生成するものだけを埋めます。そのほかのスロットはすべて`None`なので、検出器で`result.masks`を読み取るとエラーではなく`None`になります。

| スロット | クラス | 形状 | 生成するタスク |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)`にスコアとクラスを追加 | 検出、および最初に位置特定を行うタスク |
| `masks` | `Masks` | `(N, H, W)` | インスタンスセグメンテーション |
| `keypoints` | `Keypoints` | `(N, K, 2)`または`(N, K, 3)` | 姿勢推定 |
| `probs` | `Probs` | `(C,)` | 分類 |
| `obb` | `OBB` | `(N, 7)`または`(N, 8)` | 回転バウンディングボックス |
| `gaze` | `Gaze` | `(N, 2)`のピッチとヨー（ラジアン） | 視線推定 |
| `points` | `Points` | x、y、class、confidenceの`(N, 4)` | 点の位置特定 |
| `semantic_mask` | `SemanticMask` | `(H, W)`のクラスID | セマンティックセグメンテーション |
| `panoptic` | `PanopticSegmentation` | `(H, W)`のセグメントIDと`segments_info` | パノプティックセグメンテーション |
| `depth_map` | `DepthMap` | `(H, W)`の浮動小数点数 | 深度推定 |
| `normal_map` | `NormalMap` | `(H, W, 3)`の単位ベクトル | 表面法線 |
| `edges` | `EdgeMap` | `[0, 1]`内の`(H, W)`浮動小数点数 | エッジ検出 |
| `restored` | `RestoredImage` | `(H, W, 3)`のuint8 RGB | 復元と超解像 |
| `matte` | `Matte` | `[0, 1]`内の`(H, W)`浮動小数点数 | アルファマッティングと背景除去 |
| `ocr` | `OCRRegions` | `(N, 4, 2)`のポリゴンと転記テキスト | テキスト検出と認識 |
| `embeddings` | `Embeddings` | `(N, D)`のL2正規化済み行 | `embed`タスク |
| `identities` | `Identities` | N個の名前とスコア | ギャラリーを伴う`embed`タスク |
| `meshes` | `Meshes` | 身体パラメータと任意の頂点 | 人体メッシュ復元 |

これらに加えて、すべての結果が持つフィールドがあります。`orig_shape`は`(height, width)`、`path`はソースパス（メモリ内入力では`None`）、`names`はクラスIDからクラス名へのマッピング、`frame_idx`は動画とライブフレーム用、`track_id`はトラッキング時に使われます。`restore_scale`は復元結果の整数アップスケール係数です。

`result.normals`は`result.normal_map`の別名です。

`result.speed`はすべての結果に存在しますが、値が設定されるのは[アンサンブル](/docs/predict/ensembling)だけです。キーは`member_0`、`member_1`、`fusion`で、単位はミリ秒です。単一モデルでは空の辞書のままです。

## ボックス

<code-tabs name="basic" />

`Boxes`は座標とスコアを1つのパック済みテンソルではなく、別々の配列として保持します。

| 属性 | 内容 |
|---|---|
| `xyxy` | `(N, 4)`の絶対ピクセル座標、x1 y1 x2 y2 |
| `xywh` | `(N, 4)`の絶対ピクセル座標、中心x、中心y、幅、高さ |
| `xyxyn`, `xywhn` | 同じ値を画像の幅と高さで除算 |
| `conf` | `(N,)`の信頼度 |
| `cls` | `(N,)`のクラスID、浮動小数点数 |
| `id` | `(N,)`のトラックID、または`None` |
| `is_track` | `id`が設定されているか |
| `data` | すべてを連結した値。ボックス、任意のID、信頼度、クラス |

`cls`は浮動小数点配列なので、`result.names[int(cls)]`として使ってください。

`xyxyn`と`xywhn`には`orig_shape`が必要で、`Results`が自動的に設定します。

## 密なペイロード

画像全体を覆うペイロードはインスタンスごとのペイロードとは異なる動作をし、スライス時に重要になります。

`SemanticMask`は元のキャンバス上の`(H, W)`クラスIDを保持し、`255`をクラスとして数えない無視値として予約します。`classes`は存在するIDを列挙して無視値を除外し、`class_mask(id)`はブール値の`(H, W)`を返します。

`PanopticSegmentation`は`(H, W)`のセグメントIDを保持し、`0`をvoid IDとして使います。`segments_info`は少なくとも`id`と`category_id`を持つ辞書のリストです。`segment_ids`は存在するIDを列挙し、`segment_mask(id)`は1つを選択します。

`DepthMap`は`(H, W)`の相対的な逆深度を保持します。値が大きいほど近く、メートル単位の距離ではありません。有限値に対する`min`、`max`、`mean`を公開し、`normalized()`で`[0, 1]`へ再スケーリングします。

`NormalMap`はOpenCVカメラ座標系の`(H, W, 3)`単位ベクトルを保持します。`+x`は右、`+y`は下、`+z`はシーンの奥を向くため、カメラに正対する面は`(0, 0, -1)`です。`assert_normalized()`は各ピクセルが有限で単位長であることを確認します。

`EdgeMap`は`[0, 1]`内の`(H, W)`float32を保持します。連続マップはしきい値処理されずに維持されるため、`binary(threshold=0.5)`でカットオフを選択します。

`Matte`は`[0, 1]`内の`(H, W)`float32を保持し、`1`が完全な前景です。`array`はfloat32としてクリップした値を返します。

`RestoredImage`は`(H, W, 3)`のuint8 RGBを保持し、`array`で未加工のndarrayを取得し、`save(path)`で書き出せます。

`Probs`は画像の確率ベクトルを1つ保持します。`top1`と`top5`はクラスインデックス、`top1conf`と`top5conf`は対応するスコアです。

`Embeddings`はL2正規化済みの`(N, D)`行を保持するため、コサイン類似度は内積になります。`similarity(other)`はギャラリーに対して`(N, M)`、単一ベクトルに対して`(N,)`を返し、`verify(i, j, threshold=0.4)`は2つの行を比較します。

`OCRRegions`は読み取り順の`(N, 4, 2)`ポリゴンを保持し、コーナーの順序は左上、右上、右下、左下です。転記テキストは`texts`、認識スコアは`conf`、検出スコアは`det_conf`にあります。これらは実際の回転ポリゴンなので`boxes`には値を設定しません。長方形が必要な場合は`ocr.xyxy`で軸平行の外接矩形を取得できます。

## スライスと移動

`result[i]`は1つのインスタンスを保持する新しい`Results`を返します。インスタンスごとのペイロードはスライスされ、画像全体のペイロードは変更せずに引き継がれます。そのため、分類結果をスライスしても確率ベクトルが単一クラスに切り詰められず、深度結果をスライスしても`(H, W)`の配置が壊れません。

`len(result)`はインスタンス数を数えます。ボックス、点、埋め込みベクトル、OCR領域、メッシュが対象です。密な画像全体のペイロードはどれも`1`として数えられ、何もない結果は`0`です。

`to()`、`cpu()`、`cuda()`、`numpy()`はそれぞれ、値が設定された全スロットを変換した新しい`Results`を返します。元のオブジェクトは変更しません。

`update()`だけがその場で変更するメソッドで、指定されたスロットを置き換え、同じオブジェクトを返します。

## JSON

<code-tabs name="json" />

`summary()`は通常の辞書のリストを返し、`to_json()`はそのリストを`json.dumps`へ渡します。どちらも同じ3つの引数を受け取ります。`normalize=False`は座標を`[0, 1]`へ切り替え、`decimals=5`は丸め桁数を設定し、`embeddings=False`は埋め込みベクトルを含めるかどうかを制御します。

行の形はペイロードに従います。検出行は`name`、`class`、`confidence`、`box`辞書を持ちます。マスクが存在する場合は`segments`、回転バウンディングボックスでは`obb`と`corners`、視線ではラジアンと度の両方の`gaze`角度、トラッキング時は`track_id`、メッシュが存在する場合は`mesh`パラメータが追加されます。

ボックスがない場合、1つのペイロードが行を決定します。OCRは領域ごとに`text`を持つ1行、点は点ごとに1行、パノプティックはセグメントごとに`pixel_count`と`pixel_fraction`を持つ1行、セマンティックは存在するクラスごとに1行、分類は上位5クラスを生成します。深度、法線、エッジ、復元、マッティングはそれぞれ、ピクセルではなくマップを説明する要約行を1つ生成します。

2つのペイロードは意図的に省略されます。512個の浮動小数点数を持つ1行は顔ごとに約2 KBになるため、埋め込みベクトルは`embedding_dim`だけで報告されます。値を含めるには`embeddings=True`を渡してください。メッシュ頂点は人物ごとに数万個の座標になるため、一切含まれません。形状データには`result.meshes.vertices`を読み取るか、`result.meshes.save_obj(path)`を呼び出してください。

## 描画と保存

<code-tabs name="saving" />

`predict(save=True)`がアノテーションを付けて書き出す経路です。値が設定されたスロットに応じて描画ルーチンを選択します。そのため、セマンティック結果は色付きマスク、深度結果は深度表示、パノプティック結果はセグメント付き、matteは背景が透明なRGBA PNG、検出結果は下にマスクを重ねたボックスとして書き出されます。書き込まれたパスは`result.saved_path`として結果に追加されます。

`Results.plot()`は名前から想像されるより対象が限定されています。法線マップとエッジマップだけで定義され、そのほかでは`NotImplementedError`が発生します。ほかのタスクでは`save=True`を使ってください。

`Results.save(path)`も同様に対象が限定されています。matte結果を背景が透明なRGBA PNGの切り抜きとして書き出し、そのほかでは`NotImplementedError`が発生します。`Results.cutout()`は書き込まずに同じRGBA配列を返します。どちらにもソース画像が必要で、`result.path`から取得するか`image=`で渡します。

2つのペイロードは固有の書き込み機能を持ちます。復元画像には`result.restored.save(path)`、メッシュには`result.meshes.save_obj(path, index=0)`を使います。

ファイルの保存先と`output_path`および`output_file_format`の動作については[推論ソース](/docs/predict/sources)を参照してください。

## エクスポートした成果物も同じオブジェクトを返す

<code-tabs name="exported" />

`LibreYOLO()`はファイルサフィックスで振り分けるため、エクスポートした成果物は`.pt`チェックポイントと同じ呼び出しで読み込まれ、同じ`Results`を返します。`.onnx`、`.engine`、`.pte`、`.mnn`ファイルはサフィックスで認識され、OpenVINO、Paddle、ncnnのディレクトリとTritonモデルURLも認識されます。モデルをエクスポート済みビルドへ切り替えても、`result.boxes.xyxy`を読み取るコードは変わりません。すべての形式については[エクスポート](/docs/export)を参照してください。

代わりにランタイム固有のAPIを使う場合は、前処理、後処理、クラス名を自分で管理する必要があります。

