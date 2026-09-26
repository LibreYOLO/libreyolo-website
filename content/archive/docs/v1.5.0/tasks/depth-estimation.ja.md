---
title: 深度推定
seo_title: LibreYOLOによる単眼深度推定
description: LibreYOLOで1枚の画像から密な相対深度マップを推論します。深度ファミリーの比較、深度指標の読み方、深度モデルのエクスポートを解説します。
lead: >-
  深度推定は1枚の画像から各ピクセルとカメラの距離を予測します。LibreYOLOではdepthタスクとして提供され、元画像のキャンバス上に密な相対逆深度マップを返します。
keywords:
  - Python 単眼深度推定
  - 1枚の画像 深度マップ
  - 相対深度モデル
  - Depth Anything LibreYOLO
  - 密な深度予測
last_verified: 1.5.0
snippets:
  predict:
    - label: 深度マップを推論
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # 元キャンバス上の(H, W)
        print(depth.min, depth.max, depth.mean)
    - label: 値を処理
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map
        raw = depth.data          # 大きいほど近い メートル単位もスケールもない
        gray = depth.normalized() # 可視化用に[0, 1]へ再スケーリング
        print(raw.shape, float(gray.max()))
    - label: 小型の代替モデル
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 同じタスク契約でエッジランタイム向けに構築された非常に小さなネットワーク
        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
  val:
    - label: 検証して指標キーを確認
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: エクスポート
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: エクスポートしたファイルを実行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## 定義

`depth`タスクは1枚のRGB画像からピクセルごとに1つの値を予測します。LibreYOLOではその値を相対逆深度として定義します。値が大きいほどカメラに近く、メートルなどの単位はなく、2枚の画像間で共通するスケールもありません。同じ推論結果の2ピクセル間で深度を比較することには意味がありますが、別画像の値との比較には意味がありません。

推論結果では`result.depth_map`に値が格納されます。これは元画像のキャンバス上にある`(H, W)`配列を保持する`DepthMap`ペイロードです。`.min`、`.max`、`.mean`で有限値を読み取り、`.normalized()`で表示用にマップを`[0, 1]`へ再スケーリングできます。`result.boxes`は空のままなので、`conf`、`iou`、`max_det`は効果がありません。`save=True`は注釈付き写真ではなく、マップにカラーマップを適用した画像を書き込みます。

## モデル

`depth`には6つのファミリーが対応します。

[Depth Anything V2](/docs/models/depth-anything-v2)はDINOv2エンコーダーとDPTデコーダーを組み合わせ、ここでの汎用的なデフォルトです。精度と同様にライセンスもサイズ選択を左右します。SmallチェックポイントはApache-2.0ですが、BaseとLargeは非商用です。選択前にモデルページのチェックポイント表を確認してください。

[Depth Anything 3](/docs/models/depth-anything-3)はDA3MONO-LARGEチェックポイントを移植したもので、深度向けにアーキテクチャを特殊化していない通常のTransformerです。

[ZipDepth](/docs/models/zipdepth)は小型ティアです。Depth Anything V2 Largeから蒸留した再パラメーター化可能なCNNで、gatherやunfold演算に対応しないNPUコンパイラー向けに、これらの演算をデコーダーで回避する第2のチェックポイントもあります。

[MiDaS](/docs/models/midas)は、他のファミリーの測定にも使われるゼロショット相対深度の手法を確立した一連の研究です。LibreYOLOが再公開しない唯一の深度ファミリーで、チェックポイントを要求すると著者の公式GitHubリリースからアーティファクトをダウンロードし、固定済みのSHA-256を確認します。

[LibreMODUS](/docs/models/libremodus)は専用ヘッドではなく、任意入力から任意出力へのモデルにある1つの対象として深度を生成します。`modus`追加パッケージと認証済みのHugging Faceアカウントが必要で、`val()`と`export()`には対応しません。

[SenseNova-Vision](/docs/models/sensenova-vision)は、7つのタスクに対応する同じ7Bチェックポイントを使い、拡散デコードを通して深度マップを画像として生成します。`sensenova`追加パッケージが必要で、重みの利用は非商用に制限されます。ライセンスはモデルページに記載されています。

## 推論

前述の2ファミリーを除き、重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

入力解像度はファミリーごとに制約されます。Depth Anything V2とDepth Anything 3はDINOv2のパッチグリッドを基にするため、`imgsz`は14で割り切れる必要があり、LibreYOLOは実行前に確認します。`Results.plot()`はこのタスクには対応せず、サーフェス法線とエッジだけに定義されています。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

深度の検証では、各画像を同じ解像度の密な単一チャンネル深度マップと対応付けます。画像パス内の画像ディレクトリを深度ディレクトリへ置き換えてファイルを検索します。

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

マップは単一チャンネルのPNGまたはTIF、あるいは`.npy`です。値はデータセット内で一貫した単位の通常の深度です。`0`、負の値、NaN、無限大のピクセルは無効なサンプルとして指標から除外されます。整数マップは`depth_scale`で割られます。デフォルトは16ビットPNGの慣例である`256.0`です。floatの`.npy`マップはそのまま使用されます。`depth_stem_suffix`と`depth_mask_suffix`は、深度ファイルや有効性マスクに異なる命名規則を使うデータセットに対応します。完全な契約については[データセット形式](/docs/reference/dataset-formats)を参照してください。

## 学習

LibreYOLOの深度ファミリーには学習実装がありません。6つすべてで`train()`が`NotImplementedError`を送出します。各モデルページには、アップストリームで学習したチェックポイントをLibreYOLOで読み込める形式へ変換するスクリプトが記載されています。

## 検証

`val()`は共通の深度検証器を実行します。相対深度には絶対スケールがないため、まず画像ごとの最小二乗法によるスケールとシフトで各予測を正解の逆数に適合させ、次に深度へ戻します。以下の各指標は、データセットが有効と示すピクセルだけを数え、位置合わせ済みのマップに対して画像ごとに計算し、データセット全体で平均します。

<code-tabs name="val" />

`metrics/abs_rel`は平均絶対相対誤差です。残差を正解深度で割った値で、小さいほど良い指標です。`metrics/rmse`はデータセット固有の深度単位における二乗平均平方根誤差で、これも小さいほど良い指標です。`metrics/delta1`、`metrics/delta2`、`metrics/delta3`はしきい値精度です。予測と正解の比率を大きい側で取った値が、それぞれ1.25、1.25の2乗、1.25の3乗を下回る有効ピクセルの割合で、大きいほど良い指標です。`metrics/delta1`は最良チェックポイントの選択で使う`fitness`にもなります。

## エクスポート

エクスポートした深度モデルはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作します。返される`Results`にはボックスの代わりに`depth_map`が含まれます。

<code-tabs name="export" />

対応範囲はファミリーごとに異なり、Depth Anything 3は未検証の変換を試みず、検証済みの集合にない形式を拒否します。対象を決める前にモデルページと[完全なエクスポートマトリックス](/docs/reference/export-matrix)を確認してください。LibreMODUSとSenseNova-Visionはエクスポートできません。[エクスポート](/docs/export)には各形式で受け付ける引数が記載されています。
