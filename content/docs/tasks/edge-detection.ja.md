---
title: エッジ検出
seo_title: LibreYOLOによるエッジ検出
description: >-
  LibreYOLOで1枚の画像から密なエッジ確率マップを推論します。チェックポイントの変換、マップのしきい値処理、ODSとOISによる検証、エクスポートを解説します。
lead: >-
  エッジ検出は、各ピクセルが物体の境界上にある可能性を予測します。LibreYOLOではedgeタスクとして提供され、線分の集合ではなく元画像のキャンバス上に密な確率マップを返します。
keywords:
  - Python エッジ検出
  - 深層学習 境界検出
  - エッジ確率マップ
  - ODS OIS F値
  - 密なエッジ予測
last_verified: 1.5.0
snippets:
  predict:
    - label: エッジマップを推論
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreYOLOにはエッジのチェックポイントがないため先に下記の変換を行う
        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)          # (H, W) 範囲[0, 1]のfloat32
        print(edges.binary(0.5).sum())    # 0.5でのエッジピクセル数
    - label: 独自のしきい値を選択
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # しきい値を呼び出し側で決められるよう連続値のマップを維持する
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: 可視化を保存
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot()はマップを描画する エッジと法線の結果に定義されている
        result.plot().save("edges.png")
  val:
    - label: 検証して指標キーを確認
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: 探索範囲と対応許容距離を変更
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: エクスポート
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: エクスポートしたファイルを実行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## 定義

`edge`タスクは1枚のRGB画像からピクセルごとに1つの確率を予測します。`0`はエッジではないことを、`1`はエッジであることを表します。マップは連続値のままなので、二値の境界画像に変換するしきい値の選択は呼び出し側に委ねられます。適切なしきい値はデータセットや下流での用途によって異なります。

推論結果では`result.edges`に値が格納されます。これは元画像のキャンバス上にある`[0, 1]`範囲の`(H, W)` float32配列を保持する`EdgeMap`ペイロードです。`.array`はマップをNumPy配列として返し、`.binary(threshold)`はブールマスクを返します。`result.boxes`は空のままなので、`conf`、`iou`、`max_det`は効果がありません。`Results.plot()`はこのタスクに対応し、マップを直接描画します。

## モデル

`edge`には3つのファミリーが対応します。

[DexiNed](/docs/models/dexined)（Dense Extreme Inception Network）は複数の側方出力を1つの確率マップへ融合し、ネイティブの352pxで実行します。

[TEED](/docs/models/teed)（Tiny and Efficient Edge Detector）は、同じネイティブの352pxで動作する小型ネットワークです。ダウンサンプリングのストライドはDexiNedの16に対して4なので、より多くの`imgsz`値を受け付けます。

[LibreMODUS](/docs/models/libremodus)は、任意入力から任意出力へのモデルにある1つの対象としてCanny形式のエッジを生成します。`modus`追加パッケージと認証済みのHugging Faceアカウントが必要で、`val()`と`export()`には対応しないため、以下の検証とエクスポートのセクションの対象ではありません。

## 推論

LibreYOLOはエッジ用のチェックポイントを公開していません。公式にリリースされたDexiNedとTEEDの重みはBIPEDで学習されており、公開されているデータセットの条件では非商用目的に利用が制限されます。そのためLibreYOLOはそれらをミラーしません。利用する権利を持つチェックポイントを変換してから、変換済みファイルをパスで読み込んでください。

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

ローダーが認識するにはファイル名に`-edge`タスク接尾辞が必要です。`imgsz`はネットワークのダウンサンプリングストライドで割り切れる必要があり、満たさない場合はLibreYOLOが除数を示す明確なエラーを送出します。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## データセット形式

エッジの検証では、各RGB画像を同じ解像度かつ同じファイル名の基幹部を持つ単一チャンネルマップと対応付け、任意で有効性マスクも使用します。

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

対象はRGBの可視化画像ではなく、単一チャンネルのPNGまたはTIFです。整数マップはそのデータ型の最大値で割られます。floatマップは有限かつ`[0, 1]`範囲でなければなりません。マスクでは0以外のピクセルが有効と見なされ、パディングされたピクセルは指標に一切寄与しません。`edge_invert: true`は、白い背景に黒いエッジを保存するソースに対応します。完全な契約については[データセット形式](/docs/reference/dataset-formats)を参照してください。

## 学習

LibreYOLOのエッジファミリーには学習実装がありません。3つすべてで`train()`が`NotImplementedError`を送出します。各モデルページには、他で学習したチェックポイントをLibreYOLOで読み込める形式へ変換するスクリプトが記載されています。

## 検証

`val()`はBSDS形式のF値を報告します。連続値の予測は最初に4方向の勾配Non-Maximum Suppressionで細線化され、次に予測と正解のエッジピクセルを許容距離内で1対1に対応付けます。

<code-tabs name="val" />

`metrics/ODS`はデータセット全体で最適なF値です。各しきい値でデータセット全体の対応数をまとめ、そのF値の最大値を報告します。最良チェックポイントの選択で使う`fitness`にもなります。`metrics/OIS`は画像ごとの最適F値の平均で、各画像が固有のしきい値を選択できます。`metrics/best_threshold`はODSを生成した1つのしきい値で、推論時に`edges.binary()`で再利用する値です。

探索範囲を決める引数は2つあります。`edge_thresholds`は試すしきい値の集合で、デフォルトは0.01から0.99まで0.01刻みです。`edge_max_dist`は画像対角線に対する割合として表す対応許容距離で、デフォルトは`0.0075`です。これより離れたペアは一致と見なされません。

## エクスポート

エクスポートしたエッジモデルはファイルの拡張子に基づいて`LibreYOLO()`で再読み込みできるため、`.onnx`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。

<code-tabs name="export" />

エッジのエクスポートは固定解像度、バッチ1のランタイム契約を使用します。`dynamic`と1以外の`batch`は拒否され、エクスポート済みグラフは融合済みの確率マップを1つ出力します。形式ごとの対応範囲は[DexiNed](/docs/models/dexined)と[TEED](/docs/models/teed)のページ、および[完全なエクスポートマトリックス](/docs/reference/export-matrix)に記載されています。[エクスポート](/docs/export)には各形式で受け付ける引数が記載されています。
