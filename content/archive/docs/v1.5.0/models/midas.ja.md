---
title: MiDaS
families:
  - midas
seo_title: MiDaS：LibreYOLOでの単眼深度推定
description: >-
  LibreYOLOのMiDaSで単眼深度推定を行います。isl-orgからダウンロードされるMITライセンスの2つのバリアントをインストールし、推論、検証、エクスポートします。
lead: >-
  MiDaSは混合データセット上でスケール・シフト不変損失を使って学習した単眼相対深度推定モデルです。後のファミリーも再利用するゼロショット深度転移プロトコルを確立した研究系列です。LibreYOLOは深度タスク向けに、学習経路なしで推論とゼロショット検証に対応します。
keywords:
  - MiDaS
  - 単眼 深度推定
  - DPT
  - 相対深度
  - depth map
  - ゼロショット 深度推定
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ディスクにない場合は公式isl-org/MiDaS GitHubリリースから取得し
        # 使用前に固定されたSHA-256と照合
        model = LibreYOLO("LibreMiDaSl-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: smallバリアント
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # EfficientNet-Lite3エンコーダーでDPT-Largeのlサイズより小型で高速
        model = LibreYOLO("LibreMiDaSs-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイル接尾辞で振り分けるためエクスポート済み成果物も
        # チェックポイントと同様に読み込まれて同じResultsオブジェクトを返す
        model = LibreYOLO("LibreMiDaSl-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: ce2fbf3ae43e9be4
---

## インストール

MiDaSにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれます。

```bash
pip install libreyolo
```

## 推論

MiDaSは、LibreYOLOが独自のHugging Face組織で再公開しない唯一の深度ファミリーです。LibreYOLOのファイル名でチェックポイントを要求すると、`isl-org/MiDaS` のGitHubリリースから対応する公式成果物を直接ダウンロードし、固定されたSHA-256と照合した後、最初の使用前にLibreYOLOのチェックポイントメタデータでラップします。その後の実行では、キャッシュ済みのローカルファイルを再利用します。理由は「ライセンス」を参照してください。

<code-tabs name="predict" />

`result.depth_map` は密な相対逆深度マップを保持します。大きい値ほどカメラに近いことを示し、値にはメートル単位も画像をまたぐ共通スケールもありません。`save=True` はそのマップをカラーマップで可視化してディスクへ書き出します。`Results.plot()` は表面法線とエッジだけに定義されているため、このファミリーには対応しません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

2つのバリアントは同じモデルの単なるスケール違いではなく、異なるエンコーダーを使用します。`s` はEfficientNet-Lite3エンコーダーを持つMiDaS v2.1 Smallです。`l` はViT-L/16エンコーダーと、MiDaSが密な予測向けに導入したDPTデコーダーを持つDPT-Largeです。前処理も異なります。`s` はアスペクト比を維持する上限制約付きリサイズとImageNetの平均・標準偏差による正規化を使用し、`l` はアスペクト比を維持する最小制約付きリサイズと平均・標準偏差0.5を使用します。軽量なCNNには `s`、Transformerデコーダーの精度には `l` を選択してください。

このファミリーは学習を提供しません。`LibreMiDaS.train()` は無条件に `NotImplementedError` を発生させます。

## 検証

`val()` は共有の深度バリデーターを実行します。各推論を画像ごとの最小二乗スケールとシフトで正解データへ位置合わせし、標準のゼロショット相対深度指標であるAbsRel、RMSE、3つのdeltaしきい値を報告します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポート済み成果物はファイル接尾辞によって `LibreYOLO()` から再度読み込まれるため、`.onnx` または `.engine` ファイルはチェックポイントのように動作し、ボックスの代わりに `depth_map` を持つ同じ `Results` を返します。

<code-tabs name="export" />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
