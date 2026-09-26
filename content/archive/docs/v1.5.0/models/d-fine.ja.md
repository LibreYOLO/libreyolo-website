---
title: D-FINE
families:
  - dfine
seo_title: D-FINE：MITライセンスでファインチューニング、検証、エクスポート
description: >-
  LibreYOLOでD-FINEを使い、物体検出とインスタンスセグメンテーションを行います。インストール、推論、ファインチューニング、検証、エクスポートに対応し、コードはMITライセンスです。
lead: >-
  ボックス回帰を各ボックス辺の確率分布として再定式化し、デコーダー層を通じて改良する検出Transformerです。LibreYOLOは物体検出とインスタンスセグメンテーションでD-FINEをサポートします。
keywords:
  - D-FINE
  - detection transformer
  - リアルタイム 物体検出
  - インスタンスセグメンテーション
  - fine-grained distribution refinement
  - DETR
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDFINEn.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: インスタンスセグメンテーション
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の -seg サフィックスでマスクヘッドを選択するため task
        # 引数は不要
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDFINEn.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: インスタンスセグメンテーション
      language: bash
      code: |
        # 公開済みセグメンテーション重みから継続しマスクヘッドも含む
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: 検出重みからセグメンテーションへ
      language: bash
      code: |
        # 検出重みにマスクヘッドはないため明示的な転移を実行
        # ヘッドは未学習で始まり学習後にのみ有用になる
        # ここで task=segment を指定すると転移を許可
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: インスタンスセグメンテーション
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # マスク
        print(metrics["metrics/mAP50-95(B)"])   # ボックス
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640

        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640
        half=True
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイルサフィックスで振り分けるためエクスポート成果物も
        # 任意のチェックポイントと同様に読み込まれ同じ Results オブジェクトを返す
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 0216631a26185524
---

## インストール

D-FINEにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

`lora=True`によるアダプターのファインチューニングは例外で、`lora`追加パッケージが必要です。

```bash
pip install "libreyolo[lora]"
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーに共通するため、別の検出器への切り替えは1行の変更で済みます。ファイル名に`-seg`があるとセグメンテーションタスクとして自動的に解決され、`result.masks`にはボックスとともにインスタンスマスクが格納されます。`conf`と`max_det`はクエリ選択をフィルタリングします。デコーダーはNMSステップを持たない集合予測器なので、`iou`はAPIの互換性のため受け付けられますが効果はありません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

5つのサイズがあります。すべて同じ入力解像度で動作するため、表ではパラメータ数と精度で区別しています。

<benchmark-table task="detect" />

<va-embed />

セグメンテーションは検出のバックボーン、エンコーダー、デコーダーを再利用し、マスクヘッドを追加します。そのため、`-seg`チェックポイントは対応する検出チェックポイントと同じ引数を受け取ります。LibreYOLOのRT-DETRv4ファミリーはD-FINEラッパーのサブクラスとして実装されています。このデコーダー系統を継承したうえで、マスクヘッドを持たないためタスクリストを検出のみに固定し直します。

## 学習

どちらのタスクも公開済みチェックポイントから学習を開始します。

<code-tabs name="train" />

設定を変更しなければ、トレーナーは`lr0=2e-4`、`amp=False`、バッチサイズ16で132エポック実行し、改善がない状態が50エポック続くと早期終了します。検出重みはセグメンテーション学習の有効な開始点ですが、明示的な転移としてのみ使用できます。マスクヘッドは未学習で始まり、そのままでは意味のないマスクを返すためです。CLIに`task=segment`を渡すことで転移を許可します。Python経由の手順にはさらに制限があります。`LibreYOLO()`ファクトリーは該当する引数を受け取らないため、`allow_detect_to_segment_transfer=True`を指定して`LibreDFINE`を直接構築する必要があります。また、直接構築ではダウンロードされないため、重みファイルがすでにディスク上になければなりません。

`lora=True`は検出に適用されます。セグメンテーション学習では拒否され、代わりに`freeze='backbone'`を案内します。マスクヘッドがアダプターでテストされていないためです。Apple siliconでは、トレーナーが実行全体をCPUに移します。Integralのビン分割行列乗算の逆伝播でMetalのコンパイルエラーが発生するためです。MPSでの推論には影響しません。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`はメトリクス名をキーとする辞書を返し、`verbose`を有効のままにするとクラス別の結果を表示します。

<code-tabs name="val" />

`-seg`チェックポイントを対象にすると、通常の`metrics/mAP50-95`キーにはマスクのスコアが格納されます。同じ実行で`(B)`のボックスと`(M)`のマスクも報告されるため、1回の処理で両方を取得できます。

## エクスポート

<export-matrix />

エクスポートした成果物は、ファイルサフィックスに基づいて`LibreYOLO()`から再度読み込めます。そのため、`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。OpenVINO、Paddle、MNN、Core AIへのエクスポートでは、動的形状ではなく固定キャンバスを使用します。各形式が受け付ける引数と、一部の形式に追加される引数については[エクスポート](/docs/export)を参照してください。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

セグメンテーション重みには2つ目のアップストリームがあります。マスクデコーダー、マスクマッチング、マスク損失は、同じくApache-2.0のArgoHA/D-FINE-segに由来し、そのメンテナーは帰属表示を伴う再利用を承認しています。

</provenance-box>

## 引用

<citation-block />

