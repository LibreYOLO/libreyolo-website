---
title: CLIP
families:
  - clip
seo_title: LibreYOLOのCLIP：ゼロショット分類と埋め込み
description: >-
  LibreYOLOのCLIPでゼロショット画像分類と画像・テキスト埋め込みを行います。学習は不要です。set_classes()が実行時にラベルセットを定義します。
lead: >-
  CLIPは固定ラベルセットの代わりに、テキストプロンプトに対して画像を評価するdual-towerモデルです。LibreYOLOでは学習処理なしで、ゼロショット分類と画像・テキスト埋め込みに対応します。
keywords:
  - CLIP 使い方
  - OpenCLIP
  - ゼロショット分類
  - 画像 エンベディング
  - テキスト エンベディング
  - オープンボキャブラリ
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # set_classes()を呼び出さない場合、CLI predictはモデルがデフォルトで

        # 読み込む1,000個のImageNetクラス名を使用

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 画像とテキストの埋め込み
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE).embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # 両方ともL2正規化済みのため、通常の内積がコサイン類似度
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # dataはtrain/分割を持つImageFolderルート。そのフォルダー名が
        # この実行のゼロショットクラスプロンプトになる
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # 現在のset_classes()ラベルと入力解像度をグラフへ組み込み
        # どちらかを変更したら再エクスポート
    - label: CLI
      language: bash
      code: |
        # ここではset_classes()を呼び出さないため、モデルが読み込むデフォルトの
        # 1,000個のImageNetクラスを組み込み
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: 埋め込みをエクスポート
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed"は画像towerだけをトレース。クラスは不要
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## インストール

CLIPには専用の追加パッケージが必要です。同梱されたBPE tokenizerが正確なtoken IDを再現するために使うパッケージがインストールされます。

```bash
pip install "libreyolo[clip]"
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

`set_classes()`は、このモデルをオープンボキャブラリ分類器にする基本操作です。各ラベルをすべてのプロンプトテンプレートへ展開し、結果をエンコードして平均します。生成した`[K, D]`行列を分類器ヘッドとしてキャッシュするため、画像ごとに再計算しません。クラスはいつでも再度呼び出して変更できます。呼び出さない場合、LibreCLIPは1,000個のImageNet-1kクラス名を設定済みの状態で読み込まれます。

`task="embed"`では、推論はクラス確率の代わりに入力ごとに1個のL2正規化済み画像ベクトルを返し、`embed_text()`は同じベクトル空間の正規化済みテキスト行を返します。そのため、両者の通常の内積がコサイン類似度になります。どちらのタスクでも`iou`に効果はなく、NMS処理もありません。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## 検証

`val()`はImageFolderの`train/`分割にあるクラスフォルダー名を読み取り、それらを指定して`set_classes()`を呼び出した後、ゼロショットのtop-1精度とtop-5精度を測定します。学習対象がないため、精度は重みの更新ではなく、クラス名がプロンプトとしてどのように解釈されるかに依存します。検証の対象は`task="classify"`だけです。`task="embed"`にはデータセットvalidatorがありません。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートではモデルの現在の状態を固定グラフへ組み込みます。`task="classify"`では、`set_classes()`が最後に設定したラベルとエクスポート時の解像度を最終linear層へ組み込みます。そのため、エクスポートしたONNXまたはTensorRTグラフは、テキストtowerもtokenizerもない通常の`[B, K]`画像分類器です。クラスまたはサイズを変更したら再エクスポートしてください。`task="embed"`のエクスポートは画像towerだけをトレースします。どちらもONNX opset 14以降が必要で、エクスポーターがデフォルトで設定します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。どちらもCOCOで学習したものではなく、OpenCLIPのLAION-2B学習済みチェックポイント（`ViT-B-32`と`ViT-B-16`）から変換されています。

<checkpoint-table />

LAION-2Bの学習データには、CSAMコンテンツが含まれていたことが記録されています（Stanford Internet Observatory、2023年12月）。その後LAIONは、クリーンアップした再公開版のRe-LAIONを公開しました。これらの重みを別の場所で再配布する場合、利用可能ならRe-LAION由来のチェックポイントを優先してください。

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

