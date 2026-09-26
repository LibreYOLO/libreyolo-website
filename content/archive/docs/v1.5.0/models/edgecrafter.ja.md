---
title: EdgeCrafter
families:
  - ec
seo_title: EdgeCrafter：LibreYOLOで検出、姿勢推定、セグメンテーション
description: >-
  LibreYOLOでEdgeCrafterを使い、検出、姿勢推定、インスタンスセグメンテーションを行います。MITライセンスのコードでインストール、推論、検証、エクスポートに対応します。
lead: >-
  エッジハードウェア上の密な予測に適した小型Vision
  Transformerです。アップストリームではECDet、ECPose、ECSegという3つの姉妹モデルとして公開されています。LibreYOLOは3つすべてを1つのファミリーとして読み込み、チェックポイントによってタスクを決定します。
keywords:
  - EdgeCrafter
  - ECDet
  - ECPose
  - ECSeg
  - 小型 vision transformer
  - 物体検出
  - 姿勢推定
  - インスタンスセグメンテーション
  - エッジ 推論
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 姿勢推定
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファイル名の -pose サフィックスでキーポイントヘッドを選択するため
        # task 引数は不要
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: インスタンスセグメンテーション
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: 姿勢推定
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml で kpt_shape を宣言した単一クラスのキーポイントデータセットと
        # チェックポイントのネイティブサイズに合わせた imgsz が必要
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: インスタンスセグメンテーション
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ポリゴンラベルとチェックポイントのネイティブサイズに合わせた imgsz が必要
        model = LibreYOLO("LibreECs-seg.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: 姿勢推定
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: インスタンスセグメンテーション
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # マスク
        print(metrics["metrics/mAP50-95(B)"])   # ボックス
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイルサフィックスで振り分けるためエクスポート成果物も
        # 任意のチェックポイントと同様に読み込まれ同じ Results オブジェクトを返す
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## インストール

EdgeCrafterにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれています。

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

タスクはファイル名で決まるため、`-pose`または`-seg`チェックポイントは固有のヘッドを選択し、タスク引数を必要としません。3つすべてが各ファミリー共通の`Results`オブジェクトを返し、姿勢推定では`result.keypoints`、セグメンテーションでは`result.masks`が追加されます。姿勢推定はpersonの1クラスとCOCOの17キーポイントを対象とし、モデル構築時にその数が固定されます。ボックスヘッドを持たないため、各姿勢ボックスは自身のキーポイントを囲む範囲となり、キーポイントの3番目のチャンネルは点ごとのスコアではなく定数です。

`conf`と`max_det`はクエリ選択をフィルタリングします。3つのヘッドはすべてNMSステップなしでクエリ集合をデコードするため、`iou`はAPIの互換性のため受け付けられますが効果はありません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

4つのサイズがあります。すべて同じ入力解像度で動作するため、表ではパラメータ数と精度で区別しています。

<benchmark-table task="detect" />

<va-embed />

アップストリームは、3つのヘッドを持つ1つのモデルではなく、ECDet、ECPose、ECSegを3つの別個のモデルとして公開しています。これらはECViTバックボーンとハイブリッドエンコーダーを共有し、ヘッドだけが異なります。そのため、LibreYOLOは3つを1つのファミリーにまとめ、チェックポイントのファイル名でタスクを表します。したがって、サイズ文字は3つすべてで同じバックボーンとエンコーダーを意味し、どれを読み込んでも推論、検証、エクスポートは同じ引数を受け取ります。

## 学習

3つのタスクはすべて`train()`で学習します。読み込んだチェックポイントからタスクを読み取り、対応するトレーナーを選択します。

<code-tabs name="train" />

検出とセグメンテーションで確認済みなのは、サイズごとに層単位で行ったアップストリームとの`1e-5`以内の推論一致と、合成入力で損失および1回の学習ステップが動作することです。`train()`自体のdocstringによると、完全なファインチューニングの収束、マルチGPU学習、データ拡張停止後の最良モデル再読み込みステップ、Objects365からCOCOへのクラス再マッピングは未検証です。姿勢推定の処理はDETRPoseの公開レシピに従います。クラス、キーポイントのL1、OKSコストを使うHungarian matcherと、対照的キーポイントノイズ除去を採用していますが、そのエンドツーエンドの収束も未検証です。

設定を変更しなければ、トレーナーはアップストリームのレシピに従い、混合精度を有効にして`lr0=5e-4`で74エポック実行します。AdamW、フラットコサインスケジュール、0.9999のEMA、ImageNetで正規化した入力を使います。姿勢推定とセグメンテーションでは、どちらもチェックポイントのネイティブサイズに合わせた`imgsz`が必要です。評価用アンカーグリッドはモデルの構築時に作成されるため、異なる値を指定すると実行開始前に例外が発生します。姿勢推定には、`data.yaml`で`kpt_shape`を宣言し、キーポイント数がヘッドと一致する単一クラスのデータセットも必要です。

`lora=True`は検出だけに適用されます。姿勢推定とセグメンテーションでは`ValueError`が発生します。Apple siliconでは実行をGPU上に維持しつつ、変形可能アテンション内部のgrid-sampleの逆伝播という1つの演算をCPUに送ります。PyTorchがMetal上でこの演算を実装していないためです。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`はメトリクス名をキーとする辞書を返し、`verbose`を有効のままにするとクラス別の結果を表示します。

<code-tabs name="val" />

姿勢推定は`metrics/keypoints_*`にキーポイントのOKSメトリクスを報告します。セグメンテーションは通常の`metrics/mAP50-95`キーにマスクを報告し、1回の処理で両方のビューも繰り返します。ボックスは`(B)`、マスクは`(M)`に格納されます。

## エクスポート

<export-matrix />

エクスポートした成果物は、ファイルサフィックスに基づいて`LibreYOLO()`から再度読み込めます。そのため、`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。姿勢推定とセグメンテーションは、動的形状ではなく固定の640 x 640入力でエクスポートされます。OpenVINO、Paddle、MNN、ExecuTorch、Core AIを含む複数の検出ターゲットも固定キャンバスです。各形式が受け付ける引数と、一部の形式に追加される引数については[エクスポート](/docs/export)を参照してください。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

