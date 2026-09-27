---
title: DINO-DETR
families:
  - dinodetr
seo_title: 'DINO-DETR: Apache-2.0で推論とエクスポート'
description: >-
  LibreYOLOでDINO-DETRによる物体検出を実行します。Apache-2.0ライセンスのデノイジングアンカー3サイズについて、インストール、推論、検証、エクスポートを解説します。
lead: >-
  IDEA ResearchがDINOとして発表したDINO-DETRは、Deformable
  DETRのスパースアテンションに、対照的デノイジング学習と混合クエリ選択を組み合わせています。LibreYOLOでは、推論専用の検出モデルを3サイズ提供します。
keywords:
  - DINO-DETR
  - DINO
  - 検出Transformer
  - デノイジングアンカーボックス
  - 混合クエリ選択
  - 物体検出
  - IDEA Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val()はオブジェクトではなく通常のdictを返す
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: エクスポートしたファイルを使用
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリはファイルの拡張子に応じて振り分けるためエクスポートした
        # アーティファクトもチェックポイントと同様に読み込まれ同じResultsオブジェクトを返す
        model = LibreYOLO("LibreDINODETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## インストール

DINO-DETRに追加オプションは必要ありません。インポートするものはすべて基本インストールに含まれており、LibreYOLOのDeformable DETRファミリーと同じ純粋なPyTorchによるマルチスケール変形可能アテンションのコアを使用します。

```bash
pip install libreyolo
```

`libreyolo[hub-kernels]`のインストールは任意です。`kernels`パッケージが存在すると、LibreYOLOは実行時にHugging Face Hubからコンパイル済みのマルチスケール変形可能アテンションカーネルを取得し、純粋なPyTorchのコアの代わりに使用します。`LIBREYOLO_HUB_KERNELS=0`を設定すると無効に戻せます。

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

返される`Results`オブジェクトは全ファミリーで共通なので、別の検出器への切り替えは1行の変更で済みます。`conf`と`max_det`はクエリ選択を絞り込みます。`iou`はAPI互換性のため受け付けますが効果はありません。デコーダーがNMS処理を行わない集合予測器だからです。入力ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

LibreYOLOのDINO-DETRは推論専用です。アップストリームでは対照的デノイジングとハンガリアンマッチングを用いて学習しますが、その手法はここでは実装されていないため、`train()`は`NotImplementedError`を送出します。

## バリアント

3つのチェックポイントはすべて同じ入力解像度です。`r50`と`r50s5`はResNet-50バックボーンを共有し、デコーダーに渡す特徴マップのスケール数が4つと5つで異なります。`swinl`はバックボーンをSwin-Lに置き換え、同じく5つのスケールを使用します。

## 検証

`val()`は、学習に使用した形式の任意のデータセットに対して測定した適合率、再現率、mAP 50、mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートしたアーティファクトはファイルの拡張子に基づいて`LibreYOLO()`で読み込めるため、`.onnx`や`.engine`ファイルもチェックポイントと同様に動作し、同じ`Results`を返します。[エクスポート](/docs/export)には各形式で受け付ける引数が記載されています。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

3つの公式チェックポイントはHugging Faceのモデルカードではなく、著者のGoogle Driveリリースフォルダーから取得されています。アップストリームのリポジトリはリポジトリ単位でApache-2.0を宣言していますが、チェックポイント自体にライセンスファイルやライセンスのメタデータを添付していません。そのため、再配布の根拠はチェックポイント固有の許諾ではなく、リポジトリ単位の宣言です。LibreYOLOの各ミラーには、この経緯を説明する告知とともに、アップストリームのApache-2.0ライセンステキストが変更なしで同梱されています。

</provenance-box>

## 引用

<citation-block />
