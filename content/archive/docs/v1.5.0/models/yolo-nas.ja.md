---
title: YOLO-NAS
families:
  - yolonas
seo_title: YOLO-NAS：LibreYOLOで推論、学習、エクスポート
description: >-
  LibreYOLOでYOLO-NASを使い、検出と姿勢推定を行います。Deci.AIの重みはプロプライエタリで非商用に限定され、LibreYOLOはその重みを一切公開していません。
lead: >-
  Deci.AIのアーキテクチャ探索から生まれたバックボーンとネックを持ち、量子化を考慮したRepVGGブロックで構築された畳み込み検出器です。重みはDeci.AIのもので、非商用利用だけが許可されており、LibreYOLOはその重みを一切公開していません。
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - 物体検出
  - 姿勢推定
  - 量子化対応 物体検出
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ディスク上にない名前は Deci の CDN から取得
        # ダウンロード前に Deci のライセンス条件を表示し取得すると同意したことになる
        model = LibreYOLO("LibreYOLONASs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 姿勢推定
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # -pose サフィックスで姿勢推定ヘッドと固有の重み集合を選択
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: ゼロから学習
      language: python
      code: |
        from libreyolo import LibreYOLONAS

        # Deci のチェックポイントには触れずモデルをランダムな重みから開始
        # 実行結果は独自データだけに由来
        model = LibreYOLONAS(None, size="s")
        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: COCOで検証
      language: bash
      code: |
        # 同梱の COCO yaml はダウンロードスクリプトを内包するため
        # データセットがローカルにない場合は明示的な許可が必要
        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイルサフィックスで振り分けるためエクスポート成果物も
        # 任意のチェックポイントと同様に読み込まれ同じ Results オブジェクトを返す
        model = LibreYOLO("LibreYOLONASs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## インストール

YOLO-NASには基本パッケージ以外の追加パッケージは不要です。

```bash
pip install libreyolo
```

## 推論

ディスク上に存在しないチェックポイント名は、これらの重みを一切ホストしていないLibreYOLO組織ではなく、Deciの公開CDNから取得されます。転送を開始する前に、ライブラリはプロセスごとに1回Deciのライセンス条件を表示します。ダウンロードしたファイルを開く前に、SHA-256が固定値と照合されます。条件で許可される内容については[ライセンス](#licensing)を参照してください。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーに共通するため、別の検出器への切り替えは1行の変更で済みます。`conf`は信頼度のしきい値、`iou`はNMSのしきい値を設定します。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

検出と姿勢推定は異なるヘッドの下で同じアーキテクチャを使い、同じ引数を受け取ります。下の表にあるサイズは検出用です。姿勢推定はその各サイズと、さらに小さい1サイズで公開されています。姿勢推定ヘッドはCOCOのキーポイント集合を予測します。

<benchmark-table task="detect" />

<va-embed />

## 学習

<code-tabs name="train" />

`epochs`、`lr0`、`amp`を省略するとタスクごとに解決されるため、姿勢推定の実行は検出とは異なるデフォルトから始まります。オプティマイザーのデフォルトはAdamWです。クラス数はデータセットYAMLから取得され、最初のエポック前にヘッドがその数に合わせて再構築されます。姿勢推定ヘッドではキーポイント数も同じように処理されるため、COCOの姿勢推定チェックポイントを異なるサイズのスケルトンへファインチューニングできます。

ファインチューニングはDeciの重みから開始し、これがDeciのライセンスの対象です。ランダムに初期化したモデルからの学習ではDeciのチェックポイントを一切使用しません。その手順が上の3番目のスニペットです。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は、学習に使用した形式の任意のデータセットで測定した適合率、再現率、mAP 50、mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートした成果物は、ファイルサフィックスに基づいて`LibreYOLO()`から再度読み込めます。そのため、`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。LibreYOLOをインストールせず、単独のランタイムでグラフを実行することもサポートされますが、その場合は前処理と後処理を自分で実装する必要があります。各形式では異なる追加パッケージをインストールし、それぞれ固有の引数をいくつか受け取ります。どちらも該当する形式のページに記載されています。

エクスポートは、同じ重みを別のコンテナに収めたもう1つのコピーです。Deciのチェックポイントをエクスポートしても、重みの由来や適用されるライセンスは変わりません。

<code-tabs name="export" />

## チェックポイント

一覧にするものはありません。Deciのライセンスは再配布を禁止しているため、LibreYOLO組織はYOLO-NASの重みを一切公開せず、ダウンロードは別の場所から解決されます。`LibreYOLONAS<size>.pt`形式の名前、姿勢推定の場合は`LibreYOLONAS<size>-pose.pt`形式の名前が、Deciの公開CDN上の対応するオブジェクトにマッピングされます。

ライブラリがSHA-256を固定しているチェックポイントだけをこの方法で取得できます。それ以外は、検証されていない第三者のpickleを開かずに失敗し、手動でダウンロードしてパスとして渡す必要があります。すでにディスク上にあるファイルはパスから読み込まれ、ダウンロードもチェックサムゲートもありません。ローダーが認識する元の名前のDeci `.pth`もこれに含まれます。

## ライセンス

<provenance-box>

LibreYOLOはこれらの重みをホストもミラーもしません。このファミリーに該当するファイルはLibreYOLOのHugging Face組織に一切ありません。すべての自動ダウンロードは代わりにDeciの公開CDNへ接続し、開始前にプロセスごとに1回Deciの条件を表示します。また、ファイルを開く前に固定されたSHA-256と照合します。

ランダムに初期化されたモデルからの学習が代替手段です。アーキテクチャはアップストリームでApache-2.0、ここではMITなので、この方法で独自データを使って学習したモデルはDeciのチェックポイントに由来しません。

</provenance-box>

## 引用

YOLO-NASは論文なしで公開されました。下の項目は、YOLO-NASが提供されたライブラリであるSuperGradientsを対象として、著者が引用を求めているものです。

<citation-block />

