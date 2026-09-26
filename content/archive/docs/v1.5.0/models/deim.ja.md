---
title: DEIM
families:
  - deim
seo_title: LibreYOLOのDEIMとDEIMv2
description: >-
  LibreYOLOでDEIMとDEIMv2を使って物体検出を行います。50万パラメータのサイズから利用でき、インストール、推論、学習、検証、エクスポートに対応します。
lead: >-
  密な1対1マッチングで学習し、基盤とするDETRレシピよりはるかに少ないエポックで収束する検出Transformerです。LibreYOLOには2つのバージョンがあり、読み込むチェックポイントで区別されます。
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - detection transformer
  - DETR
  - 物体検出
  - リアルタイム 物体検出
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        # バージョンはファイル名の一部でファクトリーはチェックポイントに基づき
        # 振り分けるため両方とも同じ方法で読み込む
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # ライブラリが受け付ける任意のソース: ファイル フォルダー URL Webカメラ番号
        # RTSP ストリーム または .streams リスト
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml は初回使用時に128枚のサンプルをダウンロード
        # 実際の実行では data に独自データセットのYAMLを指定
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 未設定の場合 epochs batch imgsz lr0 は読み込んだサイズ向けの
        # 公開レシピから取得
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # lora 追加パッケージが必要: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() はオブジェクトではなく通常の辞書を返す
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: COCOで検証
      language: bash
      code: |
        # coco-val-only.yaml は5000枚の val2017 画像を取得し
        # 学習セットをスキップする ダウンロードスクリプトを内包するため
        # データセットがローカルにない場合は明示的な許可が必要
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # onnx 追加パッケージが必要: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイルサフィックスで振り分けるためエクスポート成果物も
        # 任意のチェックポイントと同様に読み込まれ同じ Results オブジェクトを返す
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## インストール

どちらのバージョンにもオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれています。

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

返される`Results`オブジェクトはすべてのファミリーに共通するため、別の検出器への切り替えは1行の変更で済みます。`conf`と`max_det`はクエリとクラスに対する上位k件のデコードをフィルタリングします。調整するNMSステップはなく、`iou`は受け付けられますが使用されません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

バージョン1には同じ入力サイズを使う5つのサイズがあります。バージョン2はその5つの名前を維持し、さらに小さい`atto`、`femto`、`pico`を追加します。最初の2つは、ほかのサイズより小さい入力サイズをネイティブで使用します。したがって、5つのサイズコードは両バージョンに存在し、それぞれ異なるモデルを指します。バージョンはチェックポイントのファイル名に記載されます。

<benchmark-table task="detect" />

<va-embed />

バージョン1はD-FINEのアーキテクチャを維持し、その分類目的関数を密な1対1レシピのマッチ可能性を考慮した損失に置き換えます。そのため、2つのファミリーはほぼすべてのstate dictキーを共有し、チェックポイント内のメタデータで区別されます。バージョン2はその学習規約を維持しつつ、バックボーンを組み合わせます。`s`未満ではHGNetv2、`s`以上では空間チューニングアダプターを備えたDINOv3 Vision Transformerを使います。このバックボーンにより、その4つのチェックポイントには2つ目のライセンスが適用されます。リリースする前に[ライセンス](#licensing)を確認してください。

## 学習

学習は公開済みチェックポイントから開始します。`pretrained`がトレーナーに渡ることはありません。バージョン1ではキーが不明だと警告して無視し、バージョン2では削除します。どちらもランダムに初期化されたモデルは提供しません。

<code-tabs name="train" />

バージョン1では`lr0`を自分で渡してください。Pythonの`train()`シグネチャのデフォルトは、公開済みCOCOレシピの学習率`4e-4`です。一方、ファミリーの学習設定にはファインチューニング用のデフォルトとして`1e-4`が含まれ、引数を省略した場合にCLIが解決するのはこの低い値です。設定にはその根拠となる測定結果も記録されています。ファインチューニングで実際に使うバッチサイズと小規模データセットでは、COCOの学習率によって転移性能が測定可能なほど低下しました。

バージョン2はこれらのデフォルトを自ら解決します。`epochs`、`batch`、`imgsz`、`lr0`を未設定にすると、読み込んだサイズ向けの公開レシピから各値を読み取ります。そのため、小さいサイズは指定なしでも固有の入力解像度で学習し、渡した値はレシピを上書きします。制約が適用される引数は`imgsz`です。正の32の倍数でなければならず、それ以外の場合はバージョン2が実行開始前に例外を発生させます。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は、学習に使用した形式の任意のデータセットで測定した適合率、再現率、mAP 50、mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

上記のベンチマーク表の行はLibreYOLOのベンチマークハーネスから取得されています。表の下の注記には、使用したデータセットと実行記録へのリンクが記載されています。

## エクスポート

<export-matrix />

このマトリクスは2つのバージョンを1ページで扱います。形式に対する対応状況が異なる場合、そのセルには弱い方が表示されるため、どちらのバージョンを読み込んでも実際以上の対応を示すことはありません。

エクスポートした成果物は、ファイルサフィックスに基づいて`LibreYOLO()`から再度読み込めます。そのため、`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>
S以上の4つのDEIMv2サイズはDINOv3のバックボーンを使用します。そのため、重みのリポジトリにはApache-2.0とMetaのDINOv3 Licenseの両方が適用され、LibreYOLOも同じ契約に基づいてDINOv3のバックボーンソースを配布します。このファミリーの残りすべて（S未満のすべてのDEIMv2サイズを含む）には、Apache-2.0のみが適用されます。
</provenance-box>

## 引用

<citation-block />

DEIMv2には別の論文があり、[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation)に専用の引用ブロックがあります。バージョン2のチェックポイントを使用した場合は、そちらを引用してください。

