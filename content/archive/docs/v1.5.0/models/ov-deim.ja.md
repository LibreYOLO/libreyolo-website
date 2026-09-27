---
title: OV-DEIM
families:
  - ov_deim
seo_title: LibreYOLOのOV-DEIM：オープンボキャブラリ検出
description: >-
  LibreYOLOのOV-DEIMでリアルタイムのDETR系オープンボキャブラリ検出を行います。openvocab追加パッケージをインストールし、自由なテキスト語彙で推論します。
lead: >-
  OV-DEIMは、デコーダーqueryを同梱されたMobileCLIP text
  towerのテキスト埋め込みベクトルと照合するDETR系オープンボキャブラリ物体検出器です。LibreYOLOはオープンボキャブラリ検出器層の推論専用ファミリーとしてネイティブに移植しています。
keywords:
  - OV-DEIM 使い方
  - DEIMv2
  - オープンボキャブラリ 物体検出
  - リアルタイム 物体検出
  - ゼロショット検出
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 語彙を置き換える
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # set_classes()を再度呼び出すと語彙全体を置き換え、text towerで
        # 再度埋め込み。空の結果もエラーではなく有効な結果
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## インストール

OV-DEIMはLibreYOLOのオープンボキャブラリ検出器層から読み込まれます。この層には
`openvocab`追加パッケージが必要です。

```bash
pip install "libreyolo[openvocab]"
```

この層のほかのモデルと異なり、OV-DEIMは`transformers`ラッパーではなくLibreYOLOへの
ネイティブ移植版で、対応する`transformers`モデルクラスはありません。ただし、同じ追加パッケージで
推論時に必要な`huggingface_hub`、`safetensors`、`regex`、`ftfy`がインストールされます。

## 推論

OV-DEIMはLibreYOLOが`LibreYOLO()`から読み込むチェックポイントではありません。同系列の
`LibreOpenVocab`ファクトリから読み込まれ、初回使用時にHugging Faceのスナップショットを
ダウンロードして`weights/`へキャッシュします。

<code-tabs name="predict" />

`set_classes()`は維持されるテキスト語彙を設定します。再度呼び出すとリスト全体を置き換え、
省略するとデフォルトのCOCO-80ラベルを維持します。空の結果もエラーではなく有効な結果です。
各デコーダーqueryは、同梱されたMobileCLIP-B(LT) text towerのテキスト埋め込みベクトルとの
コサイン類似度で評価されます。設定された語彙に対してオンラインで計算され、変更されるまで
キャッシュされるため、事前計算済みの埋め込みファイルなしで任意のプロンプトを使えます。

OV-DEIMにはtext-tokenしきい値がなく、検出を絞り込むのは`conf`だけです。
`text_threshold`を渡すと例外が発生します。照合は1対1のtop-K選択のため、non-maximum suppressionは
実行されません。`iou`はAPI互換性のため受け付けますが、警告を表示して何もしません。`imgsz`と
`augment=True`は完全に拒否されます。モデルが固定letterbox入力を持ち、test-time augmentationは
この層の対象外であるためです。1枚の画像に対する`predict()`はリストではなく1個の`Results`を
返します。複数の結果を得るには、ディレクトリ、画像のリスト、または動画ソースと`stream=True`を
渡します。このファミリーにCLI経路はありません。`libreyolo predict`は`LibreYOLO()`から`.pt`
チェックポイントだけを読み込むため、`LibreOpenVocab`ファミリーはPythonから実行します。
入力ソースの種類とストリーミングについては[推論](/docs/predict)を参照してください。

`predict()`を呼び出すたびに、現在の語彙を埋め込むため、同梱されたMobileCLIP-B(LT) text towerも
実行されます。それによって追加される条件については「ライセンス」を参照してください。

## バリアント

チェックポイントは`s`、`m`、`l`の3種類です。サイズを指定しない場合、この層のデフォルトは`s`です。
この層のほかのモデルと異なり、OV-DEIMは`transformers`ラッパーではなくネイティブ移植版です。
LibreYOLOはアップストリームコードと同じApache-2.0ライセンスで検出器モジュールを同梱し、DEIMv2
ファミリー向けに構築済みのDINOv3バックボーンアダプターを再利用します。`l`チェックポイントの
バックボーンはDINOv3-Sのファインチューニング版で、MetaのDINOv3 Licenseに基づいて別途
ライセンスされます。このファミリーでは精度またはレイテンシの数値がまだ公開されていません。

学習、データセット検証、エクスポートはすべてこの層の対象外です。`train()`、`val()`、
`export()`は無条件に`NotImplementedError`を発生させます。これは公開済みチェックポイントの
推論専用ラッパーです。

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

OV-DEIMでは、すべての推論呼び出しに3種類のアップストリームライセンスが重なります。
検出器の重みにはOV-DEIM独自のCC BY-NC 4.0、オンラインtext towerにはAppleのMachine Learning
Research Model license（研究用途のみ）、`l`チェックポイントのDINOv3-Sファインチューニング
バックボーンにはMetaのDINOv3 Licenseが適用されます。3種類すべてのライセンステキストが
LibreYOLOの重みリポジトリ内に同梱されます。

</provenance-box>

## 引用

<citation-block />

