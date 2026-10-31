---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: LibreYOLOのOMDet-Turbo：リアルタイムのゼロショット検出
description: >-
  LibreYOLOのOMDet-Turboでリアルタイムのオープンボキャブラリ検出を行います。openvocab追加パッケージをインストールし、自由なテキスト語彙で推論します。
lead: >-
  OMDet-TurboはOm AI
  Labが開発したリアルタイムのオープンボキャブラリ物体検出器で、クラス埋め込みベクトルを言語タスクプロンプトから分離します。LibreYOLOはオープンボキャブラリ検出器層の推論専用ファミリーとしてラップします。
keywords:
  - OMDet-Turbo 使い方
  - OmDet
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

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 独自のNMSしきい値
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["traffic light", "bicycle"])

        # OMDet-Turboはこの層でiou=を反映する唯一のファミリー
        # 独自の後処理が抑制のしきい値を引数として受け取り
        # iou=未設定時のデフォルトは0.5
        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)
        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## インストール

OMDet-TurboはLibreYOLOのオープンボキャブラリ検出器層から読み込まれます。この層には
`openvocab`追加パッケージが必要です。

```bash
pip install "libreyolo[openvocab]"
```

この追加パッケージでは、この層が呼び出すHugging Faceライブラリの`transformers`と`timm`が
インストールされます。OMDet-TurboのSwinバックボーンは`transformers`の`TimmBackbone`
ラッパーから読み込まれます。

## 推論

OMDet-TurboはLibreYOLOが`LibreYOLO()`から読み込むチェックポイントではありません。同系列の
`LibreOpenVocab`ファクトリから読み込まれ、初回使用時にHugging Faceのスナップショットを
ダウンロードして`weights/`へキャッシュします。

<code-tabs name="predict" />

`set_classes()`は維持されるテキスト語彙を設定します。再度呼び出すとリスト全体を置き換え、
省略するとデフォルトのCOCO-80ラベルを維持します。空の結果もエラーではなく有効な結果です。
Grounding DINOと異なり、OMDet-Turboはクラス埋め込みベクトルを言語タスクプロンプトから分離します。
そのため、`transformers`の後処理が返すラベルは、フレーズの曖昧性を解消する処理なしで、問い合わせた
クラスリストへ直接対応します。

OMDet-Turboにはtext-tokenしきい値がなく、検出を絞り込むのは`conf`だけです。
`text_threshold`を渡すと例外が発生します。この層で
`post_process_grounded_object_detection`内の独自non-maximum suppressionを実行する
唯一のファミリーであるため、`iou`は警告の対象にならず反映されます。`imgsz`と`augment=True`は
完全に拒否されます。`transformers`のprocessorがリサイズを担当し、test-time augmentationは
この層の対象外であるためです。1枚の画像に対する`predict()`はリストではなく1個の`Results`を
返します。複数の結果を得るには、ディレクトリ、画像のリスト、または動画ソースと`stream=True`を
渡します。このファミリーにCLI経路はありません。`libreyolo predict`は`LibreYOLO()`から`.pt`
チェックポイントだけを読み込むため、`LibreOpenVocab`ファミリーはPythonから実行します。
入力ソースの種類とストリーミングについては[推論](/docs/predict)を参照してください。

## バリアント

チェックポイントはこの層で唯一のサイズ`t`の1種類です。固定されたアップストリームrevisionの
`omlab/omdet-turbo-swin-tiny-hf`を、`transformers`の`OmDetTurboForObjectDetection`を
通じてミラーします。ミラーした重みファイルは、そのアップストリームスナップショットと
バイト単位で同一です。このファミリーでは精度またはレイテンシの数値がまだ公開されていません。

学習、データセット検証、エクスポートはすべてこの層の対象外です。`train()`、`val()`、
`export()`は無条件に`NotImplementedError`を発生させます。これは公開済みチェックポイントの
推論専用ラッパーです。

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

