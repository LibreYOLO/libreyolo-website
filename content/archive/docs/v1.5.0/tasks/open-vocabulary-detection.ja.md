---
title: オープンボキャブラリ検出
seo_title: LibreYOLOのオープンボキャブラリ検出
description: >-
  LibreYOLOでテキスト語彙から物体を検出します。Grounding
  DINO、OWLv2、OMDet-Turbo、OV-DEIMをLibreOpenVocabから読み込み、実行時にクラスを設定します。
lead: >-
  オープンボキャブラリ検出では、チェックポイントの固定クラスリストを、呼び出し時に選んだ単語へ置き換えます。LibreYOLOでは個別のタスクではありません。別のモデル階層が提供するdetectタスクであり、LibreYOLOではなくLibreOpenVocabファクトリーから読み込みます。
keywords:
  - オープンボキャブラリ検出
  - ゼロショット 物体検出
  - オープンセット検出
  - grounding dino python
  - owlv2
  - omdet turbo
  - テキストプロンプト 検出
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 語彙を入れ替える
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")

        # set_classesの設定は次に呼び出すまで維持される
        # 小文字化して冠詞を除いた後のラベルは一意でなければならない
        model.set_classes(["a red backpack", "traffic cone"])
        result = model.predict(SAMPLE_IMAGE)

        model.set_classes(["bicycle wheel"])
        result = model.predict(SAMPLE_IMAGE)
    - label: Grounding DINOのテキストしきい値
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # confはボックススコア、text_thresholdはデコードしたフレーズの
        # トークンスコアで絞り込む。未設定時はどちらもデフォルトで0.25
        # text_thresholdを受け付けるのはGrounding DINOだけで、ほかは例外を送出
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## 定義

オープンボキャブラリ検出は、通常の検出 `Results` を返します。ボックス、信頼度、クラスインデックスが含まれ、`result.names` により、それらのインデックスが要求した文字列へ対応付けられます。変わるのはクラスリストの取得元です。従来型の検出器は固定のカテゴリセットに対して学習され、その外側のカテゴリを出力することはできません。これらのモデルは推論時に語彙をテキストとして受け取るため、`set_classes(["forklift", "safety cone"])` を指定するだけで、それらをクラスにできます。

LibreYOLOに `open-vocabulary` タスクキーはありません。これらのモデルはほかの検出器と同様に `SUPPORTED_TASKS = ("detect",)` を宣言します。異なるのは読み込み経路です。LibreYOLOのstate-dictチェックポイントではなくHugging Faceのsnapshotであるため、`LibreYOLO()` ファクトリーには含まれず、代わりに `LibreOpenVocab()` で構築されます。このファクトリーは `LibreSAM()` および `LibreVLM()` の兄弟であり、`LibreYOLO()` の代替ではありません。

スコアは、後から解析した生成captionではなく、実際の検出スコアです。各ファミリーは、すべてのプロンプトのテキスト埋め込みベクトルに対して画像領域をスコアリングします。

## モデル

この階層は4つのファミリーで構成され、すべて推論専用です。いずれも別名を指定して `LibreOpenVocab` から読み込みます。

IDEA Researchの[Grounding DINO](/docs/models/grounding-dino)は `t` と `b` のサイズがあります。この階層のデフォルトであり、デコードしたフレーズのトークンスコアに対する第2のカットオフ `text_threshold` を受け付ける唯一のファミリーです。

Google Researchの[OWLv2](/docs/models/owlv2)は `b16` と `l14` のサイズがあります。CLIP形式のエンコーダーによるテキスト埋め込みベクトルに対して画像領域をスコアリングします。

Om AI Labの[OMDet-Turbo](/docs/models/omdet-turbo)は、1つの `t` サイズがあります。クラス埋め込みベクトルを言語タスクプロンプトから分離します。このページで唯一、独自の後処理内で重複ボックスを抑制するファミリーなので、`iou=` が反映されます。

[OV-DEIM](/docs/models/ov-deim)は `s`、`m`、`l` のサイズがあり、decoder queryを同梱のMobileCLIPテキストタワーによるテキスト埋め込みベクトルと照合するDETR形式の検出器です。1対1の照合とtop-K選択を使うため、どこにもNMSはありません。

この階層ではOV-DEIMの重みに制限があります。検出器の重みは非商用のCC BY-NC 4.0です。同梱のテキストタワーには研究利用専用のApple Machine Learning Research Modelライセンスが適用されます。`l` チェックポイントには、MetaのDINOv3 Licenseが適用されるDINOv3-Sバックボーンのファインチューニングも加わります。3つのライセンステキストはすべて重みのリポジトリに同梱され、ライブラリはモデルを構築する前に、重みを解決する際に同じ概要をログへ記録します。デプロイする前に[OV-DEIM](/docs/models/ov-deim)を読んでください。

この階層には1つの追加パッケージが必要です。

```bash
pip install "libreyolo[openvocab]"
```

これには、ラップされた3つのファミリー用の `transformers` と `timm`、およびネイティブ移植されたOV-DEIMに必要な `huggingface_hub`、`safetensors`、`regex`、`ftfy` パッケージが含まれます。

別の階層もテキスト語彙を受け取ります。`LibreVLM()` は[Qwen3-VL](/docs/models/qwen3-vl)や[Florence-2](/docs/models/florence-2)などの生成型視覚言語モデルを読み込み、その出力を同じ `Results` に変換します。`set_classes()` という同じインターフェースを共有します。異なるのはボックスを生成するものです。このページのファミリーはスコアを直接出力する識別的検出器ですが、VLM階層はボックスを生成します。

## 推論

<code-tabs name="predict" />

`set_classes()` は空でないラベル文字列のリストを受け取り、再度呼び出されるまで保持します。小文字化して先頭の冠詞を除いたラベルは一意でなければならないため、`"a bus"` と `"bus"` は1つの語彙内で共存できません。複数単語のフレーズもほかと同じラベルであり、各ファミリーはtokenizeの前にリストを独自のテキスト入力へ変換するため、`"traffic cone"` は `"cone"` とは異なるqueryです。

3つの推論引数はネイティブ検出器と異なる動作をします。これらのファミリーではprocessorがリサイズを担当するため、`imgsz=` は拒否されます。テスト時データ拡張はこの階層の範囲外なので、`augment=True` も拒否されます。`iou=` はprocessorが独自に抑制を実行するファミリーだけに適用されます。何も抑制されない場合は、指定すると警告され、無視されます。

`conf` を未設定のままにすると、通常の `predict()` の0.25ではなく、読み込んだファミリー独自のデフォルトが使われます。このデフォルトは階層内で統一されていません。同じ画像で2つのファミリーを比較するときは、明示的に設定してください。

この階層では `track()` から例外が送出されます。代わりにフレームごとに `predict()` を実行してください。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## 学習

この階層のどのファミリーもLibreYOLO内では学習できません。`train()` から例外が送出されます。アップストリームでファインチューニングし、得られた重みを読み込んでください。読み込んだモデルが検出するものを変更する設定は、`set_classes()` に渡す語彙だけです。

## 検証

この階層には検証機能がなく、`val()` から例外が送出されます。標準の検出検証機能は画像テンソルをモデルへ直接渡しますが、これらのファミリーでは並行して構築したテキスト条件付き入力が必要になるため、オープンボキャブラリ検証には専用の検証機能が必要です。

## エクスポート

エクスポートはこの階層の範囲外で、`export()` から例外が送出されます。これらのモデルはPyTorch上の `predict()` で実行します。

