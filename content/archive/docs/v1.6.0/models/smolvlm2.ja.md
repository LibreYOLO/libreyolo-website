---
title: SmolVLM2
families:
  - smolvlm2
seo_title: LibreYOLOのSmolVLM2：オープンボキャブラリ検出
description: >-
  LibreYOLOでSmolVLM2を使用します。インストール、オープンボキャブラリの設定、Hugging
  FaceによるApache-2.0ライセンスの視覚言語モデルを使った推論またはチャットについて説明します。
lead: >-
  SmolVLM2はHugging
  Faceの小型視覚言語モデルです。LibreYOLOはこれをオープンボキャブラリ物体検出器としてラップし、自由形式のチャットを直接公開します。クラス一覧を指定して検出するか、質問してください。
keywords:
  - SmolVLM2
  - 視覚言語モデル
  - オープンボキャブラリ検出
  - 小型 マルチモーダルモデル
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: チャット
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # 検出用の簡易機能の下にある脱出口 任意の質問に対応し
        # バウンディングボックスの質問だけに限定されない
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
source_hash: b30823b62d6347b5
---

## インストール

SmolVLM2はLibreYOLOの検出器として使うVLM階層に属します。これは独自のファクトリーを持ち、チェックポイントベースのファミリーとは別の製品面です。`vlm` 追加パッケージが必要であり、SmolVLM2独自のプロセッサーの依存関係である `num2words` も導入されます。

```bash
pip install "libreyolo[vlm]"
```

## 推論

重みは初回使用時にHugging Faceからダウンロードされ、ローカルにキャッシュされます。

<code-tabs name="predict" />

このファミリーは `LibreYOLO()` ではなく、`LibreVLM()` ファクトリーから読み込みます。VLMファミリーはチェックポイントローダーを宣言しないため、他のモデルページで説明しているファイル接尾辞による振り分けはここでは適用されません。`set_classes()` はSmolVLM2に探させるボキャブラリを設定します。この設定は維持されるため、再設定するまで、その後のすべての `predict()` または `track()` 呼び出しに適用されます。SmolVLM2では、LibreYOLO内のパーサー上書きは不要です。この階層で共有されるデフォルトと同じ、チャットテンプレートとJSON出力の組み合わせに従うため、検出プロンプトとボックス形式はファミリー固有ではありません。すべての検出結果に同じ仮信頼度が設定されるため、`conf` によるフィルタリングは順位付けではなく、すべてを残すかすべてを除外するかのどちらかです。`iou` は効果があり、後から得られた同じクラスのボックスが、すでに保持したボックスとしきい値を超えて重なると除外されます。これを行わないと、反復生成器が1つの物体に対してほぼ重複したボックスを出力する場合があるためです。SmolVLM2は `chat()` を通じて自由形式の質問にも回答します。これは `LibreVLM` ファクトリーで説明しているものと同じ脱出口です。LibreYOLOのCLIはこの階層に対応しておらず、`libreyolo predict model=...` 形式はありません。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

レジストリ内のサイズはSmolVLM2-500M-Video-Instructの1つで、`LibreVLM("smolvlm2-500m")` で読み込みます。SmolVLM2は、この階層でグラウンディング専用に設計されたモデルより検出性能が低いモデルです。LibreYOLO独自のラッパーでは、最も強力なオープンボキャブラリ選択肢としてではなく、新しいファミリーが特別な解析処理なしでここで動作できることを示す実例として説明されています。

LibreYOLOはSmolVLM2の学習、検証、エクスポートを行いません。この階層のすべてのファミリーで、`train()`、`val()`、`export()` は `NotImplementedError` を発生させます（上のサポート階層を参照）。組み込みのカスタムボキャブラリが必要な場合は、アップストリームでSmolVLM2をファインチューニングし、得られた重みを読み込んでください。すべての検出結果が同じ仮信頼度を持つため、COCO形式の検証を行う代わりに、`predict()` の出力を目視で確認してください。

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />
