---
title: InternVL3
families:
  - internvl3
seo_title: InternVL3：LibreYOLOでのオープンボキャブラリ検出
description: >-
  LibreYOLOのInternVL3でオープンボキャブラリ物体検出を行います。任意のテキストラベルで推論できますが、学習、検証、エクスポートには対応していません。
lead: >-
  InternVL3はOpenGVLabが公開したネイティブなマルチモーダル大規模言語モデルであり、1段階の事前学習で視覚と言語を共同学習します。LibreYOLOはこれをオープンボキャブラリ物体検出器としてラップします。任意のテキストラベル一覧がクラス集合になり、固定ヘッドもファインチューニングも不要です。
keywords:
  - InternVL3
  - InternVL
  - 視覚言語モデル
  - オープンボキャブラリ検出
  - VLM
  - OpenGVLab
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # オープンボキャブラリ 固定クラスヘッドではなく任意の単語を使用可能
        # 再設定するまでその後のすべてのpredict()/track()呼び出しで維持
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 未処理のチャット
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # 検出用の簡易機能の下にある脱出口 自由形式の質問や数え上げなど
        # ボックスラッパーが扱わない任意のプロンプトに対応
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 6305f020d3079d71
---

## インストール

InternVL3には `vlm` 追加パッケージが必要です。このパッケージはチャットテンプレートのバックボーン用に `transformers` を導入します。

```bash
pip install "libreyolo[vlm]"
```

## 推論

`LibreInternVL3` は `.pt` チェックポイントではなくPythonクラスです。`LibreYOLO()` ファクトリーから読み込むことはできず、`libreyolo` CLIもこれを解決しません。`LibreVLM(...)` ファクトリー（`from libreyolo import LibreVLM`）も、たとえば `LibreVLM("internvl3-2b")` という別名でこのファミリーに到達できます。これは下で使用するクラスを構築します。重みはLibreYOLOのミラーではなく、OpenGVLab独自の `-hf` Hugging Faceリポジトリから取得します。最初の呼び出しでダウンロードしてローカルにキャッシュし、その前に制限付きQwenの重みに関するライセンス通知を1回だけ記録します。

<code-tabs name="predict" />

`result.boxes` には他のファミリーと同様に解析済みの検出結果が含まれます。信頼度は仮の値です。InternVL3はボックスごとのスコアを出力しないため、すべての検出結果に同じ固定信頼度が設定されます。`conf=` はその固定値未満の行を除外するだけで、順位付けには使用できません。`iou` は、指定した重なりを超える同じクラスのほぼ重複したボックスを除外します。これは貪欲デコードが物体を繰り返すことによる副作用であり、クラス単位のNMS処理ではありません。`set_classes()` を省略すると、ボキャブラリのデフォルトはCOCO-80の名前になります。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

サイズは1b、2b、8bの3つで、すべてOpenGVLabのネイティブな `-hf` チェックポイントです。元のInternVL論文で説明された2タワーアーキテクチャではなく、Qwen LLMバックボーンを使用します。LibreYOLOのベンチマークハーネスはこのファミリーを測定していないため、比較できる公開精度値はありません。利用可能な計算資源に合わせてサイズを選択してください。

LibreYOLOはこのファミリーを推論専用として公開します。`train()`、`val()`、`export()` はすべて `NotImplementedError` を発生させます。代わりにアップストリームでファインチューニングし、その結果を読み込んでください。仮信頼度ではCOCO mAPが誤解を招くため、データセット検証は省略されています。また、トレースする状態辞書を持たない生成モデルのエクスポートは対象外です。

## ライセンス

<provenance-box>

InternVL3独自のコードはMITであり、寛容なライセンスの下で商用製品やクローズドソース製品に使用できます。このファミリーが読み込む `-hf` チェックポイントにはQwen LLMバックボーンが含まれ、Alibaba CloudのQwen Licenseに基づいて別途ライセンスされます。使用、変更、再配布は自由ですが、「Built with Qwen」または「Improved using Qwen」という帰属表示が必要です。また、商用利用には月間アクティブユーザー1億人の上限があり、それを超える場合はAlibaba独自の許可が必要です。LibreYOLOはこれらの重みをホストまたは再配布しません。`LibreInternVL3` は初回実行時に、Hugging Faceの `OpenGVLab/InternVL3-<size>-hf` から対応するサイズを直接ダウンロードし、その前にQwen Licenseに関する通知を1回だけ記録します。

</provenance-box>

## 引用

<citation-block />
