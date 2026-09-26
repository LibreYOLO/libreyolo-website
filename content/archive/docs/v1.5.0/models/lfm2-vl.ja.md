---
title: LFM2-VL
families:
  - lfm2vl
seo_title: LFM2-VL：LibreYOLOでのオープンボキャブラリ検出
description: >-
  LibreYOLOのLFM2-VLでオンデバイスのオープンボキャブラリ物体検出を行います。任意のテキストラベルで推論できますが、学習、検証、エクスポートには対応していません。
lead: >-
  LFM2-VLはLiquid
  AIが公開した小型のオンデバイス視覚言語モデルです。LibreYOLOはこれをオープンボキャブラリ物体検出器としてラップします。任意のテキストラベル一覧がクラス集合になり、固定ヘッドもファインチューニングも不要です。
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - 視覚言語モデル
  - オープンボキャブラリ検出
  - VLM
  - エッジ VLM
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # オープンボキャブラリ 固定クラスヘッドではなく任意の単語を使用可能
        # 再設定するまでその後のすべてのpredict()/track()呼び出しで維持
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 未処理のチャット
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # 検出用の簡易機能の下にある脱出口 自由形式の質問や数え上げなど
        # ボックスラッパーが扱わない任意のプロンプトに対応
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## インストール

LFM2-VLには `vlm` 追加パッケージが必要です。このパッケージはチャットテンプレートのバックボーン用に `transformers` を導入します。

```bash
pip install "libreyolo[vlm]"
```

## 推論

`LibreLFM2VL` は `.pt` チェックポイントではなくPythonクラスです。`LibreYOLO()` ファクトリーから読み込むことはできず、`libreyolo` CLIもこれを解決しません。`LibreVLM(...)` ファクトリー（`from libreyolo import LibreVLM`）も、たとえば `LibreVLM("lfm2-vl-450m")` という別名でこのファミリーに到達できます。これは下で使用するクラスを構築します。重みはLibreYOLOのミラーではなく、Liquid AI独自のHugging Faceリポジトリから取得します。最初の呼び出しでダウンロードしてローカルにキャッシュし、その前に1回だけライセンス通知を記録します。

<code-tabs name="predict" />

`result.boxes` には他のファミリーと同様に解析済みの検出結果が含まれます。信頼度は仮の値です。LFM2-VLはボックスごとのスコアを出力しないため、すべての検出結果に同じ固定信頼度が設定されます。`conf=` はその固定値未満の行を除外するだけで、順位付けには使用できません。`iou` は、指定した重なりを超える同じクラスのほぼ重複したボックスを除外します。これは貪欲デコードが物体を繰り返すことによる副作用であり、クラス単位のNMS処理ではありません。`set_classes()` を省略すると、ボキャブラリのデフォルトはCOCO-80の名前になります。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

サイズは450mと1.6bの2つで、どちらもオンデバイスデプロイ用に構築されたLiquid AIのLFM2.5-VLリリースです。LibreYOLOのベンチマークハーネスはこのファミリーを測定していないため、比較できる公開精度値はありません。利用可能な計算資源に合わせてサイズを選択してください。

LibreYOLOはこのファミリーを推論専用として公開します。`train()`、`val()`、`export()` はすべて `NotImplementedError` を発生させます。代わりにアップストリームでファインチューニングし、その結果を読み込んでください。仮信頼度ではCOCO mAPが誤解を招くため、データセット検証は省略されています。また、トレースする状態辞書を持たない生成モデルのエクスポートは対象外です。

## ライセンス

<provenance-box>

LFM Open License v1.0は商用利用、複製、変更を許可しますが、年間収益が1,000万ドル未満の場合に限られます。このしきい値以上の法人には、本契約に基づく商用利用のライセンスは一切付与されず、Liquid AIへ直接連絡する必要があります。条件を満たす非営利組織は、非商用または研究目的の利用について、このしきい値が免除されます。モデルはApache-2.0の `transformers` ライブラリを通じて読み込まれるため、LibreYOLOはLiquidAIのソースコードを同梱しません。また、重みのホストや再配布も行いません。`LibreLFM2VL` は初回実行時に、Liquid AI独自のHugging Faceリポジトリから対応するサイズを直接ダウンロードし、その前に1回だけ通知を記録します。

</provenance-box>

## 引用

<citation-block />
