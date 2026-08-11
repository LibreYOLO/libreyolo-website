---
title: LocateAnything
families:
  - locateanything
seo_title: LocateAnything：オープンボキャブラリ検出と点指示
description: >-
  LibreYOLOのLocateAnythingでオープンボキャブラリ検出と点指示を行います。任意のテキストラベルで推論できますが、学習、検証、エクスポートには対応していません。
lead: >-
  LocateAnythingはNVIDIAが公開した視覚言語グラウンディングモデルであり、座標トークンを1つずつ生成するのではなく、バウンディングボックスと点を並列にデコードします。LibreYOLOはこれをオープンボキャブラリ検出器および点指示器としてラップします。任意のテキストラベル一覧がクラス集合になり、固定ヘッドもファインチューニングも不要です。
keywords:
  - LocateAnything
  - NVIDIA
  - 視覚言語モデル
  - オープンボキャブラリ検出
  - 点検出
  - VLM
  - grounding
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # オープンボキャブラリ 固定クラスヘッドではなく任意の単語を使用可能
        # 再設定するまでその後のすべてのpredict()/track()呼び出しで維持
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: 点プロンプト
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        # task="point"ではボックスではなく一致した物体ごとに1つの点を返す
        # 読み込み済みモデルのタスクはmodel.set_task("point")で切り替え
        model = LibreLocateAnything(size="3b", task="point")
        model.set_classes(["the person closest to the camera"])
        result = model(SAMPLE_IMAGE, save=True)

        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: 未処理のチャット
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # 検出用の簡易機能の下にある脱出口 自由形式の質問や数え上げなど
        # ボックスラッパーが扱わない任意のプロンプトに対応
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 378ea758e507a096
---

## インストール

LocateAnythingには `vlm` 追加パッケージが必要です。このパッケージは `transformers` に加え、Hugging Faceのリモートコードが読み込み時にインポートする `decord`、`lmdb`、`peft` パッケージを導入します。

```bash
pip install "libreyolo[vlm]"
```

## 推論

`LibreLocateAnything` は `.pt` チェックポイントではなくPythonクラスです。`LibreYOLO()` ファクトリーから読み込むことはできず、`libreyolo` CLIもこれを解決しません。`LibreVLM(...)` ファクトリー（`from libreyolo import LibreVLM`）も、たとえば `LibreVLM("locate-anything")` という別名でこのファミリーに到達できます。これは下で使用するクラスを構築します。読み込み時にはNVIDIA独自のリモートモデルコードをHugging Faceからダウンロードして実行します。そのため、LibreYOLOは変更される可能性がある `main` ブランチではなく、1つの固定コミットリビジョンにダウンロードを固定し、最初のダウンロード前に1回だけライセンス通知を記録します。

<code-tabs name="predict" />

`result.boxes`（`detect` タスク）と `result.points`（`point` タスク）には、他のファミリーと同様に解析済み出力が含まれます。信頼度は仮の値です。LocateAnythingはボックスごとのスコアを出力しないため、すべての検出結果に同じ固定信頼度が設定されます。`conf=` はその固定値未満の行を除外するだけで、順位付けには使用できません。`set_classes()` を省略すると、ボキャブラリのデフォルトはCOCO-80の名前になります。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## バリアント

公開サイズは3bの1つです。2つのタスクが同じ重みを共有します。`detect`（デフォルト）はボックスを返し、`task="point"` は代わりに一致した物体ごとに1つの点を `result.points` で返します。読み込み済みモデルで両者を切り替えるには `model.set_task("point")` を使用します。LibreYOLOのベンチマークハーネスはこのファミリーを測定していないため、比較できる公開精度値はありません。

LibreYOLOはこのファミリーを推論専用として公開します。`train()`、`val()`、`export()` はすべて `NotImplementedError` を発生させます。代わりにアップストリームでファインチューニングし、その結果を読み込んでください。仮信頼度ではCOCO mAPが誤解を招くため、データセット検証は省略されています。また、トレースする状態辞書を持たない生成モデルのエクスポートは対象外です。

## ライセンス

<provenance-box>

NVIDIA Licenseは使用、複製、変更を許可しますが、NVIDIAとその関連会社以外の利用者について、モデルとその派生物を非商用利用、研究、評価だけに制限します。収益のしきい値や有償の例外はありません。LocateAnything-3Bは別途ライセンスされる2つのコンポーネントも組み合わせています。言語バックボーンのQwen2.5-3B-InstructはQwen Research License、視覚エンコーダーのMoonViT-SO-400MはMITです。LibreYOLOはこれらのいずれもホスト、ミラー、再配布しません。`LibreLocateAnything` は初回実行時に、固定された1つのコミットのHugging Face上の `nvidia/LocateAnything-3B` から、重みと必要なリモートコードを直接ダウンロードします。

</provenance-box>

## 引用

<citation-block />
