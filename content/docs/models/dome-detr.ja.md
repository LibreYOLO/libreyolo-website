---
title: Dome-DETR
families:
  - domedetr
seo_title: Dome-DETR：LibreYOLOでの微小物体検出
description: >-
  Dome-DETRで微小物体の検出、学習、検証を行います。ミラーされた学習済みチェックポイントには、学術研究目的のみの条件が引き続き適用されます。
lead: >-
  D-FINEを基盤とする微小物体専用モデルです。密度ヘッドが物体の位置を判断し、エンコーダーのアテンションを物体のあるウィンドウに限定し、固定数ではなく密度に応じてクエリ数を決めます。LibreYOLOは物体検出でDome-DETRをサポートします。
keywords:
  - Dome-DETR
  - 微小物体検出
  - 小物体検出
  - 航空画像 物体検出
  - ドローン 物体検出
  - リモートセンシング
  - VisDrone
  - AI-TOD
  - DETR
  - density adaptive queries
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 学習済み重みは学術研究目的に限定
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt", device="cpu")
        print(model(SAMPLE_IMAGE).boxes)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
source_hash: 8482301790a9b8d9
---

## インストール

Dome-DETRにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

変換済みの6つのチェックポイントは、LibreYOLOのミラーから自動的にダウンロードされます。アップストリームの条件により、利用は学術研究に限定されます。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーに共通するため、別の検出器への切り替えは1行の変更で済みます。`conf`と`max_det`はクエリ選択をフィルタリングします。デコーダーはNMSステップを持たない集合予測器なので、`iou`はAPIの互換性のため受け付けられますが効果はありません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

このファミリーでは2つの機能が無効です。PAQIのクエリ数はデータに依存し、順伝播の形状が画像ごとに変わるため、CUDAグラフのキャプチャは無効です。これはグラフキャプチャでは吸収できない変化です。テスト時拡張は単一の固定正方形サイズで実行されるため、マルチスケールTTAの要求は何も行いません。

## バリアント

s、m、lの3サイズがあり、すべて800 x 800で動作します。サイズでバックボーンが決まり、重みの由来となるデータセットでデコーダーの深さとクエリ予算が決まるため、サイズコードだけではグラフを特定できません。AI-TOD-V2の重みは画像ごとに300〜1500クエリ、VisDroneの重みは250〜500クエリを選択します。largeモデルはAI-TOD-V2では4つのデコーダー層を実行し、VisDroneでは6つを実行します。

Dome-DETRはD-FINEに3つの要素を追加したものです。DeFEは密度マップを予測します。MWASはそのマップを使い、すべての場所にアテンションを適用する代わりに、実際に物体があるウィンドウへエンコーダーのアテンションを限定します。PAQIは固定の300件をデコードする代わりに、同じ密度からクエリ集合のサイズを決めます。効果は物体が最小の領域に集中し、物体が大きくなるほど小さくなります。アップストリーム独自のアブレーションでは、非常に小さい物体のAPが14.0から17.8へ上昇する一方、中サイズの物体のAPは45.4から46.4への上昇にとどまります。航空画像、ドローン画像、リモートセンシング画像では[D-FINE](/docs/models/d-fine)と併用してください。D-FINEの代替ではありません。

このファミリーにはVision Analysisのベンチマーク行が記録されていません。

## 学習

Dome-DETRは学習できます。学習ではアップストリームの完全な目的関数を実行します。D-FINEの損失に加え、DeFEの密度とカウントの教師信号を使います。また、パディングされたクエリを分類項から除外し、画像ごとのノイズ除去アテンションマスクによって、ある画像のパディングが別の画像に漏れないようにします。

<code-tabs name="train" />

設定はD-FINEのレシピを継承し、MWASの要件に合わせて変更されています。`imgsz`は800、`lr0`は`2e-4`で、バックボーンのパラメータグループは`backbone_lr_mult=0.1`でスケーリングされます。さらに、MWASのウィンドウでは入力がstride 8で割り切れる必要があるため、`multi_scale`は強制的に無効になります。`batch`のデフォルトはD-FINEの16ではなく4です。PAQIは各バッチを最も幅の広いメンバーに合わせてパディングするため、メモリ使用量はバッチ内の平均的な画像ではなく、最も負荷の高い画像に左右されます。

精度について正直に示すべき注意点が1つあります。アップストリームは`MultiStepLR(milestones=[80, 120], gamma=0.8)`を使って160エポック学習しますが、ここでのデフォルトは同じ160エポックにD-FINEのフラットコサインスケジュールを使います。アップストリームのスケジュールはここでは再現されておらず、論文のAP値も再現されていません。そのため、それらはこのレシピで到達できるという保証ではなく、アップストリーム著者の結果として扱ってください。論文との一致が目的なら、アップストリームのスケジュールを指定してください。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`はメトリクス名をキーとする辞書を返し、`verbose`を有効のままにするとクラス別の結果を表示します。

<code-tabs name="val" />

検証は、学習に使用した形式の独自データセットに対して実行されます。このファミリーには測定対象となるCOCOチェックポイントが存在しないため、ライブラリのCOCO検証ゲートはここでは適用されません。

## エクスポート

どの形式でもエクスポートはサポートされておらず、要求するとファイルを生成せずに例外が発生します。

理由はPAQIです。密度でフィルタリングされた提案と、貪欲な密度適応抑制ループから画像ごとのクエリ数を決めます。そのため、デコーダーの出力長はグラフではなく入力の特性になります。トレースを行うと、トレース用画像で偶然生成されたクエリ数が固定され、ほかのすべての画像で誤った結果を何の警告もなく返す成果物になります。静的な形式では250〜1500件の候補すべてに対して抑制を展開する必要があり、固定の上位k件にまとめると、このファミリーが存在する理由である微小物体の再現率が失われます。エクスポート可能な検出Transformerが必要なら、[D-FINE](/docs/models/d-fine)を選んでください。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box>

6つのミラーには、アップストリームの学術研究目的のみという制限が引き続き適用されます。コードには別のライセンスが適用されます。アップストリームのリポジトリはApache-2.0、LibreYOLOへの移植版はMITで、独自データを使って自分で学習した重みは自分のものです。

</provenance-box>
