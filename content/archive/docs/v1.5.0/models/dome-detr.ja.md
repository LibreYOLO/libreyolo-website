---
title: Dome-DETR
families:
  - domedetr
seo_title: Dome-DETR：LibreYOLOでの微小物体検出
description: >-
  LibreYOLOでDome-DETRを使い、航空画像やドローン画像の微小物体を検出します。アップストリームの重みを変換し、MITライセンスのコードで推論、ファインチューニング、検証を行います。
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
last_verified: 1.5.0
snippets:
  predict:
    - label: 変換してから推論
      language: bash
      code: |
        # LibreYOLO は Dome-DETR の重みをホストしないためチェックポイントを
        # アップストリームのリポジトリから取得して一度だけ変換
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 単なる名前ではなくローカルパスを指定 このファミリーではダウンロードなし
        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        result = model("drone-frame.jpg", save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt
        source=drone-frame.jpg save=True
    - label: クラス名
      language: python
      code: |
        from libreyolo import LibreYOLO

        # COCO チェックポイントはないためクラスは重みの学習に使った
        # データセットに由来しチェックポイントのメタデータから読み取る
        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")
        print(aitod.model.names)     # 9個の AI-TOD-V2 クラス

        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        print(visdrone.model.names)  # 12個の VisDrone クラス
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
source_hash: 381f01d769e7c420
---

## インストール

Dome-DETRにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

自動ダウンロードされるものはありません。LibreYOLOはこれらの重みをホストしていないため、アップストリームのチェックポイントを取得し、一度変換してから、変換済みファイルをパスで読み込みます。理由については[ライセンス](#licensing)で説明します。

<code-tabs name="predict" />

返される`Results`オブジェクトはすべてのファミリーに共通するため、別の検出器への切り替えは1行の変更で済みます。`conf`と`max_det`はクエリ選択をフィルタリングします。デコーダーはNMSステップを持たない集合予測器なので、`iou`はAPIの互換性のため受け付けられますが効果はありません。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

このファミリーでは2つの機能が無効です。PAQIのクエリ数はデータに依存し、順伝播の形状が画像ごとに変わるため、CUDAグラフのキャプチャは無効です。これはグラフキャプチャでは吸収できない変化です。テスト時拡張は単一の固定正方形サイズで実行されるため、マルチスケールTTAの要求は何も行いません。

## バリアント

s、m、lの3サイズがあり、すべて800 x 800で動作します。サイズでバックボーンが決まり、重みの由来となるデータセットでデコーダーの深さとクエリ予算が決まるため、サイズコードだけではグラフを特定できません。AI-TOD-V2の重みは画像ごとに300〜1500クエリ、VisDroneの重みは250〜500クエリを選択します。largeモデルはAI-TOD-V2では4つのデコーダー層を実行し、VisDroneでは6つを実行します。

Dome-DETRはD-FINEに3つの要素を追加したものです。DeFEは密度マップを予測します。MWASはそのマップを使い、すべての場所にアテンションを適用する代わりに、実際に物体があるウィンドウへエンコーダーのアテンションを限定します。PAQIは固定の300件をデコードする代わりに、同じ密度からクエリ集合のサイズを決めます。効果は物体が最小の領域に集中し、物体が大きくなるほど小さくなります。アップストリーム独自のアブレーションでは、非常に小さい物体のAPが14.0から17.8へ上昇する一方、中サイズの物体のAPは45.4から46.4への上昇にとどまります。航空画像、ドローン画像、リモートセンシング画像では[D-FINE](/docs/models/d-fine)と併用してください。D-FINEの代替ではありません。

LibreYOLOはベンチマーク対象となるチェックポイントを公開していないため、このファミリーのベンチマーク行も公開していません。

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

一覧にするものはありません。LibreYOLOはDome-DETRの重みを公開しておらず、`LibreDOMEDETR<size>-<dataset>.pt`形式の名前からダウンロードが解決されることもありません。

アップストリームは2つのデータセットそれぞれについてs、m、lの6チェックポイントを公開しています。AI-TOD-V2は9クラス、VisDroneは12クラスです。COCOチェックポイントはないため、正規ファイル名には常にデータセットのサフィックスが含まれ、クラス名はファミリー定数ではなくチェックポイントのメタデータに格納されます。単独の`LibreDOMEDETRs.pt`を要求すると、存在する2つのファイル名と変換コマンドを示すメッセージとともに直ちに例外が発生します。404になるダウンロードは試みません。

`weights/convert_domedetr_weights.py`が変換を行います。LibreYOLOのグラフを再構築してアップストリームのテンソルを読み込みます。キーが1つでも欠けている、予期しない、または形状が誤っている場合は何も書き出しません。そのため、変換済みファイルは完全に一致するか、存在しないかのどちらかです。アップストリームの`.pth`を指定し、サイズとバリアントを渡してください。

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

数値的忠実度について、`weights/parity_domedetr.py`は6つのチェックポイントすべてでこの移植版とアップストリーム実装を比較します。最初にMWASのウィンドウマスクをビット単位で確認し、`pred_logits`と`pred_boxes`の両方で`max_abs_diff == 0.0`を報告します。さらに、すべての損失項をアップストリームのcriterionと個別に比較します。この検証の位置付けを明確にしておきます。アップストリームのチェックアウトと公開済みチェックポイントがディスク上に必要で、手動で実行するスクリプトです。継続的インテグレーションには含まれず、これを再現するCIジョブもありません。

## ライセンス

<provenance-box>

このファミリーがミラーされない理由は重みにあります。アップストリームのモデルカードではメタデータにライセンス項目がなく、本文ではプロジェクトがApache-2.0だとしつつ、素材を学術研究目的だけに制限しています。この2つの解釈は一致せず、より厳格な方も再配布を許諾していません。そのため、LibreYOLOは明確化されるまでファイルをコピーせず、アップストリームのリポジトリにリンクします。ここでの[YOLO-NAS](/docs/models/yolo-nas)にも同じ考え方が適用されます。

コードは別の問題であり、より明確です。アップストリームのリポジトリはApache-2.0、LibreYOLOへの移植版はMITで、独自データを使って自分で学習した重みは自分のものです。

</provenance-box>

## 引用

Dome-DETRはACM Multimedia 2025で「Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection」として発表されました。プレプリントは[arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741)にあります。著者はリポジトリでBibTeXブロックを公開していないため、ここでは手作業で組み立てたものを掲載していません。

<citation-block />

