---
title: RT-DETR
families:
  - rtdetr
seo_title: LibreYOLOのRT-DETR、RT-DETRv2、RT-DETRv4
description: >-
  LibreYOLOでRT-DETR、RT-DETRv2、RT-DETRv4を使って物体検出を行い、RT-DETRv2では回転バウンディングボックスも扱います。Apache-2.0の重みを使い、インストール、推論、学習、検証、エクスポートに対応します。
lead: >-
  リアルタイム推論向けに構築された検出Transformerです。密なグリッドではなく固定されたクエリ集合をデコードするため、NMSを実行しません。LibreYOLOには3つのバージョンがあり、読み込むチェックポイントで区別されます。バージョン2は回転バウンディングボックスにも対応します。
keywords:
  - RT-DETR
  - RT-DETRv2
  - RT-DETRv4
  - real-time detection transformer
  - DETR
  - 物体検出
  - 回転バウンディングボックス 検出
  - OBB
  - DOTA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        # バージョンはファイル名の一部でファクトリーはチェックポイントに基づき
        # 振り分けるため3つすべて同じ方法で読み込む
        model = LibreYOLO("LibreRTDETRv4s.pt")

        # ライブラリが受け付ける任意のソース: ファイル フォルダー URL Webカメラ番号
        # RTSP ストリーム または .streams リスト
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: 回転バウンディングボックス
      language: python
      code: |
        from libreyolo import LibreYOLO

        # バージョン2のみ -obb サフィックスでタスクを選択しチェックポイント自体の
        # テンソルから回転ありと認識するため task 引数は不要
        # この重みは DOTA v1.0 の15航空クラス向けで 1024 px
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)     # (N, 5): cx cy w h ラジアン
        print(obb.xyxyxyxy)  # 同じ行を4つのコーナーポイントとして表示
        print(result.boxes.xyxy)  # 外接する軸平行ボックス
    - label: 回転バウンディングボックスのCLI
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # coco128.yaml は初回使用時に128枚のサンプルをダウンロード
        # 実際の実行では data に独自データセットのYAMLを指定
        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # lora 追加パッケージが必要: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: マルチGPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() はオブジェクトではなく通常の辞書を返す
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: COCOで検証
      language: bash
      code: |
        # coco-val-only.yaml は5000枚の val2017 画像を取得し
        # 学習セットをスキップする ダウンロードスクリプトを内包するため
        # データセットがローカルにない場合は明示的な許可が必要
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: 回転バウンディングボックス
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 回転ありの検証は回転 IoU で照合するため位置が正しくても
        # 角度が誤った予測は不一致として扱う
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95(OBB)"])
        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # onnx 追加パッケージが必要: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: 回転バウンディングボックス
      language: bash
      code: >
        # ONNX と TorchScript は回転タスクで検証済みのターゲット

        # FP32 バッチ1 固定の 1024 x 1024 キャンバスを使用

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: エクスポートしたファイルを使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # ファクトリーはファイルサフィックスで振り分けるためエクスポート成果物も
        # 任意のチェックポイントと同様に読み込まれ同じ Results オブジェクトを返す
        model = LibreYOLO("LibreRTDETRr18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## インストール

RT-DETRにオプションの追加パッケージは不要です。インポートするものはすべて基本インストールに含まれ、`rtdetr`追加パッケージは何も追加しない安定した名前として用意されています。

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

返される`Results`オブジェクトはすべてのファミリーに共通するため、別の検出器への切り替えは1行の変更で済みます。`conf`と`max_det`はクエリとクラスに対する上位k件のデコードをフィルタリングします。調整するNMSステップはなく、`iou`は受け付けられますが使用されません。回転ありのチェックポイントは`result.obb`を直接埋め、外接する軸平行長方形で`result.boxes`も埋めます。ソース、ストリーミング、結果の処理については[推論](/docs/predict)を参照してください。

## バリアント

3つのバージョンに2つのタスクがあり、サイズコードは単一の系列ではありません。バージョン1はResNetまたはHGNetv2というバックボーン名に基づいてサイズを命名します。バージョン2はResNetの名前だけを再利用します。バージョン1がすでに2つのHGNetv2サイズを提供し、そこでのバージョン2の結果は非常に近いため、LibreYOLOは重複する重みを公開していません。バージョン4は単純な文字系列を使用し、バージョン1のHGNetv2名と衝突します。そのため、サイズコードだけではモデルを特定できません。バージョンはチェックポイントのファイル名に記載されます。

<benchmark-table task="detect" />

<va-embed />

バージョン2はバージョン1のアーキテクチャとstate dictの構成を維持し、変形可能アテンションのサンプリング方法を変更します。そのため、2つは形状ではなくチェックポイント内のメタデータで区別されます。バージョン4は異なる系統です。D-FINEのアーキテクチャとトレーナーを再利用し、DINOv3の視覚基盤モデルの教師からHGNetv2の生徒へ蒸留した重みを使います。LibreYOLOでは、`LibreRTDETRv4`はマスクヘッドを無効に固定した`LibreDFINE`のサブクラスなので、検出専用のままです。

### バージョン2の回転バウンディングボックス

バージョン2は2つ目のタスクを持つ唯一のバージョンです。サポートするタスクは`detect`と`obb`で、2つはグラフもサイズ系列も共有しません。検出は640 pxのResNetサイズを使い、回転あり検出は1024 pxでHGNetv2のn、s、m、l、x系列を使います。入力サイズはファミリー単位ではなくタスク単位で解決されます。チェックポイントは、5座標のボックスヘッドとバージョン2のサンプリングパラメータという自身のテンソルから回転ありと認識されます。そのため、`-obb`の重みは`task`引数なしで回転ありのグラフに読み込まれ、2つが一致しない場合は暗黙に再解釈されず、明確なエラーになります。

公開済みファイルは`LibreRTDETRv2n-obb.pt`から`LibreRTDETRv2x-obb.pt`までです。公式のDOTA v1.0シングルスケールチェックポイントをLibreYOLO形式に変換したもので、planeやshipからharbor、helicopterまで15の航空クラスを扱い、クラス名はチェックポイントに記録されています。検出側と異なり、回転ありのタスクは推論専用です。推論、検証、エクスポートは動作しますが、回転ありのモデルで`train()`を呼び出すと例外が発生します。トラッキングとテスト時拡張も回転バウンディングボックスをサポートしません。タスク、ラベル形式、メトリクスについては[回転物体検出](/docs/tasks/oriented-detection)で説明します。

## 学習

学習は公開済みチェックポイントから開始します。3つすべてのバージョンで`pretrained`は受け付けられた後に破棄されるため、`pretrained=False`を指定してもランダムに初期化されたモデルにはなりません。このセクションの内容はすべて検出に関するものです。バージョン2の回転ありタスクは推論専用で、2つは異なるバックボーンを使うため、検出重みからの転移経路はありません。

<code-tabs name="train" />

正しく設定すべき引数は学習率で、各バージョンがライブラリ全体の値ではなく固有のデフォルトを持ちます。Pythonの`train()`シグネチャはそのバージョンの学習設定から値を読み取り、`lr0`を渡さない場合はCLIも同じ値を解決します。バージョン1と2は`lr_backbone`も受け取り、元のレシピに従ってデフォルトを`lr0`の20分の1に設定します。バージョン4はD-FINEトレーナーで実行され、代わりに`backbone_lr_mult`でバックボーンのパラメータグループをスケーリングします。

変更する理由がない限り、`imgsz`はチェックポイントのネイティブサイズのままにしてください。別のサイズでの検証と推論も動作しますが、1つ問題が残ります。トークン数がネイティブサイズと一致する長方形サイズでは、誤ったアスペクト比用に構築された埋め込みベクトルが再利用されます。

データセット、データ拡張、マルチGPU、ロガーについては[学習](/docs/train)を参照してください。

## 検証

`val()`は、学習に使用した形式の任意のデータセットで測定した適合率、再現率、mAP 50、mAP 50-95を含む`metrics/`キーの辞書を返します。

<code-tabs name="val" />

上記のベンチマーク表の行はLibreYOLOのベンチマークハーネスから取得されています。表の下の注記には、使用したデータセットと実行記録へのリンクが記載されています。

回転ありの検証は同じ呼び出しで実行され、同じキーに加えて、`(OBB)`サフィックス付きで4つのキーを繰り返し報告します。照合には外接長方形のIoUではなく回転IoUを使うため、角度の誤りは不一致になります。このタスクでは`augment=True`が拒否されます。

## エクスポート

<export-matrix />

このマトリクスは各系統を1ページで扱います。3つのバージョンで形式に対する対応状況が異なる場合、そのセルには最も弱いものが表示されるため、どのバージョンを読み込んでも実際以上の対応を示すことはありません。回転ありの行はバージョン2だけに該当します。ONNXとTorchScriptはFP32、バッチ1、固定の1024 x 1024キャンバスで検証済みです。OpenVINO、TensorRT、ExecuTorchは変換と再読み込みができますが、クエリ集合全体で未加工出力の一致基準を満たしていません。上位のボックスは1 px未満の差で一致する一方、末尾ではずれが生じます。

エクスポートした成果物は、ファイルサフィックスに基づいて`LibreYOLO()`から再度読み込めます。そのため、`.onnx`または`.engine`ファイルはチェックポイントと同様に動作し、同じ`Results`を返します。

<code-tabs name="export" />

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

ファイル名にはバージョン、サイズ、タスクの順に含まれます。検出重みは`LibreRTDETR<size>.pt`、`LibreRTDETRv2<size>.pt`、`LibreRTDETRv4<size>.pt`で、すべて640 pxです。回転ありの重みはバージョン2だけに存在し、タスクサフィックスが追加されます。`LibreRTDETRv2n-obb.pt`から`LibreRTDETRv2x-obb.pt`までで、すべて1024 px、COCOではなくDOTA v1.0で学習されています。

## ライセンス

<provenance-box></provenance-box>

## 引用

<citation-block />

上のブロックは、著者がバージョン1と2の検出向けに公開しているものです。バージョン2の回転あり重みには、3つ目のアップストリームがあります。DOTAチェックポイントの取得元であるApache-2.0のRiO-DETRリポジトリ[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR)です。これらのいずれかを使用した場合は、そのプロジェクトを引用してください。バージョン4は別のグループによる別の論文で、[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation)に専用の引用ブロックがあります。バージョン4のチェックポイントを使用した場合は、そちらを引用してください。

