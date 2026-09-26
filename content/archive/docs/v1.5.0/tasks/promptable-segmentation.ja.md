---
title: プロンプト可能セグメンテーション
seo_title: LibreYOLOによるプロンプト可能セグメンテーション
description: >-
  LibreYOLOで点、ボックス、テキスト概念を物体マスクへ変換します。LibreSAMを通してSAM、SAM 2、SAM
  3、EdgeTAM、MobileSAM、PicoSAM3を読み込みます。
lead: >-
  プロンプト可能セグメンテーションはクリックをマスクへ変換します。物体を指すか周囲にボックスを描くと、モデルが輪郭を返します。LibreYOLOでは独立したタスクキーではなく、LibreSAMファクトリを通して読み込むモデルティアです。結果は通常のセグメンテーションResultsです。
keywords:
  - プロンプト可能セグメンテーション
  - 対話型セグメンテーション
  - Python Segment Anything
  - 点プロンプト
  - ボックスプロンプト
  - Python SAM
  - クリック マスク生成
last_verified: 1.5.0
snippets:
  predict:
    - label: 点とボックスのプロンプト
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 点はピクセル単位の[x, y] ラベルは正が1 負が0
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # ポリゴン
        print(result.boxes.xyxy)    # マスクから導出した密なボックス

        # ボックスプロンプトはボックスごとに1つのマスクを返す
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 1回エンコードして何度もプロンプト
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # set_imageは重い画像エンコーダーを1回実行してキャッシュする
        model.set_image(SAMPLE_IMAGE)
        first = model.predict(points=[640, 420], labels=[1])
        second = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
    - label: すべてをセグメンテーション
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # プロンプトがないと画像全体に点のグリッドを配置する
        # デフォルトの一辺32点では約1024回デコードしCPUでは遅い
        result = model.predict(SAMPLE_IMAGE, points_per_side=8)
        print(len(result.masks))
    - label: 曖昧性マスク
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # 1点は袖 シャツ 人物のいずれとも解釈できる multimask=Trueは
        # 最良の1つではなく全体と部分を表す3つのマスクすべてを返す
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## 定義

プロンプト可能セグメンテーションは画像と空間プロンプトを受け取り、プロンプトが指すもののマスクを返します。何も分類しないためクラスリストはなく、`result.boxes`は独立した検出ではなくマスクから導出した密なボックスを保持します。`result.masks`にマスクデータ、`result.masks.xy`にポリゴンが格納されます。

インターフェースはプロンプトです。`points`はピクセル単位の`[x, y]`座標で、物体ごとに1組を指定し、`labels`は各点を正（1、含める）または負（0、除外する）として示します。`bboxes`は`[x1, y1, x2, y2]`で、ボックスごとに1つのマスクを返します。点とボックスは組み合わせることができ、その場合は物体ごとに対応させ、同じ長さにする必要があります。すべてのプロンプトを省略すると、画像全体に点のグリッドを置くすべてセグメンテーション経路を実行します。

単一の点は本質的に曖昧です。袖をクリックしても、袖、シャツ、人物のいずれを意味する可能性もあります。そのため`multimask=True`は、プロンプトごとに最良の1つだけでなく、全体と部分を表す3つのマスクすべてを返します。`conf`は検出の信頼度ではなく、マスク品質のスコアであるモデルの予測IoUに基づいて絞り込みます。

LibreYOLOに`promptable`タスクキーはありません。このティアはインスタンスセグメンテーションと同じ`segment`キーとして登録されます。違いは呼び出し形式であり、そのため`LibreYOLO()`、`LibreOpenVocab()`、`LibreVLM()`と並ぶ固有のファクトリ`LibreSAM()`を持ちます。単一の`predict(image)`シグネチャでは、これらのモデルが前提とするループを表現できません。`set_image()`は画像エンコーダーを1回実行して埋め込みベクトルをキャッシュします。それ以降の`source=None`を使う`predict()`呼び出しはプロンプトのデコードだけを行い、`reset_image()`でキャッシュを消去します。計算コストの大半は画像エンコーダーで、画像ごとに1回だけ実行されるため、同じ画像への2つ目のプロンプトでは完全に省略されます。

## モデル

6つのファミリーを別名から`LibreSAM`で読み込めます。

[SAM](/docs/models/sam)がデフォルトで、`base`、`large`、`huge`のサイズがあり、`b`、`l`、`h`とも記述できます。

[SAM 2](/docs/models/sam-2)は`sam2-tiny`、`sam2-small`、`sam2-base-plus`、`sam2-large`です。LibreYOLOでは画像経路に対応します。

[SAM 3](/docs/models/sam-3)は`sam3`で、テキスト概念のプロンプトを受け付ける唯一のファミリーです。`text="yellow school bus"`は一致するすべてのインスタンスを返します。他のファミリーへ`text=`を渡すと、SAM 3を案内するメッセージとともに例外を送出します。重みにはLibreYOLOのMITライセンスではなくMeta独自のSAM Licenseが適用され、リポジトリはアクセス制限されています。初回ダウンロード前にモデルページで条件に同意し、`hf auth login`で認証してください。デプロイ前に[SAM 3](/docs/models/sam-3)を確認してください。

[EdgeTAM](/docs/models/edgetam)は`edgetam`で、SAM 2のオンデバイス向けバリアントです。LibreYOLOでは画像経路に対応します。

[MobileSAM](/docs/models/mobilesam)は`mobilesam`で、SAMのViT-Hエンコーダーを蒸留済みTinyViTエンコーダーへ置き換えます。

[PicoSAM3](/docs/models/picosam3)は`picosam3`で、エッジセンサー上のボックスプロンプト領域向け小型CNNです。ここではボックスプロンプトが契約のすべてです。点、テキスト、マスク、multimask、すべてセグメンテーションは、SAM 2またはSAM 3を案内するメッセージとともに例外を送出します。

このティアの追加パッケージは`transformers`を通して読み込む4つのファミリーに対応します。

```bash
pip install "libreyolo[sam]"
```

MobileSAMとPicoSAM3はLibreYOLOのネイティブ移植で、実行に`transformers`のインストールは必要ありません。

## 推論

<code-tabs name="predict" />

`source`と`set_image()`は連続した手順ではなく、どちらか一方を選びます。1回だけの呼び出しでは`predict()`へ画像を渡します。繰り返しプロンプトを指定する場合は先に`set_image()`を呼び出し、各プロンプトで`predict(source=None)`を実行します。`predict()`へ`device=`を渡すと、その呼び出しと以降の呼び出し用にモデルを移動し、キャッシュ済みの埋め込みベクトルを無効にします。

すべてセグメンテーションは高コストなモードです。`points_per_side`のデフォルトは32で、画像に対して約1024回のデコードを行います。CPUで対話的に使う場合は小さくしてください。このモードで`conf`を省略するとファミリー固有のグリッドしきい値が適用されます。一方、プロンプト経路で`conf`を省略するとすべてのマスクを残します。どちらのモードでも`conf=0.0`でフィルタリングを無効にし、`max_det`で返すマスク数を制限できます。

このバージョンではマスクプロンプトに対応せず、`masks=`は無視されず例外を送出します。ティア全体で`track()`も例外を送出します。これらは画像セグメンターなので、フレームごとに`predict()`を実行してください。入力ソースと結果の処理については[推論](/docs/predict)を参照してください。

## 学習

このティアのファミリーはLibreYOLO内部では学習できません。`train()`は例外を送出します。アップストリームでファインチューニングし、その重みを読み込んでください。

## 検証

このティアには検証器がなく、`val()`は例外を送出します。プロンプト可能マスクには採点対象となる固定クラス集合がないため、通常の検出やセグメンテーション指標を対応付けるものがありません。採点するには、対象となるプロンプトに対して自分で用意した参照マスクと比較してください。

## エクスポート

ティア全体としてエクスポートは対象外で、`export()`は例外を送出しますが、1つだけ例外があります。[PicoSAM3](/docs/models/picosam3)は生の96×96領域CNNを`roi_image -> mask_logits`としてONNXへエクスポートします。ボックスのクロップと画像座標へのマスクサイズ変更はPython側に残ります。その他のファミリーはすべてPyTorchの`predict()`を通して実行します。ライブラリ内の他の場所で利用できる形式については[エクスポート](/docs/export)を参照してください。
