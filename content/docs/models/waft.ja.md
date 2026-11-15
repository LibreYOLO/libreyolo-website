---
title: WAFT
families: [waft]
seo_title: "PythonでWAFTのオプティカルフロー：Warping-Alone Field Transforms、BSD-3"
description: "LibreYOLOでWAFTを実行し、コストボリュームなしで密なオプティカルフローを推定します。サイズはエンコーダーごとにt、s、m、lで、s、m、lはBSD-3-Clause、tはDINOv3 Licenseです。"
lead: "WAFT（Warping-Alone Field Transforms）はWangとDengによるオプティカルフロー手法です（ICLR 2026）。RAFTのコストボリュームを、高解像度の特徴量ワーピングとViTによるリファインメントループに置き換えています。LibreYOLOはアップストリームのコードを移植し、特徴量エンコーダーごとに1つずつ、4つのゼロショットチェックポイントをミラーしています。"
keywords: [WAFT, Warping-Alone Field Transforms, オプティカルフロー, 密 オプティカルフロー, DINOv3, Depth Anything V2, Twins-SVT, flo ファイル]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreWAFTm-flow.pt")   # 初回使用時に重みをダウンロード
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32 でピクセル単位の (dx, dy)
        print(flow.mean_magnitude)    # ピクセル単位の平均変位
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreWAFTm-flow.pt")

        # N フレームから N - 1 個の結果 結果 i はフレーム i から i + 1 へのフロー
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreWAFTm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreWAFTm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreWAFTm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均エンドポイント誤差 ピクセル単位
        print(metrics["metrics/px1"])      # 誤差が 1 px 未満のピクセルの割合
        print(metrics["metrics/fl_all"])   # KITTI の外れ値率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreWAFTm-flow.pt data=my_flow.yaml
source_hash: 0cd50470188b142d
---

WAFTはRAFTとその系譜のモデルが構築する相関ボリュームを使わず、代わりに各リファインメントステップの前に、現在のフロー推定で2枚目のフレームの高解像度特徴量をワープします。4つのチェックポイントはすべてアップストリームのゼロショットモデルです（TartanAir、続いてChairs、続いてThings）。特定のエンコーダーを選ぶ理由がなければ、アップストリームのREADMEが下流での利用に推奨するa1チェックポイントである`m`を選んでください。

## インストール

WAFTには追加のextraは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Face organizationからダウンロードされます。モデルを直接構築したい場合は、`LibreWAFT`クラスも`libreyolo`からエクスポートされています。

<code-tabs name="predict" />

フローには少なくとも2枚のフレームが必要です：画像1枚だけでは例外になります。画像のリスト、名前順に並べたディレクトリ、動画、ストリームはそれぞれ1つのシーケンスで、`N`枚のフレームから`N - 1`個の結果が得られます。`result.flow`は、最初のフレームの元のキャンバス上の`(H, W, 2)`のfloat32配列を持つ`FlowField`です。`flow.data[y, x]`はピクセル単位の`(dx, dy)`で、`+x`は右、`+y`は下です。WAFTは信頼度マップを出力しないため、`flow.confidence`は`None`です。`result.boxes`は`None`です。`save=True`はMiddleburyのカラーホイールを出力し、`arrows=True`または`warp=True`で他の表示を選び、`save_flow=True`は各組の生のフィールドも`.flo`ファイルとして出力します。ペイロードと表示の詳細は[オプティカルフローのタスク](/docs/tasks/optical-flow)ページで説明しています。

## バリアント

| サイズ | アップストリームのチェックポイント | パラメータ | ファイル |
| --- | --- | ---: | --- |
| `t` | a2、DINOv3 ViT-S/16エンコーダー | 56.5 M | `LibreWAFTt-flow-dinov3.pt` |
| `s` | a2、Depth Anything V2 ViT-S | 56.9 M | `LibreWAFTs-flow-dav2.pt` |
| `m` | a1、Depth Anything V2 ViT-S | 64.2 M | `LibreWAFTm-flow.pt` |
| `l` | a2、Twins-SVT-Largeエンコーダー | 136.0 M | `LibreWAFTl-flow-twins.pt` |

エンコーダーはファイル名の重みバリアントで、各サイズは表の名前でのみ存在します。それ以外の組み合わせは、公開されている名前を示して例外になります。`LibreWAFT`のデフォルトはサイズ`m`です。推論は本来の解像度で行われます：ネットワークはエンコーダーに応じてフレームの組を112、32、または16の倍数にパディングし、フローを元のサイズに切り戻します。リファインメント反復はアップストリームの評価時のデフォルトである5回に固定されています。`imgsz`は長辺の上限を設定します（デフォルトは8192で、実質的に上限なし）。

学習は提供していません：`train()`は`NotImplementedError`を発生させます。

## 検証

`val()`はフローのデータセット（シーケンスフォルダーごとのフレームと、Middleburyの`.flo`、KITTIの16ビットPNG、または`.npy`の正解データ。[データセット形式](/docs/tasks/optical-flow#dataset-format)を参照）に対して共有のフローバリデーターを実行します。平均エンドポイント誤差`metrics/epe`、`metrics/px1`と`metrics/px3`（誤差が1 pxおよび3 px未満の有効ピクセルの割合）、KITTIの外れ値率`metrics/fl_all`を報告します。

<code-tabs name="val" />

## エクスポート

<export-matrix />

エクスポートはまだどの形式でも利用できません。オプティカルフローのエクスポートは、flowタスクの2入力グラフ（`image1`、`image2`）とバックエンドのベクトル再スケールの契約がいずれかのファミリーで実装されるまで保留されています。

## チェックポイント

このファミリーで公開されているすべての重みファイルです。

<checkpoint-table />

## ライセンス

<provenance-box>

WAFTは[Princeton Vision & Learning Lab](https://github.com/princeton-vl/WAFT)のWangとDengの成果です。アップストリームのコードはBSD-3-Clauseライセンスの下で同梱されています。アップストリームは個別の重みライセンスを公開していないため、LibreYOLOはチェックポイントを、リポジトリのBSD-3-Clauseライセンスに各チェックポイントが組み込むエンコーダーのライセンスを加えた条件の下で、アクセス制限なしでミラーしています。

- `s`、`m`、`l`はBSD-3-Clauseです。それらのエンコーダーはApache-2.0です（`s`と`m`はDepth Anything V2 Small、`l`はtimmから派生したTwins-SVT-Largeのコピー）。商用利用およびクローズドソースでの利用が許可されています。
- `t`にはWAFTの著者がファインチューニングしたDINOv3 ViT-S/16が含まれるため、Metaの**DINOv3 License**が適用されます：OSI承認ではない独自のライセンスで、固有の利用制限と貿易管理の条項があります。`t`を商用で使う前に必ず読んでください。

</provenance-box>

## 引用

<citation-block />
