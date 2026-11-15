---
title: FlowSeek
families: [flowseek]
seo_title: "PythonでFlowSeekのオプティカルフロー：深度基盤モデルの事前知識を活用"
description: "LibreYOLOでFlowSeekを実行し、Depth Anything V2エンコーダーとモーション基底で密なオプティカルフローを推定します。サイズtとsはApache-2.0、mとlは非商用です。"
lead: "FlowSeekはPoggiとTosiによるオプティカルフローのネットワークです（ICCV 2025）。SEA-RAFT形式の再帰型ネットワークに、凍結したDepth Anything V2エンコーダーと深度から導いたモーション基底を加えています。LibreYOLOは4つのサイズでこれを移植しています。tとsはApache-2.0、mとlは非商用利用に限られます。"
keywords: [FlowSeek, オプティカルフロー, 密 オプティカルフロー, Depth Anything V2, motion bases, SEA-RAFT, flo ファイル]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFlowSeekt-flow.pt")   # 初回使用時に重みをダウンロード
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32 でピクセル単位の (dx, dy)
        print(flow.mean_magnitude)    # ピクセル単位の平均変位
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFlowSeekt-flow.pt")

        # N フレームから N - 1 個の結果 結果 i はフレーム i から i + 1 へのフロー
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFlowSeekt-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreFlowSeekt-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFlowSeekt-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均エンドポイント誤差 ピクセル単位
        print(metrics["metrics/px1"])      # 誤差が 1 px 未満のピクセルの割合
        print(metrics["metrics/fl_all"])   # KITTI の外れ値率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFlowSeekt-flow.pt data=my_flow.yaml
source_hash: b7af409bbc92d9c6
---

FlowSeekは深度基盤モデルから幾何の知識を借ります。Depth Anything V2エンコーダーの特徴量と、深度から計算した一連のモーション基底が、再帰的なフロー更新に入力されます。ホストしているチェックポイントは、アップストリームの最終的な汎用ファインチューニング（TartanAir、Chairs、Things、続いてSintel、KITTI、HD1K、Thingsの混合ステージ）で、アップストリームがSpringとLayeredFlowで評価しているものです。商用の作業には`t`または`s`を選んでください。`m`と`l`には非商用のエンコーダーが含まれます。

## インストール

FlowSeekには追加のextraは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Face organizationからダウンロードされます。モデルを直接構築したい場合は、`LibreFlowSeek`クラスも`libreyolo`からエクスポートされています。

<code-tabs name="predict" />

フローには少なくとも2枚のフレームが必要です：画像1枚だけでは例外になります。画像のリスト、名前順に並べたディレクトリ、動画、ストリームはそれぞれ1つのシーケンスで、`N`枚のフレームから`N - 1`個の結果が得られます。`result.flow`は、最初のフレームの元のキャンバス上の`(H, W, 2)`のfloat32配列を持つ`FlowField`です。`flow.data[y, x]`はピクセル単位の`(dx, dy)`で、`+x`は右、`+y`は下です。FlowSeekは信頼度マップを出力しないため、`flow.confidence`は`None`です。`result.boxes`は`None`です。`save=True`はMiddleburyのカラーホイールを出力し、`arrows=True`または`warp=True`で他の表示を選び、`save_flow=True`は各組の生のフィールドも`.flo`ファイルとして出力します。ペイロードと表示の詳細は[オプティカルフローのタスク](/docs/tasks/optical-flow)ページで説明しています。

## バリアント

アップストリームは2つのネットワークを学習しています。`s`と`l`は、アップストリームの評価設定と同じく、`t`と`m`の重みをより多くのリファインメント反復で再利用します。

| サイズ | コンテキストネット | Depth Anything V2エンコーダー | 反復回数 | 重みのライセンス |
| --- | --- | --- | ---: | --- |
| `t` | ResNet-18 | ViT-S | 4 | Apache-2.0 |
| `s` | ResNet-18 | ViT-S | 12 | Apache-2.0 |
| `m` | ResNet-34 | ViT-B | 4 | CC BY-NC 4.0、非商用 |
| `l` | ResNet-34 | ViT-B | 12 | CC BY-NC 4.0、非商用 |

ファイル名は`LibreFlowSeek<size>-flow.pt`です。`LibreFlowSeek`のデフォルトはサイズ`m`なので、商用利用できる重みが必要な場合は`t`または`s`のチェックポイントを明示的に渡してください。推論は本来の解像度で行われ、ネットワーク内で8の倍数にパディングされます。パディング後のキャンバスの各辺は128 px以上である必要があります。`imgsz`は長辺の上限を設定します。

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

FlowSeekは[ボローニャ大学](https://github.com/mattpoggi/flowseek)のPoggiとTosiの成果です。コードはApache-2.0で、SEA-RAFT由来のBSD-3-Clauseの部分を含み、LibreYOLOに同梱されています。

各チェックポイントにはDepth Anything V2エンコーダーが組み込まれているため、チェックポイントのライセンスはサイズによって異なります。

- `t`と`s`はViT-Sエンコーダーを組み込んでおり、**Apache-2.0**です：商用利用およびクローズドソースでの利用が許可されています。
- `m`と`l`は**CC BY-NC 4.0**のViT-Bエンコーダーを組み込んでいます：**非商用利用のみ**で、帰属表示が必要です。それらのモデルカードには非商用のバナーがあり、LibreYOLOは読み込み時に通知を表示します。

アップストリームはチェックポイント用の個別のライセンスを公開していないため、リポジトリのApache-2.0ライセンスが黙示の許諾として、エンコーダーのライセンスによる制限付きで適用されます。重みはLibreYOLOのHugging Face organizationにアクセス制限なしでミラーされています。

</provenance-box>

## 引用

<citation-block />
