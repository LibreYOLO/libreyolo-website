---
title: RAFT
families: [raft]
seo_title: "PythonでRAFTのオプティカルフロー：フレーム間の密な動き、BSD-3"
description: "LibreYOLOでRAFTを実行し、動画フレーム間の密なオプティカルフローを推定します。smallとlargeのチェックポイントに加え、SintelとKITTIでのファインチューニング版があり、BSD-3-Clauseです。"
lead: "RAFTはTeedとDengによる再帰型の全ペアオプティカルフローネットワークです（ECCV 2020）。2枚のフレームを受け取り、1枚目の各ピクセルについて2枚目への変位を返します。LibreYOLOはアップストリームの推論コードを移植し、flowタスク向けに公開済みの4つのチェックポイントをミラーしています。"
keywords: [RAFT, raft オプティカルフロー, オプティカルフロー, 密 オプティカルフロー, Recurrent All-Pairs Field Transforms, raft-things, raft-sintel, raft-kitti, flo ファイル]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")   # 初回使用時に重みをダウンロード
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32 でピクセル単位の (dx, dy)
        print(flow.mean_magnitude)    # ピクセル単位の平均変位
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")

        # N フレームから N - 1 個の結果 結果 i はフレーム i から i + 1 へのフロー
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRAFTl-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreRAFTl-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均エンドポイント誤差 ピクセル単位
        print(metrics["metrics/px1"])      # 誤差が 1 px 未満のピクセルの割合
        print(metrics["metrics/fl_all"])   # KITTI の外れ値率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRAFTl-flow.pt data=my_flow.yaml
source_hash: 5576df12c513723d
---

RAFTは2枚のフレームのすべてのピクセルの組の間に4D相関ボリュームを構築し、再帰的な更新ブロックで1つのフローフィールドをリファインします。SEA-RAFT、FlowSeek、MEMFOFをはじめ、後の多くのフローネットワークが出発点とするリファレンスです。よく知られたベースラインが欲しい場合や、KITTIまたはSintelに近い映像向けにそのファインチューニング版が欲しい場合に選んでください。

## インストール

RAFTには追加のextraは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Face organizationからダウンロードされます。モデルを直接構築したい場合は、`LibreRAFT`クラスも`libreyolo`からエクスポートされています。

<code-tabs name="predict" />

フローには少なくとも2枚のフレームが必要です：画像1枚だけでは例外になります。画像のリスト、名前順に並べたディレクトリ、動画、ストリームはそれぞれ1つのシーケンスで、`N`枚のフレームから`N - 1`個の結果が得られます。`result.flow`は、最初のフレームの元のキャンバス上の`(H, W, 2)`のfloat32配列を持つ`FlowField`です。`flow.data[y, x]`はピクセル単位の`(dx, dy)`で、`+x`は右、`+y`は下です。RAFTは信頼度マップを出力しないため、`flow.confidence`は`None`です。`result.boxes`は`None`です。`save=True`はMiddleburyのカラーホイールを出力し、`arrows=True`または`warp=True`で他の表示を選び、`save_flow=True`は各組の生のフィールドも`.flo`ファイルとして出力します。ペイロードと表示の詳細は[オプティカルフローのタスク](/docs/tasks/optical-flow)ページで説明しています。

## バリアント

| サイズ | アップストリームのチェックポイント | ファイル |
| --- | --- | --- |
| `s` | raft-small | `LibreRAFTs-flow.pt` |
| `l` | raft-things（標準） | `LibreRAFTl-flow.pt` |
| `l` | raft-sintelのファインチューニング | `LibreRAFTl-flow-sintel.pt` |
| `l` | raft-kittiのファインチューニング | `LibreRAFTl-flow-kitti.pt` |

`LibreRAFT`のデフォルトはサイズ`l`です。推論はフレーム本来の解像度で行われ、アップストリームの`InputPadder`と同じく8の倍数にパディングされ、リファインメント反復は20回に固定されています（アップストリームの`demo.py`）。`imgsz`は長辺の上限を設定します。それより大きいフレームは縮小され、フィールドは元のキャンバスへ拡大し直されます。

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

RAFTは[Princeton Vision & Learning Lab](https://github.com/princeton-vl/RAFT)のTeedとDengの成果です。アップストリームの推論コードはBSD-3-Clauseライセンスの下で同梱されており、アップストリームのダウンロードスクリプトにある4つのチェックポイントは、同じライセンスの下でLibreYOLOのHugging Face organizationにアクセス制限なしでミラーされています。BSD-3-Clauseは、著作権表示、条件、免責事項を残す限り、商用利用およびクローズドソースでの利用を許可しています。

</provenance-box>

## 引用

<citation-block />
