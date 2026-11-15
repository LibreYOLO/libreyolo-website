---
title: DPFlow
families: [dpflow]
seo_title: "PythonでDPFlowのオプティカルフロー：最大8Kに対応するデュアルピラミッドフロー"
description: "LibreYOLOでDPFlowを実行し、入力サイズに適応するピラミッドで高解像度フレームの密なオプティカルフローを推定します。コードはApache-2.0で、重みは非商用です。"
lead: "DPFlowはHenrique Morimitsuらによる適応型のオプティカルフロー推定モデルです（CVPR 2025）。デュアルピラミッドのエンコーダーが入力サイズに応じて深さを変えるため、低解像度で学習したモデルが4Kや8Kのフレームにも汎化します。LibreYOLOはptlflowからこれを移植しています。重みのライセンスはCC BY-NC-SA 4.0で、非商用利用に限られます。"
keywords: [DPFlow, オプティカルフロー, 高解像度 オプティカルフロー, 4k optical flow, デュアルピラミッド, ptlflow, 密 オプティカルフロー, flo ファイル]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDPFlowm-flow.pt")   # 初回使用時に重みをダウンロード
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32 でピクセル単位の (dx, dy)
        print(flow.mean_magnitude)    # ピクセル単位の平均変位
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDPFlowm-flow.pt")

        # N フレームから N - 1 個の結果 結果 i はフレーム i から i + 1 へのフロー
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDPFlowm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreDPFlowm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDPFlowm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均エンドポイント誤差 ピクセル単位
        print(metrics["metrics/px1"])      # 誤差が 1 px 未満のピクセルの割合
        print(metrics["metrics/fl_all"])   # KITTI の外れ値率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDPFlowm-flow.pt data=my_flow.yaml
source_hash: 6bdbd3161d29d017
---

DPFlowは画像の対角線の長さからピラミッドのレベル数を選びます。1K前後で3レベル、解像度が2倍になるごとに1レベル増えます。LibreYOLOはこの規則を維持し、デフォルトではフレーム本来の解像度で実行します。適応的な深さが効いてくるのはまさにこの場合です。高解像度の映像を扱う研究に選んでください。重みは非商用です。

## インストール

DPFlowには追加のextraは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Face organizationからダウンロードされます。モデルを直接構築したい場合は、`LibreDPFlow`クラスも`libreyolo`からエクスポートされています。

<code-tabs name="predict" />

フローには少なくとも2枚のフレームが必要です：画像1枚だけでは例外になります。画像のリスト、名前順に並べたディレクトリ、動画、ストリームはそれぞれ1つのシーケンスで、`N`枚のフレームから`N - 1`個の結果が得られます。`result.flow`は、最初のフレームの元のキャンバス上の`(H, W, 2)`のfloat32配列を持つ`FlowField`です。`flow.data[y, x]`はピクセル単位の`(dx, dy)`で、`+x`は右、`+y`は下です。DPFlowは信頼度マップを出力しないため、`flow.confidence`は`None`です。`result.boxes`は`None`です。`save=True`はMiddleburyのカラーホイールを出力し、`arrows=True`または`warp=True`で他の表示を選び、`save_flow=True`は各組の生のフィールドも`.flo`ファイルとして出力します。ペイロードと表示の詳細は[オプティカルフローのタスク](/docs/tasks/optical-flow)ページで説明しています。

## バリアント

アーキテクチャは1つでサイズは`m`、チェックポイントは4つです。

| ファイル | 学習ステージ |
| --- | --- |
| `LibreDPFlowm-flow.pt` | FlyingThings3D（`things`）、汎化用のチェックポイント（標準） |
| `LibreDPFlowm-flow-sintel.pt` | Sintelでのファインチューニング |
| `LibreDPFlowm-flow-kitti.pt` | KITTIでのファインチューニング |
| `LibreDPFlowm-flow-spring.pt` | Springでのファインチューニング |

反復スケジュールはアップストリームのテスト設定に固定されており、ピラミッドレベルごとに4回のリファインメント反復を行います。各組は独立に推定され、アップストリームのフレーム間のウォームスタートは使いません。`imgsz`は長辺の上限を設定します（デフォルトは8192で、実質的に上限なし）。これを超えるフレームは縮小され、フィールドは元のキャンバスへ拡大し直されます。

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

DPFlowはHenrique Morimitsuらの成果で、[ptlflow](https://github.com/hmorimitsu/ptlflow)で公開されています。コードと重みには異なるライセンスが適用されます。

- コードは**Apache-2.0**です：ptlflowの`models/dpflow`を、timm由来のヘルパー（Apache-2.0）とRAFTから改変した入力パディング（BSD-3-Clause）とともに同梱しています。
- 重みは**CC BY-NC-SA 4.0**です：**非商用利用のみ**で、帰属表示が必要であり、これを元に作成して共有するものにも同じライセンスを適用する必要があります。DPFlowのREADMEは、学習データセットのライセンスのため、重みは学術・研究目的に限られるとも述べています。4つのチェックポイントはすべて研究用の重みとして扱ってください。

LibreYOLOは重みをライセンス文と非商用のバナーとともに、アクセス制限なしでミラーしており、読み込み時に通知を表示します。

</provenance-box>

## 引用

<citation-block />
