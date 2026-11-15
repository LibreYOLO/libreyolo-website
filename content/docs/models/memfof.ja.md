---
title: MEMFOF
families: [memfof]
seo_title: "PythonでMEMFOFのオプティカルフロー：メモリ効率の高いマルチフレームフロー"
description: "LibreYOLOでMEMFOFを実行し、Full HD動画に対してメモリ効率の高い3フレームのオプティカルフローを推定します。1つのネットワークに6つのチェックポイントがあり、BSD-3-Clauseです。"
lead: "MEMFOFはMSU Graphics & Media Labによる、メモリ効率の高いマルチフレームのオプティカルフローネットワークです（ICCV 2025 Highlight）。t-1、t、t+1の3フレームを読み取り、中央のフレームのフローを予測するもので、Full HD動画を対象としています。LibreYOLOのフローファミリーの中で、3フレームのウィンドウを読み取る唯一のものです。"
keywords: [MEMFOF, マルチフレーム オプティカルフロー, オプティカルフロー, full hd オプティカルフロー, 動画 オプティカルフロー, 密 オプティカルフロー, flo ファイル]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMEMFOFm-flow.pt")   # 初回使用時に重みをダウンロード
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32 でピクセル単位の (dx, dy)
        print(flow.mean_magnitude)    # ピクセル単位の平均変位
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMEMFOFm-flow.pt")

        # N フレームから N - 1 個の結果 結果 i はフレーム i から i + 1 へのフロー
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMEMFOFm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreMEMFOFm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMEMFOFm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均エンドポイント誤差 ピクセル単位
        print(metrics["metrics/px1"])      # 誤差が 1 px 未満のピクセルの割合
        print(metrics["metrics/fl_all"])   # KITTI の外れ値率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMEMFOFm-flow.pt data=my_flow.yaml
source_hash: 6df7e06e52ee9087
---

MEMFOFはRAFTの系譜にあるモデルで、中央のフレームの後方向（`t`から`t-1`）と前方向（`t`から`t+1`）のフローを1回のパスで予測します。LibreYOLOは前方向のフローを`Results.flow`として返します。後方向のフローも計算されますが、現時点ではタスクの契約には含めていません。ランナーが`(t-1, t, t+1)`のウィンドウを渡し、シーケンスの先頭ではフレーム0を`t-1`として繰り返すため、通常どおりリスト、フォルダー、動画を渡せます。高解像度の実世界の動画に選んでください。

## インストール

MEMFOFには追加のextraは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Face organizationからダウンロードされます。モデルを直接構築したい場合は、`LibreMEMFOF`クラスも`libreyolo`からエクスポートされています。

<code-tabs name="predict" />

フローには少なくとも2枚のフレームが必要です：画像1枚だけでは例外になります。画像のリスト、名前順に並べたディレクトリ、動画、ストリームはそれぞれ1つのシーケンスで、`N`枚のフレームから`N - 1`個の結果が得られます。`result.flow`は、最初のフレームの元のキャンバス上の`(H, W, 2)`のfloat32配列を持つ`FlowField`です。`flow.data[y, x]`はピクセル単位の`(dx, dy)`で、`+x`は右、`+y`は下です。MEMFOFは信頼度マップを出力しないため、`flow.confidence`は`None`です。`result.boxes`は`None`です。`save=True`はMiddleburyのカラーホイールを出力し、`arrows=True`または`warp=True`で他の表示を選び、`save_flow=True`は各組の生のフィールドも`.flo`ファイルとして出力します。ペイロードと表示の詳細は[オプティカルフローのタスク](/docs/tasks/optical-flow)ページで説明しています。

## バリアント

アーキテクチャは1つ（ResNet-34 FPN）でサイズは`m`、チェックポイントは6つです。

| ファイル | アップストリームのチェックポイント |
| --- | --- |
| `LibreMEMFOFm-flow.pt` | MEMFOF-Tartan-T-TSKH、著者が実世界の動画向けに推奨（標準） |
| `LibreMEMFOFm-flow-tartan-t.pt` | 以前に公開された学習ステージ |
| `LibreMEMFOFm-flow-tartan.pt` | 以前に公開された学習ステージ |
| `LibreMEMFOFm-flow-kitti.pt` | KITTIでのファインチューニング |
| `LibreMEMFOFm-flow-sintel.pt` | Sintelでのファインチューニング |
| `LibreMEMFOFm-flow-spring.pt` | Springでのファインチューニング |

フレームは本来の解像度で処理され、アップストリームと同じreplicateパディングで16の倍数にそろえられます。長辺が`imgsz`（デフォルトは1920で、MEMFOFが対象とするFull HD）を超えるフレームは先に縮小され、フィールドは元のサイズへ拡大し直されます。推論はアップストリームのデフォルトどおり、fp32で8回のリファインメント反復を行います。

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

MEMFOFは[MSU Graphics & Media Lab](https://github.com/msu-video-group/memfof)の成果です。アップストリームの推論コードはBSD-3-Clauseライセンスの下で同梱されています。そのアテンションモジュールはWTFPLの下にあるGMAに由来します。Hugging Face上のアップストリームのチェックポイントにはbsd-3-clauseのタグが付いていますが、ライセンスファイルは含まれていないため、LibreYOLOは変換した6つのチェックポイントをコードリポジトリのBSD-3-Clauseライセンスの下で、アクセス制限なしでミラーしています。BSD-3-Clauseは、著作権表示、条件、免責事項を残す限り、商用利用およびクローズドソースでの利用を許可しています。

</provenance-box>

## 引用

<citation-block />
