---
title: SEA-RAFT
families: [searaft]
seo_title: "PythonでSEA-RAFTのオプティカルフロー：信頼度マップ付きの高速なRAFT、BSD-3"
description: "LibreYOLOでSEA-RAFTを実行し、フレーム間の密なオプティカルフローを、不確実性ヘッドによるピクセルごとの信頼度マップとともに推定します。サイズはs、m、lで、BSD-3-Clauseです。"
lead: "SEA-RAFT（Simple, Efficient, Accurate RAFT）はWang、Lipson、DengによるRAFTの系譜のモデルです（ECCV 2024）。2枚のフレーム間の密なオプティカルフローを予測し、LibreYOLOの他のフローファミリーとは異なり、ピクセルごとの信頼度マップも予測します。LibreYOLOはアップストリームの推論コードを移植し、SとMのチェックポイントをミラーしています。"
keywords: [SEA-RAFT, SEARAFT, オプティカルフロー, 密 オプティカルフロー, RAFT, フロー 信頼度, オプティカルフロー 不確実性, flo ファイル]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")   # 初回使用時に重みをダウンロード
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32 でピクセル単位の (dx, dy)
        print(flow.mean_magnitude)    # ピクセル単位の平均変位
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")

        # N フレームから N - 1 個の結果 結果 i はフレーム i から i + 1 へのフロー
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSEARAFTm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreSEARAFTm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均エンドポイント誤差 ピクセル単位
        print(metrics["metrics/px1"])      # 誤差が 1 px 未満のピクセルの割合
        print(metrics["metrics/fl_all"])   # KITTI の外れ値率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSEARAFTm-flow.pt data=my_flow.yaml
source_hash: 836f41517d948354
---

SEA-RAFTはRAFTの全ペア相関ピラミッドを維持しつつ、初期フローを直接回帰し、ConvNeXtの更新ブロックでリファインし、混合ラプラス損失で学習します。予測された混合分布の分散を、LibreYOLOは`flow.confidence`に変換します。[オプティカルフローのタスクページ](/docs/tasks/optical-flow)が例で使っているファミリーです。フローとあわせてピクセルごとの信頼度が欲しい場合に選んでください。

## インストール

SEA-RAFTには追加のextraは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Face organizationからダウンロードされます。モデルを直接構築したい場合は、`LibreSEARAFT`クラスも`libreyolo`からエクスポートされています。

<code-tabs name="predict" />

フローには少なくとも2枚のフレームが必要です：画像1枚だけでは例外になります。画像のリスト、名前順に並べたディレクトリ、動画、ストリームはそれぞれ1つのシーケンスで、`N`枚のフレームから`N - 1`個の結果が得られます。`result.flow`は、最初のフレームの元のキャンバス上の`(H, W, 2)`のfloat32配列を持つ`FlowField`です。`flow.data[y, x]`はピクセル単位の`(dx, dy)`で、`+x`は右、`+y`は下です。`flow.confidence`は予測された混合分布の分散から導かれる`[0, 1]`の`(H, W)`マップです：値が大きいほど、そのピクセルのベクトルに対するネットワークの確信度が高いことを示します。`result.boxes`は`None`です。`save=True`はMiddleburyのカラーホイールを出力し、`arrows=True`または`warp=True`で他の表示を選び、`save_flow=True`は各組の生のフィールドも`.flo`ファイルとして出力します。ペイロードと表示の詳細は[オプティカルフローのタスク](/docs/tasks/optical-flow)ページで説明しています。

## バリアント

| サイズ | エンコーダー | リファインメント反復回数 | ファイル |
| --- | --- | ---: | --- |
| `s` | ResNet18 | 4 | `LibreSEARAFTs-flow.pt` |
| `m` | ResNet34 | 4 | `LibreSEARAFTm-flow.pt` |
| `l` | ResNet34、`m`の重み | 12 | `LibreSEARAFTl-flow.pt` |

アップストリームはLの重みを個別には提供していません：`l`は`m`のチェックポイントを4回ではなく12回の反復で実行したものです。3つともアップストリームのTartan-C-T-TSKH-spring540x960チェックポイントです。`LibreSEARAFT`のデフォルトはサイズ`m`です。長辺が学習時のクロップ幅である960 px以下のフレームは、本来の解像度で8の倍数にパディングして処理されます。それより大きいフレームは縮小され、フィールドは元のサイズへ拡大し直されます。この上限は`imgsz`で変更できます。

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

SEA-RAFTは[Princeton Vision & Learning Lab](https://github.com/princeton-vl/SEA-RAFT)のWang、Lipson、Dengの成果です。アップストリームのコードのうち推論に必要な部分が、BSD-3-Clauseライセンスの下で同梱されています。アップストリームはSとMの重みを個別の重みライセンスなしで公開しています。LibreYOLOはそれらをリポジトリのBSD-3-Clauseライセンスの下で、アクセス制限なしでミラーしています。BSD-3-Clauseは、著作権表示、条件、免責事項を残す限り、商用利用およびクローズドソースでの利用を許可しています。

</provenance-box>

## 引用

<citation-block />
