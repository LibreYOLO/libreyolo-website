---
title: UniMatch
families: [unimatch]
seo_title: "PythonでUniMatch GMFlowのオプティカルフロー：グローバルマッチング、MIT"
description: "LibreYOLOでUniMatch（GMFlow）を実行し、グローバルマッチングで密なオプティカルフローを推定します。サイズsとlに加え、SintelとKITTIでのファインチューニング版があり、MITライセンスです。"
lead: "UniMatchは1つのトランスフォーマーマッチャーでオプティカルフロー、ステレオ、深度推定を統一するモデルで（TPAMI 2023）、GMFlowはそのフローモデルです。LibreYOLOはGMFlowのチェックポイントとともに、MITの下でオプティカルフローのタスクを提供します。"
keywords: [UniMatch, GMFlow, GMFlow+, オプティカルフロー, グローバルマッチング, 密 オプティカルフロー, flo ファイル]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreUniMatchl-flow.pt")   # 初回使用時に重みをダウンロード
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32 でピクセル単位の (dx, dy)
        print(flow.mean_magnitude)    # ピクセル単位の平均変位
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreUniMatchl-flow.pt")

        # N フレームから N - 1 個の結果 結果 i はフレーム i から i + 1 へのフロー
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreUniMatchl-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreUniMatchl-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreUniMatchl-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均エンドポイント誤差 ピクセル単位
        print(metrics["metrics/px1"])      # 誤差が 1 px 未満のピクセルの割合
        print(metrics["metrics/fl_all"])   # KITTI の外れ値率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreUniMatchl-flow.pt data=my_flow.yaml
source_hash: 9a4b34da82ff287d
---

GMFlowはオプティカルフローをグローバルマッチングの問題として扱います。トランスフォーマーが2枚のフレームの特徴量を比較し、すべての位置に対するsoftmaxでマッチを求め、必要に応じて局所的な回帰でリファインします。同梱しているアップストリームのコードにはステレオと深度の経路も含まれていますが、LibreYOLOで接続しているのはフローだけです。実世界の映像には、アップストリームの推奨どおり`l`を選んでください。

## インストール

UniMatchには追加のextraは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Face organizationからダウンロードされます。モデルを直接構築したい場合は、`LibreUniMatch`クラスも`libreyolo`からエクスポートされています。

<code-tabs name="predict" />

フローには少なくとも2枚のフレームが必要です：画像1枚だけでは例外になります。画像のリスト、名前順に並べたディレクトリ、動画、ストリームはそれぞれ1つのシーケンスで、`N`枚のフレームから`N - 1`個の結果が得られます。`result.flow`は、最初のフレームの元のキャンバス上の`(H, W, 2)`のfloat32配列を持つ`FlowField`です。`flow.data[y, x]`はピクセル単位の`(dx, dy)`で、`+x`は右、`+y`は下です。UniMatchは信頼度マップを出力しないため、`flow.confidence`は`None`です。`result.boxes`は`None`です。`save=True`はMiddleburyのカラーホイールを出力し、`arrows=True`または`warp=True`で他の表示を選び、`save_flow=True`は各組の生のフィールドも`.flo`ファイルとして出力します。ペイロードと表示の詳細は[オプティカルフローのタスク](/docs/tasks/optical-flow)ページで説明しています。

## バリアント

| サイズ | アップストリームのチェックポイント | ファイル |
| --- | --- | --- |
| `s` | `gmflow-scale1-things`：1/8でのグローバルマッチング（Chairs、続いてThings） | `LibreUniMatchs-flow.pt` |
| `l` | `gmflow-scale2-regrefine6-mixdata`：2スケールで6回のリファインメントステップを行うGMFlow+（標準） | `LibreUniMatchl-flow.pt` |
| `l` | Sintelでのファインチューニング | `LibreUniMatchl-flow-sintel.pt` |
| `l` | KITTI-2015でのファインチューニング | `LibreUniMatchl-flow-kitti.pt` |

SintelとKITTIのファインチューニング版があるのは`l`だけです。`LibreUniMatch`のデフォルトはサイズ`l`です。前処理はアップストリームの`inference_flow`を再現します：縦長のフレームは横長に転置され、両方のフレームは各辺がサイズごとのパディング係数（`s`は16、`l`は32）の倍数に切り上がるようにリサイズされます。フィールドは元のサイズへリサイズし直され、ベクトルも再スケールされます。アップストリームに対する意図的な修正が1つあります：縦長のフレームでは、アップストリームは転置を戻した後も`dx`と`dy`を入れ替えたままにしますが、LibreYOLOはフィールドがフローの契約に従うようにそれらを入れ替えます。

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

UniMatchは[Autonomous Vision Group](https://github.com/autonomousvision/unimatch)（テュービンゲン大学）の成果です。アップストリームの`unimatch`パッケージはMITライセンスの下でそのまま同梱されています。アップストリームはGMFlowのチェックポイントを個別の重みライセンスなしで公開しているため、LibreYOLOはそれらを公開元プロジェクトのMITライセンスの下でホストしています。これは黙示の許諾であり、LibreYOLOの解釈で、公開元による確認は得ていません。アクセス制限はありません。MITは、著作権表示とライセンス表示を複製物に添える限り、商用利用およびクローズドソースでの利用を許可しています。

</provenance-box>

## 引用

<citation-block />
