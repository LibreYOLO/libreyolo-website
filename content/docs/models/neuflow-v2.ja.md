---
title: NeuFlow v2
families: [neuflowv2]
seo_title: "PythonでNeuFlow v2のオプティカルフロー：9Mパラメータのエッジ向けモデル、Apache-2.0"
description: "LibreYOLOでNeuFlow v2を実行し、エッジデバイス上で高速に密なオプティカルフローを推定します。9Mパラメータのネットワーク1つに、mixed、Sintel、Thingsのチェックポイントがあり、Apache-2.0です。"
lead: "NeuFlow v2はNortheastern University Field Robotics Labによる高効率なオプティカルフローネットワークで、エッジデバイス向けに作られています。パラメータは約9Mで、固定サイズのキャンバス上で動作します。LibreYOLOのフローファミリーの中のエッジ向けモデルです。"
keywords: [NeuFlow v2, NeuFlow, オプティカルフロー, エッジ オプティカルフロー, リアルタイム オプティカルフロー, 密 オプティカルフロー, flo ファイル]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")   # 初回使用時に重みをダウンロード
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)        # (H, W, 2) float32 でピクセル単位の (dx, dy)
        print(flow.mean_magnitude)    # ピクセル単位の平均変位
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")

        # N フレームから N - 1 個の結果 結果 i はフレーム i から i + 1 へのフロー
        for result in model("video.mp4", stream=True, save=True, save_flow=True):
            print(result.flow.max_magnitude)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreNeuFlowv2s-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreNeuFlowv2s-flow.pt source=video.mp4 save=true arrows=true save_flow=true
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均エンドポイント誤差 ピクセル単位
        print(metrics["metrics/px1"])      # 誤差が 1 px 未満のピクセルの割合
        print(metrics["metrics/fl_all"])   # KITTI の外れ値率
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNeuFlowv2s-flow.pt data=my_flow.yaml
source_hash: 22669af5be77f2b3
---

NeuFlow v2は浅いCNNエンコーダーに2つのクロスアテンション層と1/16スケールでのグローバルなsoftmaxマッチングを組み合わせ、続いて1/16および1/8スケールでの軽量な畳み込みリファイナーと、学習された8倍のconvexアップサンプラーを使います。精度の最後の一押しよりも速度とメモリが重要な場合、たとえばフレームレートでの動画処理や小型のハードウェアで選んでください。

## インストール

NeuFlow v2には追加のextraは不要です。インポートするものはすべて基本インストールに含まれています。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Face organizationからダウンロードされます。モデルを直接構築したい場合は、`LibreNeuFlowv2`クラスも`libreyolo`からエクスポートされています。

<code-tabs name="predict" />

フローには少なくとも2枚のフレームが必要です：画像1枚だけでは例外になります。画像のリスト、名前順に並べたディレクトリ、動画、ストリームはそれぞれ1つのシーケンスで、`N`枚のフレームから`N - 1`個の結果が得られます。`result.flow`は、最初のフレームの元のキャンバス上の`(H, W, 2)`のfloat32配列を持つ`FlowField`です。`flow.data[y, x]`はピクセル単位の`(dx, dy)`で、`+x`は右、`+y`は下です。NeuFlow v2は信頼度マップを出力しないため、`flow.confidence`は`None`です。`result.boxes`は`None`です。`save=True`はMiddleburyのカラーホイールを出力し、`arrows=True`または`warp=True`で他の表示を選び、`save_flow=True`は各組の生のフィールドも`.flo`ファイルとして出力します。ペイロードと表示の詳細は[オプティカルフローのタスク](/docs/tasks/optical-flow)ページで説明しています。

## バリアント

アーキテクチャは1つでサイズは`s`、同じネットワークに学習データの異なる3つのバリアントがあります。

| ファイル | アップストリームのチェックポイント |
| --- | --- |
| `LibreNeuFlowv2s-flow.pt` | `neuflow_mixed.pth`、複数のデータセットで学習（標準） |
| `LibreNeuFlowv2s-flow-sintel.pt` | `neuflow_sintel.pth` |
| `LibreNeuFlowv2s-flow-things.pt` | `neuflow_things.pth` |

他のフローファミリーとは異なり、NeuFlow v2は本来の解像度では動作しません。両方のフレームは固定サイズのキャンバスへ引き伸ばしてリサイズされ、デフォルトはアップストリームの`infer.py`と同じく432x768（高さ×幅）です。`imgsz`には整数または`(height, width)`の組を指定でき、各辺は16の倍数です。ネットワークは1/16で1回、1/8で8回のリファインメントステップを実行し、フィールドは元のキャンバスへリサイズし直され、ベクトルの各成分はその軸のリサイズ比率で乗算されます。

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

NeuFlow v2は[Northeastern University Field Robotics Lab](https://github.com/neufieldrobotics/NeuFlow_v2)の成果です。アップストリームのモデルコードはApache-2.0ライセンスの下で同梱されています。アップストリームは3つのチェックポイントを個別の重みライセンスなしで同じApache-2.0のリポジトリにコミットしているため、LibreYOLOはそれらをApache-2.0の下で、アクセス制限なしでミラーしています。Apache-2.0は商用利用およびクローズドソースでの利用を許可しています。再配布の際はライセンス文と帰属表示を残してください。

</provenance-box>

## 引用

<citation-block />
