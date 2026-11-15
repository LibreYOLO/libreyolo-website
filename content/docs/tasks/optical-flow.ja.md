---
title: オプティカルフロー
seo_title: LibreYOLOのオプティカルフロー推定
description: >-
  LibreYOLOでRAFT、SEA-RAFT、WAFT、FlowSeek、DPFlow、UniMatch、NeuFlow v2、MEMFOFを使い、動画フレーム間の密なオプティカルフローを推定し、.floファイルを保存し、エンドポイント誤差を検証します。
lead: >-
  オプティカルフローは、あるフレームの各ピクセルが次のフレームでどこへ移動するかを予測します。LibreYOLOは1.8.0の新機能であるflowタスクとして提供します：フレームの組を入力すると、ピクセル変位の密なフィールドが1つ出力され、8つのファミリーが同じ呼び出しで使えます。
keywords:
  - オプティカルフロー python
  - raft オプティカルフロー
  - sea-raft
  - 密な動き推定
  - flo ファイル
  - end-point error
  - 動画 モーション 推定
last_verified: "1.8.0"
snippets:
  predict:
    - label: 2枚のフレーム
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")
        results = model(["frame_0001.jpg", "frame_0002.jpg"])

        flow = results[0].flow
        print(flow.data.shape)           # (H, W, 2) float32、ピクセル単位の(dx, dy)
        print(flow.mean_magnitude)       # ピクセル単位の平均変位
    - label: 動画またはフレームのリスト
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNeuFlowv2s-flow.pt")

        # Nフレームから N - 1 個の結果。結果iはフレームiからi + 1へのフロー
        for result in model("video.mp4", stream=True, vid_stride=2):
            print(result.flow.max_magnitude)
    - label: 表示形式と生のフィールド
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRAFTl-flow.pt")
        frames = ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"]

        model(frames, save=True)                    # カラーホイール（デフォルト）
        model(frames, save=True, arrows=True)       # フレーム上に矢印
        model(frames, save=True, warp=True)         # ワープによる確認
        results = model(frames, save=True, save_flow=True)   # .floも出力

        results[0].flow.save("pair0.npy")           # .floまたは.npy
  cli:
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSEARAFTm-flow.pt source=frame_0001.jpg,frame_0002.jpg save=true
        libreyolo predict model=LibreSEARAFTm-flow.pt source=video.mp4 save=true arrows=true save_flow=true
        libreyolo val model=LibreSEARAFTm-flow.pt data=my_flow.yaml
  val:
    - label: 検証して指標キーを読み取る
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSEARAFTm-flow.pt")
        metrics = model.val(data="my_flow.yaml")

        print(metrics["metrics/epe"])      # 平均エンドポイント誤差、ピクセル単位
        print(metrics["metrics/px1"])      # 誤差が1 px未満のピクセルの割合
        print(metrics["metrics/px3"])
        print(metrics["metrics/fl_all"])   # KITTIの外れ値率
source_hash: ba3dde4eb92b3f1c
---

## 定義

`flow`タスクはフレームのシーケンスを受け取り、連続する各組について、フレーム`t`からフレーム`t + 1`への各ピクセルの変位を予測します。別名の`optical-flow`、`optical_flow`、`opticalflow`もこのタスクに解決され、チェックポイントには`-flow`の接尾辞が付きます。

推論結果は`result.flow`に入ります。これは最初のフレームの元のキャンバス上の`(H, W, 2)`のfloat32配列を保持する`FlowField`です。`flow.data[y, x]`はピクセル単位の`(dx, dy)`で、`+x`は右、`+y`は下です。`.dx`、`.dy`、`.magnitude`、`.angle`、`.max_magnitude`、`.mean_magnitude`でこれを読み取り、`.warp(image)`はフィールドを使って画像を逆方向にワープします。`flow.confidence`は`[0, 1]`の範囲の省略可能な`(H, W)`マップで、これを予測するファミリー（不確実性ヘッドを持つSEA-RAFT）だけが値を入れ、それ以外は`None`のままにします。`flow.next_frame`はワープ表示用に2枚目のフレームを保持します。`result.boxes`は`None`です。

## モデル

`flow`には8つのファミリーがあります。まだどれにもモデルページはないため、以下のスニペットのようにチェックポイント名で読み込んでください。

| ファミリー | サイズ | 重みのライセンス |
|---|---|---|
| SEA-RAFT | s, m, l | BSD-3-Clause |
| RAFT | s, l、およびSintelとKITTIのバリアント | BSD-3-Clause |
| NeuFlow v2 | s、およびSintelとThingsのバリアント | Apache-2.0 |
| WAFT | t, s, m, l | BSD-3-Clause。`t`はDINOv3を使うため、DINOv3 Licenseが適用されます |
| FlowSeek | t, s, m, l | tとsはApache-2.0。mとlは非商用（CC BY-NC、Depth Anything V2エンコーダー） |
| DPFlow | m、およびSintel、KITTI、Springのバリアント | 非商用（CC BY-NC-SA 4.0） |
| UniMatch | s, l、およびSintelとKITTIのバリアント | MIT |
| MEMFOF | m、および5つのバリアント | BSD-3-Clause |

チェックポイント名は`Libre<Family><size>-flow.pt`の形式で、たとえば`LibreSEARAFTm-flow.pt`、`LibreRAFTl-flow-kitti.pt`、`LibreNeuFlowv2s-flow.pt`です。非商用の重みは読み込み時に通知を表示します。NeuFlow v2は小型のエッジ向けモデルです。MEMFOFは1組あたり3枚のフレームを読み込み、ランナーはシーケンスの先頭と末尾で端のフレームを繰り返します。8つすべてがアップストリームのコードに対するパリティを記録済みで、重みはLibreYOLOのHugging Face organizationにミラーされています。

## 推論

<code-tabs name="predict" />

画像1枚だけでは例外になります：フローには少なくとも2枚のフレームが必要です。画像のリストまたはタプル、名前順に並べたディレクトリ、動画、ストリームはそれぞれ1つのシーケンスとして扱われ、`N`枚のフレームから`N - 1`個の結果が得られます。結果`i`はフレーム`i`からフレーム`i + 1`へのフローを持ち、`orig_img`はフレーム`i`です。動画では`vid_stride`でフレームの間隔を設定し、`stream=True`で1組ごとに1つの結果を返します。

表示形式は3つあります。デフォルトはMiddleburyのカラーホイールで、色相が方向、彩度が大きさを表し、動きがゼロの箇所は白になります。`arrows=True`は16ピクセルごとに1本の矢印を付けてフレームを描画します。`warp=True`はフレーム`t`と、それに向けて逆方向にワープしたフレーム`t + 1`を並べて表示し、フィールドが正しければ2つのパネルが一致します。`save=True`は選んだ表示を出力し、`save_flow=True`は各組の生のフィールドも`<frame stem>_<frame index>.flo`という名前のMiddlebury形式の`.flo`ファイルとして出力します。`FlowField.save(path)`は`.flo`または`.npy`を直接書き出します。

シェルからは、フレームをカンマ区切りの`source`として渡し、`arrows=true`、`warp=true`、`save_flow=true`を追加します：

<code-tabs name="cli" />

ソースとストリーミングについては[推論](/docs/predict)を、CLIのすべてのフラグについては[`libreyolo predict`](/docs/cli/predict)を参照してください。

## データセット形式

フローのデータセットはフレームのシーケンスの集まりです。フレームは`images/<split>/<sequence>/`の下に置かれ、ファイル名順に並べられます。1つのシーケンスディレクトリ内の連続する各組が1つのサンプルで、その正解データは`flows/`の下に、同じ相対パスと最初のフレームのstemで置かれます。

```text
dataset/
  data.yaml
  images/
    val/alley_1/frame_0001.png
    val/alley_1/frame_0002.png
    val/alley_1/frame_0003.png
  flows/
    val/alley_1/frame_0001.flo
    val/alley_1/frame_0002.flo
```

```yaml
path: dataset
val: images/val
flows_dir: flows
flow_format: flo
```

`flow_format`は`flo`（Middlebury、デフォルト）、`kitti`（青チャンネルを有効マスクとする16ビットRGB PNG）、`npy`（非有限値が無効を表す`(H, W, 2)`のfloat配列）のいずれかです。省略可能な組ごとのマスク`<stem>_valid.png`は、有効なピクセルを非ゼロで示します。`flow_valid_suffix`でこの接尾辞を変更できます。シーケンスの最後のフレームにはフローファイルがなく、異なるディレクトリのフレームが組になることはありません。

## 学習

フローは推論のみを提供します。フローのファミリーで`train()`を呼ぶと`NotImplementedError`が送出されます。

## 検証

`val()`は、有効なピクセル上で予測ベクトルと正解ベクトルを位置合わせなしで直接比較します。

<code-tabs name="val" />

`metrics/epe`は平均エンドポイント誤差で、予測ベクトルと真のベクトルのピクセル単位のユークリッド距離であり、低いほど良好です。`metrics/px1`と`metrics/px3`は、誤差が1ピクセル未満と3ピクセル未満の有効ピクセルの割合です。`metrics/fl_all`はKITTIの外れ値率で、誤差が3ピクセルと真の大きさの5パーセントの両方を超えるピクセルの割合です。品質とともに増加するため、ベストチェックポイントの評価値には`metrics/px1`を使います。

## エクスポート

フローのエクスポートは、あるファミリーで2入力のグラフとバックエンドの再スケール契約が実装されるまで、すべてのフォーマットでブロックされています。[エクスポート対応表](/docs/reference/export-matrix)を参照してください。
