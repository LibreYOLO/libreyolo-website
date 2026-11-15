---
title: Video Depth Anything
families: [video_depth_anything]
seo_title: "PythonでVideo Depth Anything：長い動画でも一貫した深度"
description: "LibreYOLOのVideo Depth Anythingで、動画のフレーム間で時間的に一貫した相対深度を得ます。サイズはs、b、l。sはApache-2.0、bとlは非商用です。"
lead: "Video Depth AnythingはByteDanceによるもので、Depth Anything V2のエンコーダーに時間方向のヘッドを載せ、フレーム間で深度が一貫するようにしています。LibreYOLOはフレームのリスト、フォルダー、動画、ストリームを1つのシーケンスとして処理し、すべて同じ相対スケールでフレームごとに1つの深度マップを返します。"
keywords: [Video Depth Anything, 動画 深度推定, 時間的に一貫した 深度, Depth Anything V2, 相対深度, DINOv2]
last_verified: "1.8.0"
snippets:
  predict:
    - label: 動画
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVideoDepthAnythings-depth.pt")

        # フレームごとに1結果、遅延は最大1ウィンドウでストリーミング
        for result in model("clip.mp4", stream=True):
            depth = result.depth_map.data      # (H, W)、相対逆深度
    - label: 1枚の画像
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVideoDepthAnythings-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)   # 1フレームのシーケンス
        print(result.depth_map.data.shape)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreVideoDepthAnythings-depth.pt source=clip.mp4 save=True
source_hash: cfd2451ca039b9df
---

## インストール

Video Depth Anythingにオプションのextraは不要です。

```bash
pip install libreyolo
```

## 推論

重みは初回使用時にLibreYOLOのHugging Faceミラーからダウンロードされます。

<code-tabs name="predict" />

複数画像のソースはすべて1つのシーケンスになります。フレームのリスト、名前順に並べたディレクトリ、動画ファイル、ストリームです。`N`フレームから`N`個の結果が得られ、それぞれがそのフレームのキャンバス上の`result.depth_map`を持ちます。内部では、フレームはアップストリームの重なり合う32フレームのウィンドウで処理され、各ウィンドウはそれ以前のウィンドウにスケールとシフトを合わせるため、シーケンス全体が1つの相対スケールを共有します。1枚の画像は1フレームのシーケンスで、アップストリームはそれを自身のコピーで1ウィンドウ分に埋めます。

出力はLibreYOLOの共有の深度の契約に従います。相対逆深度で、値が大きいほど近く、値にメートル単位はありません。入力サイズはすべてのエンコーダーでアップストリームのデフォルトの518です。16:9より横長のフレームは、アップストリームと同じく比例して小さく処理されます。ペイロードについては[深度推定](/docs/tasks/depth-estimation)のタスクページを参照してください。

## バリアント

エンコーダーのサイズは`s`、`b`、`l`の3つで、それぞれ1つのチェックポイントがあります。ライセンスはサイズによって異なるため（下記参照）、商用製品で使うのは`s`です。アップストリームのメートル深度のチェックポイントはホストしておらず、変換スクリプトはそれらを受け付けません。

学習は提供していません。`train()`は例外を発生させます。アップストリームの相対深度のチェックポイントは、ライブラリの`weights/convert_video_depth_anything_weights.py`で変換してください。

## エクスポート

<export-matrix />

エクスポートには対応していません。ネットワークは32フレームのウィンドウを読み、ウィンドウはホスト側でつなぎ合わされますが、単一画像の深度エクスポートの契約はこれを記述しないため、`export()`は例外を発生させます。

## チェックポイント

<checkpoint-table />

## ライセンス

<provenance-box>

Video Depth AnythingはByteDanceによるものです（[DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)）。同梱のコードはApache-2.0です。重みのライセンスはサイズによって異なります。

- `s`（Video-Depth-Anything-Small）：Apache-2.0、商用利用が可能です。
- `b`と`l`（BaseとLarge）：CC BY-NC 4.0、**非商用のみ**です。非商用の通知が同梱されています。

LibreYOLOは変換したチェックポイントを、テンソルを変更せずに`LibreYOLO/LibreVideoDepthAnything{s,b,l}-depth`としてミラーしており、ゲートされていません。

</provenance-box>

## 引用

<citation-block />
