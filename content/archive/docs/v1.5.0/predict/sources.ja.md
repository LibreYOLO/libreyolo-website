---
title: 推論ソース
seo_title: LibreYOLOの推論ソース
description: >-
  predictが受け付けるすべてのソースを説明します。画像、フォルダー、URL、動画ファイル、Webカメラ、RTSP、YouTube、画面キャプチャ、画像リスト、.streamsファイルに対応します。
lead: >-
  source引数は何かを開く前に分類されるため、1つの呼び出しでJPEG、フォルダー、MP4、Webカメラ番号、RTSP
  URL、画面領域、カメラのリストを処理できます。
keywords:
  - yolo 動画 推論 python
  - rtsp
  - Webカメラ 物体検出 python
  - フォルダー 画像 一括推論
  - 画面キャプチャ 物体検出
  - 複数 rtsp ストリーム
  - streams ファイル
  - youtube 推論
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  ソース分類はlibreyolo/utils/source.py（classify_source、SourceKind、StreamSource、MultiStreamSource）で確認しました。受け付ける画像型とディレクトリ拡張子はlibreyolo/utils/image_loader.pyで確認しました。動画拡張子と保存先はlibreyolo/utils/video.pyで確認しました。画面構文はlibreyolo/utils/screen.pyで確認しました。返り値の形状と引数のデフォルトはlibreyolo/models/base/inference.pyのInferenceRunner.__call__で確認しました。
snippets:
  images:
    - label: 1枚の画像
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # 単一画像ソースはリストではなく1つの Results を返す
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: メモリ内の画像
      language: python
      code: |
        import numpy as np
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        pil_image = Image.open(SAMPLE_IMAGE)
        array = np.asarray(pil_image)
        raw_bytes = open(SAMPLE_IMAGE, "rb").read()

        for source in (pil_image, array, raw_bytes):
            result = model(source)
            print(type(source).__name__, len(result.boxes))
    - label: フォルダー
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("sample_folder")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(3):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # フォルダーはパス順に画像ごと1つの Results を含むリストを返す
        results = model(str(folder))
        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: 動画ファイル（独自の動画を指定）
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # clip.mp4 をディスク上の動画ファイルに置き換える
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: 3フレームごとにディスクへ書き込み
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Webカメラ（接続されたカメラが必要）
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Webカメラ番号 0 ライブソースは終了しないためループを制限
        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP（接続可能なカメラURLが必要）
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: .streamsファイル（独自のカメラを指定）
      language: python
      code: >
        import itertools

        from pathlib import Path


        from libreyolo import LibreYOLO


        Path("cameras.streams").write_text(
            "# one source per line, blank lines and comments are skipped\n"
            "rtsp://192.168.1.64:554/Streaming/Channels/101\n"
            "rtsp://192.168.1.65:554/Streaming/Channels/101\n",
            encoding="utf-8",
        )


        model = LibreYOLO("LibreYOLO9s.pt")

        for result in itertools.islice(model("cameras.streams", stream=True),
        100):
            print(result.frame_idx, len(result.boxes))
    - label: カメラのリスト
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: 1枚のスクリーンショット（mssとデスクトップセッションが必要）
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # stream=True を指定しない場合は1フレームだけ取得
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: 1台のモニターの領域を継続的に取得
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # "screen <monitor> <left> <top> <width> <height>"

        for result in itertools.islice(model("screen 1 100 200 512 256",
        stream=True), 50):
            print(len(result.boxes))
source_hash: c371965951dd0181
---

## ソースの分類方法

`classify_source`は何かを開く、またはダウンロードする前に、次の順序で値を調べます。最初に一致した規則が適用されます。

| ソース | 読み取り方法 |
|---|---|
| `"screen"`、`"screen 1"`、`"screen 1 100 200 512 256"` | 画面キャプチャ |
| 0以上の`int`、または同名のファイルがない数字だけの文字列 | Webカメラ |
| `rtsp://`、`rtmp://`、`tcp://`、`udp://`のURL | ネットワークストリーム |
| パスが`.m3u8`で終わる`http(s)://` URL | ネットワークストリーム |
| YouTubeページのURL | ネットワークストリーム |
| 全項目がライブソースまたは動画であるリストかタプル | 複数のライブストリーム |
| そのほかのリストまたはタプル | 画像のバッチ |
| `.streams`で終わるパス | 複数のライブストリーム |
| 動画拡張子を持つパス | 動画ファイル |
| 既存のディレクトリ | 画像フォルダー |
| そのほか | 単一画像 |

ライブソースと画像が混在するリストでは`TypeError`が発生します。負のWebカメラ番号では`ValueError`が発生します。

分類器はネットワークに一切アクセスしないため、入力ミスのあるURLは`predict`の呼び出し時ではなく、キャプチャを開くときに明らかになります。

## 画像

<code-tabs name="images" />

単一画像ソースは7種類の型を受け付けます。

| 型 | 読み取り方法 |
|---|---|
| `str`または`pathlib.Path` | ローカルファイル、`http(s)://`、`s3://`、`gs://` |
| `PIL.Image.Image` | RGBへ変換 |
| `numpy.ndarray` | 2Dグレースケール、または3DのHWCかCHW。4D配列では最初の画像を使用 |
| `torch.Tensor` | CHWまたはNCHWをRGBとして読み取り。バッチテンソルでは最初の画像を使用 |
| `bytes` | エンコードされた画像データ |
| `io.BytesIO` | エンコードされた画像データ |

前処理の前にすべてRGBへ変換されます。チャンネル順序が曖昧なのはNumPy配列だけなので、`color_format`で制御します。デフォルトの`"auto"`は配列をそのまま維持し、`"bgr"`はチャンネルを反転します。OpenCVで読み取ったフレームには後者が必要です。

浮動小数点配列は自身の範囲に基づいて再スケーリングされます。`1.0`以下の値は255倍され、それより大きい値は`[0, 255]`内にクリップされます。RGBA配列ではアルファチャンネルを削除します。

リモートパスにはそれぞれ1つのパッケージが必要で、どれもデフォルトではインストールされません。`http(s)://`には`requests`、`s3://`には`boto3`、`gs://`には`gcsfs`が必要です。

## フォルダー

ディレクトリは再帰的に走査して並べ替えられ、次のいずれかのサフィックスを持つすべてのファイルが画像になります。`.jpg`、`.jpeg`、`.png`、`.gif`、`.webp`、`.bmp`、`.tiff`、`.tif`です。フォルダー内のそのほかのものはスキップされます。空のフォルダーでは例外を発生させず、空のリストを返します。

フォルダーとリストは`batch`を受け付ける2種類のソースです。対応するファミリーでは、チャンクごとに積み重ねた順伝播を1回実行します。[推論性能](/docs/predict/performance)を参照してください。

## 動画ファイル

<code-tabs name="video" />

パスのサフィックスが`.asf`、`.avi`、`.gif`、`.m4v`、`.mkv`、`.mov`、`.mp4`、`.mpeg`、`.mpg`、`.ts`、`.wmv`、`.webm`のいずれかなら、動画として扱われます。

`.gif`は両方の一覧に含まれます。動画の確認が先に行われるため、`.gif`パスを直接`predict`に渡すと動画として開かれます。走査対象フォルダー内の`.gif`は静止画像として読み込まれます。

`vid_stride`はN番目ごとのフレームを処理し、デフォルトは`1`です。`stream=True`を指定しないと動画全体が1つのリストへデコードされ、間引き後に500フレームを超える場合は`stream=True`を推奨する警告が表示されます。

動画から得られる各`Results`は`frame_idx`を持ちます。

## Webカメラ、ネットワークストリーム、YouTube

<code-tabs name="live" />

ライブソースには終端がないため、`stream=True`が必要です。指定しない場合、`predict`は無限のリストを収集しようとせず`ValueError`を発生させます。

フレームはキャプチャごとに1つのバックグラウンドスレッドで読み取られます。デフォルトではキューに最新フレームだけを保持するため、モデルがカメラより遅い場合は遅延が積み上がらずにフレームをスキップします。`stream_buffer=True`ではキャプチャしたすべてのフレームを保持し、フレームを維持する代わりにレイテンシが増大します。

Webカメラ番号は`int`または数字の文字列です。Windowsでは最初にDirectShowバックエンドでキャプチャを開き、失敗した場合はデフォルトのバックエンドにフォールバックします。

YouTubeページのURLは動画をダウンロードせずに直接メディアURLへ解決され、そのために`yt-dlp`が必要です。

```bash
pip install "libreyolo[stream]"
```

ストリームのラベルは、ログへ記録する、またはファイル名に使う前に機密部分が伏せられます。認証情報を含むURLは`user:***@host`として表示され、署名付きURLとbearer tokenが含まれるため、直接ストリームのラベルからクエリ文字列が削除されます。YouTubeの動画IDは認証情報ではないため維持されます。

## 複数のカメラを同時に処理

<code-tabs name="streams" />

`.streams`ファイルでは1行に1つのソースを記述します。空行と`#`で始まる行は無視されます。残る各行はWebカメラ番号、ネットワークストリーム、YouTube URL、動画ファイルのパスのいずれかでなければなりません。それ以外の場合は行番号を示して`ValueError`が発生します。空のファイルでは、カメラなしで開始せず例外が発生します。

ライブソースのリストまたはタプルを使うと、ファイルなしで同じ処理を行えます。

各キャプチャに固有のスレッドが割り当てられ、すべてのフレームが1つのジェネレーターへ多重化されます。各処理ではアクティブなストリームをポーリングし、準備ができたものを生成します。そのため、遅いカメラが速いカメラを待たせず、異なるカメラのフレームが交互に現れます。終了したストリームはローテーションから外れ、そのほかは続行します。

## 画面キャプチャ

<code-tabs name="screen" />

画面ソースは`screen`という単語の後に0個、1個、4個、5個の整数を続けます。それ以外の個数では`ValueError`が発生します。

| 形式 | キャプチャ対象 |
|---|---|
| `"screen"` | すべてのモニターを統合 |
| `"screen 1"` | モニター1 |
| `"screen 100 200 512 256"` | 統合デスクトップ上のボックス |
| `"screen 1 100 200 512 256"` | モニター1上のボックス |

ボックス座標は`left top width height`で、選択したモニターの左上隅を基準とします。画面ソースはFPSを30÷`vid_stride`として報告し、これは保存する動画のFPSになります。キャプチャには`mss`パッケージが必要です。

```bash
pip install mss
```

`stream=True`を指定しない場合、画面ソースは1フレームを取得して1つの`Results`を返します。これは画像ファイルで推論する場合のスクリーンショット版です。`stream=True`を指定すると、ループが中断されるまでキャプチャします。

## predictの返り値

返り値の形はソースと`stream`によって異なります。

| ソース | `stream=False` | `stream=True` |
|---|---|---|
| 単一画像 | 1つの`Results` | 1つの`Results`を生成するジェネレーター |
| 画像のリスト | `Results`のリスト | ジェネレーター |
| フォルダー | `Results`のリスト | ジェネレーター |
| 動画ファイル | `Results`のリスト | ジェネレーター |
| 画面 | 1つの`Results` | 終端のないジェネレーター |
| Webカメラ、ネットワークストリーム、`.streams` | `ValueError` | 終端のないジェネレーター |

単一画像は`Results`オブジェクト自体を返します。インデックス付けを行うと画像ではなく検出を選択するため、単一画像の推論結果に対する`result[0]`は最初の画像ではなく最初のボックスです。これらのオブジェクトに含まれる内容については[結果の操作](/docs/predict/results)を参照してください。

## saveの書き込み先

`save=True`はアノテーション済み出力を返さず、runディレクトリの下に書き込みます。

画像は自動的に番号が増える`runs/detect/predict`、`runs/detect/predict2`などへ、ソースのファイル名を維持して保存されます。1つのプロセス内の画像はすべて同じディレクトリに入るため、2つの入力フォルダーに同じファイル名があると互いに上書きします。メモリ内画像には再利用できるファイル名がないため、`image0`、`image1`のように番号が付きます。

動画とライブソースは、ソースに基づく名前を持つ1つの`.mp4`として書き込まれます。

`output_path`はディレクトリを上書きします。サフィックスを持つパスはファイル、持たないパスはディレクトリとして扱われます。`output_file_format`は静止画像のエンコードを選択し、`jpg`、`png`、`webp`を受け付けます。

保存後、書き込まれたパスは`result.saved_path`として結果にも追加されます。

