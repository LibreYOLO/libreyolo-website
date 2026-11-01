---
title: トラブルシューティング
seo_title: LibreYOLOでよくあるエラーの解決方法
description: LibreYOLOで頻繁に発生するエラー、それぞれの意味、修正方法を説明します。例外を送出せずに誤った出力を生成する2つの問題も含みます。
lead: 表示されるメッセージ別にエラーをまとめています。最後の2項目では逆の問題を扱います。コードは動作し、もっともらしいものを返しますが、その結果は誤っています。
keywords:
  - libreyolo エラー
  - modulenotfounderror libreyolo
  - libreyolo cuda out of memory
  - libreyolo notimplementederror
  - libreyolo トラブルシューティング
last_verified: 1.6.0
source_hash: 9644a377afc524f9
---

表示されるテキスト別にエラーをまとめています。該当するメッセージがない場合は、[FAQ](/docs/faq)で障害以外の疑問を確認できます。また、`libreyolo models` で現在のインストールが実際に読み込めるものを確認できます。

## インポートしていないパッケージを示すModuleNotFoundError

一部のファミリーには任意の追加パッケージが必要です。メッセージには追加パッケージ名ではなく、不足しているパッケージ名が表示されるため、トレースバックから修正方法が分かるとは限りません。

`libreyolo models` を実行してください。依存関係が不足しているファミリーごとに、有効化するための正確なpipコマンドが表示されるため、パッケージから追加パッケージへの対応を自身で調べる必要はありません。`libreyolo models --json` では同じ内容をオブジェクトとして出力します。

RF-DETRとDINOv2には`libreyolo[rfdetr]`が必要で、そのメッセージには追加パッケージ名が表示されます。これがない場合、`LibreYOLO("LibreRFDETRn.pt")`、DINOv2の重み名、`libreyolo predict model=rfdetr-n`はインストール用の行を表示して停止し、CLIは`model_load_failed`で終了します。

```
ModuleNotFoundError: RF-DETR support requires extra dependencies.
Install with: pip install libreyolo[rfdetr]
```

[インストールページ](/docs/install)には、すべての追加パッケージとその対象が記載されています。

## ONNX推論にはonnxruntimeが必要

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

どのランタイムが必要かはハードウェアによって決まるため、基本パッケージはランタイムに依存しません。CPU用には `onnxruntime`、CUDA用には `onnxruntime-gpu` をインストールしてください。どちらも同じ `onnxruntime` モジュールを提供するため、両方ではなく片方だけをインストールします。

## ONNXモデルが見つからない

```
FileNotFoundError: ONNX model not found: <path>
```

パスはスクリプトではなく、作業ディレクトリを基準に解決されます。エクスポート先が黙って別の場所になった場合にも、このエラーが表示されます。`export()` は書き込んだパスを返すため、名前を推測せずに戻り値を取得してください。

## train()からのNotImplementedError

すべてのファミリーが学習できるわけではありません。一部は推論、検証、エクスポート専用に移植されており、実行したように見せかけずに `train()` から例外を送出します。

[FAQの項目](/docs/faq)でその理由を説明しています。学習スクリプトを書く前に特定のファミリーを確認するには、そのモデルページで学習の可否を確認してください。

## export()からのNotImplementedError

ファミリーがタスクに対応していても、エクスポートには対応していない場合があります。よく遭遇するのがEoMTです。`export()` はsemanticタスクを受け付けますが、`segment` と `panoptic` では例外を送出します。これらに必要なquery-maskランタイム仕様が定義されていないためです。

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

各ファミリーのページには、検証済みのタスクと形式の組み合わせを示すエクスポートマトリックスがあります。

## CUDAのメモリ不足

まず `batch` を減らし、次に `imgsz` を減らしてください。どちらもおおむねサイズに応じてメモリ量を変えますが、モデルが見る内容を変えずに減らせるのはバッチです。

学習ではなく検証時に失敗する場合、検証は独自のバッチサイズで実行されるため、そちらも減らしてください。

Windowsでは、ディスプレイ用GPUにメモリ不足とは別の障害モードがあります。応答がタイムアウト時間より長く止まったGPUをドライバーがリセットし、実行中の処理を終了させます。このとき、メモリ不足ではなくランダムなCUDAエラーに見えます。モニターを駆動するカード上で長時間動くカーネルは、この問題を引き起こす可能性があります。

## 重みをダウンロードできない

重みは初回使用時にHugging Faceから取得され、ローカルにキャッシュされます。[FAQ](/docs/faq)では、キャッシュの保存場所と完全オフラインで実行する方法を説明しています。

ダウンロードで404が発生した場合は、渡したファイル名を確認してください。URLはタスク接尾辞を含むファイル名から生成されるため、公開済みチェックポイントと一致しない名前を使うと、存在しないURLが生成されます。各モデルページのチェックポイント表には、公開されている正確なファイル名が記載されています。

## スクリプトのトップレベルでtrain()またはval()が呼び出された

```
train() is called at the top level of your script, and DataLoader workers re-run that script on this platform, so it runs with workers=0. Put the call under `if __name__ == "__main__":` to load data in parallel.
```

Windowsには `fork` がなく、macOSのPythonもデフォルトでは使わないため、dataloaderのworkerはスクリプトを再インポートして起動します。`spawn`と`forkserver`の起動方式は、どのプラットフォームでも同じ動作をします。`if __name__ == "__main__":`ガードがないと、各workerが`train()`または`val()`の呼び出しをもう一度実行することになります。1.7.1以降、LibreYOLOはこれを事前に検知し、別の値を渡していても、その呼び出しを代わりに `workers=0` で実行して、上の行を一度だけログに出力します。学習は動作しますが、データはメインプロセスで読み込まれるため遅くなります。1.7.0では、同じスクリプトはガードを追加するか `workers=0` を渡すよう促すエラーで停止していました。

並列workerを戻すには、呼び出しをガードの下に置いてください。

```python
def main():
    ...  # モデルを構築してtrain()を呼び出す

if __name__ == "__main__":
    main()
```

ノートブック、CLI、およびworkerが `fork` で起動するLinuxは影響を受けません。

## 例外を送出しない2つの障害

このページのほかの部分ではエラーを扱っています。次の2つは、コードが実行されて正しそうなものが返るため、さらに深刻です。

### 1つの結果に対するインデックス指定

`predict()` は、画像が1枚なら1つの `Results`、複数ならリストを返します。1枚画像の戻り値にインデックスを指定すると、画像ではなく1件の*検出結果*が選択されます。

```python
result = model.predict("image.jpg")   # 1つのResults
result.boxes                          # すべての検出結果、正しい
result[0].boxes                       # 1件だけの検出結果、通知なし
```

`Results` へのインデックス指定は部分集合を返す有効な操作なので、例外は送出されません。リスト形式を前提に書いたコードは、画像ごとに1つのボックスだけを黙って報告します。リストであると分かっているものだけにインデックスを指定してください。

### 指標を属性として読み取る

`val()` は指標名をキーとする辞書を返します。唯一の属性である `box` は、指標の値ではなく、画像ごとの結果とクラスごとのしきい値を保持します。

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # 正しい
metrics.box.map               # AttributeError
```

キーは `metrics/` と `speed/` で名前空間化されています。タスクによって項目が異なるため、辞書を一度出力し、そのタスクが生成した内容を確認してください。

## 学習前にデータセットを確認

学習失敗のほとんどはデータセットの問題です。`libreyolo doctor data.yaml` は検出データセット全体の健全性チェックを実行し、重大度別に所見を報告します。これは最初のエポックのトレースバックを読むより高速です。

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

チェック項目の一覧については[doctorコマンド](/docs/cli/doctor)を参照してください。


## 永続ワーカーがデータセットの変更を認識できない

有効な`close_mosaic`または`set_epoch`フックは、ワーカーが使うデータセットのコピーに届く必要があります。独自の複数ワーカーローダーが互換性のない永続コピーを保持する場合は、`persistent_workers=False`を使うか、変更後にワーカーを再構築してください。デフォルトの非永続経路には影響しません。
