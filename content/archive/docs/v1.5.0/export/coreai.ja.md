---
title: Core AI
seo_title: LibreYOLOからApple Core AIへエクスポートする
description: >-
  LibreYOLOモデルをApple Core
  AIの.aimodelアセットにエクスポートします：macOS専用、固定キャンバス、FP32、そして利用側が守らなければならない名前付き出力の順序の取り決め。
lead: >-
  Core AIはAppleのオンデバイス推論スタックです。LibreYOLOはtorch.exportでモデルをキャプチャし、Core
  AIコンバーターを通してローワリングし、モデルのメタデータとエクスポートされた出力名を持つ.aimodelアセットを書き出します。
keywords:
  - libreyolo core ai エクスポート
  - aimodel
  - coreai-torch
  - torch.export apple
  - apple オンデバイス推論
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: フラグ
    value: export(format="coreai")
    mono: true
  - label: 出力
    value: メタデータを付与した.aimodelアセット1つ
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: 再読み込み
    value: LibreYOLO経由ではありません。利用側はCore AIランタイムを直接使います。
  - label: 形状
    value: 固定キャンバス。dynamic=TrueはNotImplementedErrorを送出します。
  - label: 精度
    value: FP32のみ。half=Trueとint8=Trueは拒否されます。
  - label: 要件
    value: macOS。ツールチェーンは他の環境では変換も実行もできず、coreai-torchはtorchを2.11.xに固定します。
verification: >-
  devブランチのlibreyolo/export/coreai.py、libreyolo/export/coreai_compat.py、libreyolo/export/exporter.py、libreyolo/export/support.pyおよびpyproject.tomlから読み取りました。
snippets:
  install:
    - label: macOSでのインストール
      language: bash
      code: |
        # すべての集約extraから意図的に除外。coreai-torchがtorchを2.11.xに固定し、
        # 環境全体をそのバージョンに引きずり込むため
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # weights/LibreYOLO9t.aimodelを書き出す
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: 引数
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int、または(height, width)。これが実行キャンバス
            batch=1,
            output_path=None, # Noneならweights/<stem>.aimodelに書き出す
        )

        # dynamic=TrueはNotImplementedErrorを送出
        # half=Trueとint8=Trueは検証時に拒否される
  outputs:
    - label: 利用側を組み込む前に出力の順序を確認
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="coreai", imgsz=640)

        # アセットのメタデータは、エクスポートされた出力名をグラフ順で
        # "coreai_output_names"に記録する。Core AIが返す辞書はそのリストを使って
        # 名前でマッピングし、eagerのタプルと位置で対応づけないこと
  support:
    - label: エクスポート前に1つのファミリーとタスクを確認
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## インストール

このフォーマットはmacOS専用です。`coreai-torch`の依存指定には
`sys_platform == 'darwin'`マーカーが付いており、ツールチェーンは他のどこでも変換も
実行もできません。

<code-tabs name="install" />

このextraは、`coreai-torch`がtorchを2.11系に固定するため、`libreyolo[all]`を含む
すべての集約extraの外に置かれています。その組み合わせに縛られても構わない環境に
インストールしてください。

## エクスポート

<code-tabs name="export" />

キャプチャは`torch.export`であり、単一の記録されたトレースではなく、ガードを伴う
本物のグラフキャプチャです。これはCore MLの経路より厳格です：ホスト側のスカラー
読み取りやデータ依存の制御フローは黙って焼き込まれるのではなく拒否されるため、
いくつかのファミリーはキャプチャ失敗が記録されたうえでここではブロックされて
います。

3つの準備ステップは、エクスポートが成功しても失敗しても呼び出し元の実モデルを
復元するスコープの中で実行されます。Darknet系のファミリーでは、Core AI 0.4.1が
Darknetの平方根の後にイプシロンを足す式を保持しないため、推論時のバッチ正規化が
直前の畳み込みに厳密に畳み込まれます。グリッド系とアンカー系のファミリーでは、
固定キャンバス向けにアンカーが凍結されます。RF-DETRでは、コンバーターに
`aten._upsample_bicubic2d_aa`のローワリングがないため、モデル自身のベイク経路を
再実行して位置埋め込みが要求されたキャンバス向けに再ベイクされます。

ローワリングでは、Core AIコンバーターにDETR系ファミリーが使うデフォーマブル
アテンションのサンプラーのローワリングがないため、`aten.grid_sampler_2d`に対する
PyTorchの参照分解を分解テーブルに組み込みます。

アセットは最小OSとしてv27を宣言しますが、これはツールチェーンが提供する唯一の値
です。これが制限するのはデプロイであって変換ではありません：変換とPython側の実行
はwheelに含まれるランタイムを通じてより古いmacOSでも動作しますが、数値はOSの
バージョン間で異なるため、記録されたパリティはmacOS 27で測定しています。

## アーティファクトを実行する

`libreyolo/backends`にCore AIのエントリはないため、`LibreYOLO()`は`.aimodel`を
読み込みません。利用側はCore AIランタイムを直接使い、前処理、デコード、NMS、座標の
リスケールは利用側の担当です。サポートマトリクスで検証済みとなっている行は、
エクスポートされたグラフが参照と同じ数値を計算するという主張であって、`predict`が
それを実行するという主張ではありません。

利用側が自力で再導出できない唯一のものが、出力の順序です：

<code-tabs name="outputs" />

Core AIが返すのは名前付きの辞書で、そのキーの順序はeager forwardのタプル順とも、
推測できる何かとも一致しません。エクスポートされた名前がまさにこの理由でアセットの
メタデータに`coreai_output_names`として書き込まれます。名前でマッピングしてくだ
さい。

## 制約

固定キャンバス、FP32、バッチはエクスポート時のまま。`dynamic=True`は
`NotImplementedError`を送出し、`half=True`と`int8=True`は検証時に拒否されます。

変換側の対応範囲は広いです。検証済みの組み合わせには、YOLO9系ファミリー、YOLOX、
YOLO7、Darknet時代の4つの検出器、YOLO-NAS、PicoDet、RTMDet、RT-DETR、RT-DETRv2、
RT-DETRv4、D-FINE、DEIM、DEIMv2、EC、RF-DETRの物体検出、4つのCNN分類ファミリーと
クラス固定のCLIPおよびSigLIP2、Depth Anything V2とZipDepth、NAFNetとReal-ESRGANの
復元、PIDNetとLingBotVisionのセマンティックセグメンテーション、そしてFOMOの点検出が
含まれます。それぞれに独自の記録済みコンテキストがあり、`libreyolo formats`がそれを
出力します。

ブロックされているものと、組み合わせごとに記録された理由：

| 組み合わせ | 理由 |
|---|---|
| EoMTのセマンティックセグメンテーション | 厳格なキャプチャが`GuardOnDataDependentSymNode`で失敗します：マスク経路のどこかがテンソルから値を読み取り、それで分岐しています |
| SegFormerのセマンティックセグメンテーション | キャプチャ経路は未評価であり、公開されている重みはフォーマットに関係なく非商用です |
| L2CSの視線推定 | モデル自体がONNX、TorchScript、ExecuTorch、TensorRT、OpenVINOのみをサポートしており、これはモデル側の判断です |
| Depth Anything 3の深度推定 | このファミリーはすべてのフォーマットでエクスポートを拒否します |

RF-DETRには、アーティファクトを比較する前に読んでおく価値のある注意点が1つ
あります。そのパリティはONNXではなくCore AIエクスポーター自身が準備するグラフに
対して記録されており、640のキャンバスではRF-DETRのONNXアーティファクトはその準備
済みグラフと一致しません。Core AIの再ベイクはeagerモデルが行うアンチエイリアス付き
リサイズを保持しますが、ONNXの経路ではアンチエイリアスが無効になります。したがって
ONNXは、ネイティブでないキャンバスではそのファミリーの有効な参照になりません。

Appleの以前のフォーマットについては[Core ML](/docs/export/coreml)を参照してくだ
さい。ファミリーとタスクの完全な一覧は[エクスポートマトリクス](/docs/reference/export-matrix)を
参照してください。1つの組み合わせについては：

<code-tabs name="support" />
