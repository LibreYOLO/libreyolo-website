---
title: "RT-DETRをUltralyticsなしで使う方法：v1・v2・v4、Apache-2.0対応"
description: "Apache-2.0の重みをMITライブラリで使い、RT-DETR、RT-DETRv2、RT-DETRv4を推論・ファインチューニングできます。predict、train、val、exportを1つのAPIで実行します。"
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "RT-DETRは商用利用できますか？"
    a: "寛容なライセンスの実装を使えばできます。RT-DETRとRT-DETRv2の元のコード（lyuwenyu/RT-DETR）およびRT-DETRv4（RT-DETRs/RT-DETRv4）はApache-2.0で、LibreYOLOはApache-2.0の重みをMITライブラリで使い、3つのバージョンすべてを実行できます。ultralyticsパッケージ内のRT-DETR実装はAGPL-3.0で、クローズドソースで使う場合は有償のEnterprise Licenseが代替手段です。これは一般的な情報であり、法的助言ではありません。"
  - q: "RT-DETRv4のライセンスは何ですか？"
    a: "Apache-2.0です。github.com/RT-DETRs/RT-DETRv4のLICENSEファイルはApache-2.0です。RT-DETRv4は学習時のみDINOv3の教師モデルを使います。公開された生徒モデルの重みにDINOv3のパラメーターは含まれないため、MetaのDINOv3ライセンスはその重みに及びません。"
  - q: "UltralyticsなしでRT-DETRをファインチューニングできますか？"
    a: "できます。LibreYOLOでは公開済みのCOCO重みを使い、model.train()またはコマンドラインのlibreyolo trainでRT-DETR、RT-DETRv2、RT-DETRv4の検出チェックポイントをファインチューニングできます。RT-DETRv2の回転ボックスモデルは推論専用です。"
  - q: "LibreYOLOにはどのRT-DETRの重みがありますか？"
    a: "RT-DETRはr18、r34、r50、r50m、r101、l、x、RT-DETRv2はr18、r34、r50、r50m、r101、RT-DETRv4はs、m、l、xで、すべて640 pxのCOCO検出モデルです。RT-DETRv2には、DOTA v1.0で1024 pxにて学習したn、s、m、l、xの回転ボックスモデルもあります。すべてのファイルがApache-2.0です。"
---

RT-DETRはBaiduが開発したリアルタイム物体検出トランスフォーマーです。密なグリッドではなく固定数のクエリをデコードするため、調整が必要なNMS処理がありません。検索すると、最初に見つかる実装の1つはPythonパッケージ`ultralytics`に含まれるものです。ここで、モデル自体にはなかったライセンスの疑問が生じます。

## RT-DETRのライセンスはリポジトリによって異なります

ライセンスが適用されるのはモデルではなく、リポジトリです。RT-DETRには複数のリポジトリがあり、ライセンスはそれぞれ異なります。

| リポジトリ | 対象 | ライセンス |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETRとRT-DETRv2、PyTorchとPaddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | `ultralytics`パッケージ内のRT-DETR | AGPL-3.0またはEnterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR、RT-DETRv2、RT-DETRv4 | MITライセンスのコード、Apache-2.0の重み |

Ultralyticsの実装は優れています。[RT-DETRのページ](https://docs.ultralytics.com/models/rtdetr/)には、学習、検証、推論、エクスポートに対応する学習済みの`rtdetr-l.pt`と`rtdetr-x.pt`が掲載されています。AGPL-3.0ライセンスのパッケージに含まれており、プロジェクト全体をオープンソースにしたくないチーム向けに、UltralyticsはEnterprise Licenseを販売しています。どちらも適さない場合は、アーキテクチャの作者が公開しているApache-2.0版を利用できます。[YOLOのライセンス解説](/articles/yolo-licenses-explained)ではYOLOファミリー全体の同じパターンを説明し、[商用ライセンスガイド](/articles/yolo-commercial-license)ではクローズドソース製品にとってAGPL-3.0が何を意味するかを説明しています。

## LibreYOLOでRT-DETRを実行する

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

コマンドラインでも同じ呼び出し方ができます。

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf`と`max_det`は、クエリとクラスに対するtop-kデコードの結果をフィルタリングします。NMSがないため、`iou`は受け付けますが使用されません。返される`Results`オブジェクトはすべてのLibreYOLOモデルで共通なので、検出器は1行の変更で切り替えられます。

## RT-DETR、RT-DETRv2、RT-DETRv4を1つのローダーで使う

バージョンはファイル名に含まれ、`LibreYOLO()`はチェックポイントに応じて処理を振り分けるため、3つのバージョンを同じ方法で読み込めます。

* **RT-DETR：** `LibreRTDETR`はr18、r34、r50、r50m、r101（ResNetバックボーン）、l、x（HGNetv2）に対応します。
* **RT-DETRv2：** `LibreRTDETRv2`はr18、r34、r50、r50m、r101に対応します。バージョン1のアーキテクチャを維持し、変形可能アテンションのサンプリング方法を変更しています。
* **RT-DETRv4：** `LibreRTDETRv4`はs、m、l、xに対応します。D-FINEのアーキテクチャを再利用し、DINOv3の教師モデルからHGNetv2の生徒モデルへ知識蒸留した重みを使います。

すべての検出用重みはCOCOで学習され、640 pxで動作します。参考値として、上流のREADMEではCOCOでRT-DETR-R18が46.5 AP、RT-DETR-Lが53.0 AP、RT-DETRv4-Sが49.8 AP、RT-DETRv4-Xが最大57.0 APと報告されています。[RT-DETRのドキュメントページ](/docs/models/rt-detr)には、すべてのチェックポイントについてLibreYOLO独自のベンチマーク表があります。

RT-DETRv2は回転ボックスにも対応します。`LibreRTDETRv2n-obb.pt`から`LibreRTDETRv2x-obb.pt`までのモデルは、公式のDOTA v1.0チェックポイントをLibreYOLO形式に変換したもので、1024 pxで15種類の航空画像クラスを扱います。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

[回転物体検出](/docs/tasks/oriented-detection)では、ラベル形式と評価指標を説明しています。

## 独自データでRT-DETRをファインチューニングする

学習は公開済みチェックポイントから始めます。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

実際に学習する場合は、`data`に独自データセットのYAMLファイルを指定します。各バージョンにはそれぞれの学習率のデフォルト値があり、バージョン1と2では元のレシピに従い、バックボーンの学習率を`lr0`の20分の1に設定します。CLIでのマルチGPU実行には`device=0,1`を指定します。また、`lora`追加パッケージをインストールして`lora=True`にすると、[LoRA](/docs/train/lora)アダプターでファインチューニングできます。RT-DETRv2の回転ボックスモデルは推論専用です。データセット、拡張、ロガーについては[学習](/docs/train)を参照してください。

## 検証とエクスポート

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()`は通常の辞書を返します。エクスポート先にはONNX、TorchScript、ExecuTorch、TensorRT、OpenVINOがあります。[モデルページ](/docs/models/rt-detr)のエクスポート対応表では、タスクごとに検証済みの形式を確認できます。エクスポートした`.onnx`または`.engine`ファイルを`LibreYOLO()`で再読み込みすると、同じ`Results`が返されます。形式ごとのガイドは[エクスポート](/docs/export)を参照してください。

## 元のリポジトリを使うのが適切な場合

論文の結果を作者の設定そのままで再現したい場合、Paddle版が必要な場合、またはRT-DETRv4のDINOv3教師モデルによる蒸留を自分で実行したい場合は、上流のリポジトリを使ってください。LibreYOLOでファインチューニングできるのは公開済みのv4生徒モデルであり、教師モデルは実行しません。また、すでにプロジェクトがAGPL-3.0の対象である場合や、Ultralytics Enterprise Licenseの適用を受けている場合、ライセンス上の理由で移行する必要はありません。

## ライセンスについて

LibreYOLOのコードはMITです。すべてのRT-DETR重みファイルはApache-2.0で、各Hugging Face重みリポジトリには、重みを再配布するときに保持するようApache-2.0が求める上流のLICENSEとNOTICEが含まれています。独自データで学習した重みは利用者のものです。RT-DETRv4は学習時の教師モデルとしてのみDINOv3を使い、公開された生徒モデルにDINOv3のパラメーターは含まれません。これは一般的な情報であり、法的助言ではありません。公開前に最新のLICENSEファイルを確認してください。

## 試してみる

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLOはMITライセンスで、Linux、Mac、Windows上で動作します。GPU、Apple Silicon、通常のCPUのいずれでもコードを変更せずに使えます。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [RT-DETRのドキュメント](/docs/models/rt-detr)
