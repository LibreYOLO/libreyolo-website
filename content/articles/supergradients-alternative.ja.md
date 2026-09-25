---
title: "SuperGradientsの代替：リアルタイムコンピュータビジョン向け"
description: "NVIDIAが2024年にDeciを買収して以降、SuperGradientsのリリースは止まっています。LibreYOLOは同じYOLO-NASの重みを実行できる、メンテナンス中の代替ライブラリです。予測、学習、エクスポートを1つのMITライセンスAPIで行えます。"
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "SuperGradientsは現在もメンテナンスされていますか？"
    a: "いいえ。最後のリリースは2024年4月の3.7.1で、NVIDIAによるDeci買収の直前でした。それ以降リリースはなく、issueにも回答がなく、ドキュメントサイトもオフラインになりました。正式にアーカイブされたわけではありませんが、誰もいない状態です。"
  - q: "SuperGradientsの最適な代替手段は何ですか？"
    a: "YOLO-NASを実行するなら、LibreYOLOは同じDeciの重みを、メンテナンス中のMITライセンスAPIで読み込み、学習、検証、エクスポートにも対応します。他のアーキテクチャでSuperGradientsの学習レシピに依存している場合は、依存関係を固定すれば古いリポジトリも引き続き動作します。"
  - q: "YOLO-NASを商用利用できますか？"
    a: "どのライブラリで読み込んでも、Deciの事前学習済み重みは非商用ライセンスです。アーキテクチャ自体は寛容なライセンスなので、自分のデータでYOLO-NASをゼロから学習すれば、Deciのチェックポイントを一度も使っていないモデルを作れます。"
  - q: "LibreYOLOはSuperGradientsの機能をすべて置き換えますか？"
    a: "すべてではありません。LibreYOLOはYOLO-NASのCoreMLエクスポートに対応しておらず、このモデルファミリーのTensorRT経路は未検証です。また、SuperGradientsの量子化を考慮した学習レシピに直接相当する機能もありません。これらが必要な場合は、古いリポジトリを残してください。"
---

SuperGradientsはDeciのオープンソース学習ライブラリであり、これまでに公開されたリアルタイム検出器の中でも特に高精度なYOLO-NASの開発元でした。2024年5月にNVIDIAがDeciを買収し、SuperGradientsは動きを止めました。最後のリリース3.7.1は2024年4月8日でした。supergradients.comのドキュメントはオフラインになりました。約120件のissueが未解決のまま残り、そのうち[「このプロジェクトは終了したのですか？」](https://github.com/Deci-AI/super-gradients/issues/2062)というタイトルのissueにもメンテナーから回答がありません。それ自体が答えと言えます。

リポジトリはアーカイブされていないため、一見するとまだ活動中に見えます。しかし、インストールしようとすると放置された状態をすぐに実感します。`super-gradients`は`torchmetrics==0.8`を固定しており、現在のPyTorch環境と競合します。さらにhydra、omegaconf、boto3、tensorboardも依存関係に加わります。月日が経つほど、これらのバージョン固定はほかの環境との競合を強めていきます。修正は期待できません。

だからといって、YOLO-NASのモデル自体が劣っているわけではありません。Largeバリアントはリアルタイム速度で、今もCOCOの52.2 mAPを記録します。モデルに問題はありません。開発基盤が崩れてしまったのです。

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS accuracy vs parameters - visionanalysis.org">
</iframe>

## LibreYOLOで同じ重みを実行する

LibreYOLOはDeciのCDNからYOLO-NASのチェックポイントを直接読み込み、ほかのすべてのモデルファミリーと同じAPIで扱えます。hydraの設定を書く必要も、予測ラッパーをほどく必要もありません。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

S、M、Lの検出バリアントはすべて動作し、姿勢推定にも対応しています。`LibreYOLONASs-pose.pt`に置き換えると、COCOキーポイントが返されます。すべてのモデルファミリーが同じ`Results`オブジェクトを返すため、YOLO-NASとRF-DETRまたはD-FINEを自分のデータで比較するときも、別のコードベースを用意せず、1行を変更するだけです。

## 推論だけでなく学習とエクスポートにも対応

SuperGradientsはまず学習ライブラリだったため、本当の代替手段には学習機能が必要です。LibreYOLOでは、ほかのモデルと同じ呼び出し方でYOLO-NASをデータセットに合わせてファインチューニングできます。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

ランダム初期化したモデルから学習することもできます。これはライセンス上重要です（詳しくは後述します）。どのデータ形式のデータセットでも、検証によってmAP指標を取得できます。エクスポート形式はONNX、TorchScript、OpenVINO、NCNN、TFLiteに対応しています。YOLO-NASについては、元のライブラリが提供していたものより幅広いエクスポート形式です。詳しくは[YOLO-NASのドキュメントページ](https://www.libreyolo.com/docs/models/yolo-nas)を、簡潔な説明は[YOLO-NASは現在もメンテナンスされています](/articles/yolo-nas-with-libreyolo)をご覧ください。

## 古いリポジトリを使い続けるのが適切なケース

率直に説明します。SuperGradientsをインストールしたままにしておくべきケースは3つあります。

**CoreML。** LibreYOLOはYOLO-NASのCoreMLエクスポートに対応していません。Appleデバイス向けに出荷していて`.mlpackage`が必要な場合、SuperGradientsには今も動作する方法があります。

**TensorRT。** SuperGradientsは、バッチサイズの癖も含めて、YOLO-NASのTensorRTフローを文書化し、テストしていました。このモデルファミリーのLibreYOLOのTensorRT対応は未検証です。

**量子化を考慮した学習レシピ。** YOLO-NASはINT8向けに設計されており、SuperGradientsにはその性能を引き出すQATレシピが用意されていました。LibreYOLOはINT8とFP16の量子化でエクスポートできますが、これらの学習レシピに相当する機能はありません。

2024年頃の環境を固定すれば、今もリポジトリは動作します。ただし、メンテナンスされていない基盤の上に新しいものを構築するのは避けてください。

## 重みに関する注意

事前学習済みYOLO-NASの重みはDeciが提供しており、非商用ライセンスで公開されています。そのライセンスは、どのライブラリで読み込んでも重みに適用されます。LibreYOLOは重みをホストもミラーもしていません。ダウンロード元はDeciのパブリックCDNで、ダウンロードの前にDeciの利用条件が表示されます。研究や非商用の用途であれば、どちらの方法でも問題ありません。

商用利用できるYOLO-NASが必要な場合、明確な方法があります。アーキテクチャ自体は寛容なライセンスなので、自分のデータでゼロから学習すれば、Deciのチェックポイントに由来しないモデルを作れます。LibreYOLOは`LibreYOLONAS(None, size="s")`でこれに対応しています。

## 試してみる

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLOはMITライセンスで、Linux、Mac、Windowsで動作します。GPU、Apple Silicon、通常のCPUのいずれでもコードを変更せずに使えます。1つのAPIでYOLO-NAS、RF-DETR、D-FINE、DEIM、YOLOX、RTMDetなどに対応し、検出、セグメンテーション、姿勢推定、分類、深度推定、トラッキングを扱えます。

GitHubでスターを付けてください：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | ドキュメント：[libreyolo.com/docs](https://www.libreyolo.com/docs)
