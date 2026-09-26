---
title: ライセンス
seo_title: LibreYOLOのライセンス：コードと重み
description: LibreYOLO独自のコードはMITです。組み込まれたアップストリームコードと公開済みチェックポイントには独自のライセンスがあり、その一部は非商用です。
lead: >-
  LibreYOLOには個別にライセンスされる3種類のものがあります。独自のコード、モデルファミリーに組み込まれたアップストリームコード、学習済みチェックポイントです。多くの場合、それらのライセンスは同一ではありません。
keywords:
  - libreyolo ライセンス
  - mit コンピュータビジョン ライブラリ
  - 非商用 モデル 重み
  - model checkpoint license
  - apache-2.0 物体検出
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## LibreYOLO独自のコード

ライブラリはMITです。Python API、CLI、trainer、validator、exporter、データセットloader、`weights/` 以下の変換スクリプトが対象です。商用またはクローズドソース製品で使用でき、再配布するコピーに著作権表示とライセンステキストを含めれば、それ以上の義務はありません。

許諾の対象はコードまでです。[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE)ファイルには明確に記載されています。

> それらのライセンスはさまざまで、すべてが寛容なライセンスではありません。一部の公開済み重みは非商用または別の制限があり、このMIT Licenseはそれらには適用されません。モデルを選ぶことは、そのライセンスを選ぶことでもあります。

## ファミリーごとのアップストリームコード

ほとんどのファミリーは公開済み研究の移植であり、一部はアップストリームのソースを直接組み込みます。組み込まれたファイルは元の著作権表示と元のライセンスを維持します。MITがそれを上書きすることはなく、LibreYOLOが他者の著作物を再ライセンスすることもありません。もっともよく登場するのはApache-2.0とBSD-3-Clauseです。

Apache-2.0はDETRの系譜とTransformer関連の多くに適用されます。Meta AI（FAIR）のDETR、SenseTimeのDeformable DETR、BaiduのLW-DETR、Leilei WangらによるOV-DEIM、LibreYOLOがHugging Face Transformersから移植するSegFormer実装、PaddlePaddle AuthorsのPP-OCRv5、ETH ZurichのComputer Vision LabによるSwinIR、ByteDance SeedのDepth Anything 3です。Ross Wightmanとtimm contributorsによるtimmから派生した分類器にも適用されます。ResNet、DeiT、EfficientNetV2、MobileNetV4、Swinなどは、timmのImageNetテンソルを変更せずに読み込めるよう、そのモジュール名を反映しています。

BSD-3-Clauseはtorchvisionから派生したすべてのものに適用されます。Faster R-CNN、Mask R-CNN、FCOS、RetinaNet、SSD300、AlexNet、VGG、FCN、DeepLabv3です。

MITは、MegviiのNAFNet、Xingyi ZhouのCenterNet、作者のKin-Yiu WongとHao-Tang TsuiがMultimediaTechLabで再公開したYOLOv7など、より小さなグループに適用されます。YOLOv1からYOLOv4のファミリーは、Joseph Redmon、およびYOLOv4ではAlexey BochkovskiyによるDarknetプロジェクトのアーキテクチャを再現します。Darknetはpublic domainなので、それらには義務がありません。

同梱される1つのsubtreeには、オープンソースライセンスではないものがあります。DEIMv2ファミリーには、Meta PlatformsのDINOv3 License Agreementという独自の非OSIライセンスが適用されるDINOv3バックボーンコードが含まれます。そのコードを再配布する場合はagreementのコピーを同梱する必要があり、ITARの対象となる活動、軍事・戦争目的、原子力産業、諜報、兵器開発での利用が禁止されています。これらの条件はそのsubtreeだけに適用されます。

リポジトリ内の2つのファイルに全体像が記載されています。[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE)には、同梱されるすべてのサードパーティーsubtreeと、そのパス、ライセンスファイル、アップストリームソースが列挙されています。[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)には、LibreYOLOの派生元であるアップストリームプロジェクトが列挙され、各ライセンステキストが全文で転載されています。

## チェックポイントごとの重み

学習済み重みファイルはパッケージ内に同梱されません。公開済みチェックポイントは[LibreYOLO organization](https://huggingface.co/LibreYOLO)以下のHugging Faceにあり、各リポジトリには重みの由来となったプロジェクトを反映する独自の `LICENSE` と帰属表示があります。

条件の正式な情報源はそのリポジトリです。このページ、モデルページ、ソースツリー内の概要ではありません。ファイルの命名方法とダウンロード元については[チェックポイントと重み](/docs/weights)を参照してください。

ライセンスはファミリー間で異なり、1つのファミリー内のファイル間でも異なります。後者の例を2つ示します。

- YOLO9のCOCOチェックポイントはMITです。VisDrone2019-DETで学習された `LibreYOLO9P2s-visdrone.pt` は、非商用のCC BY-NC-SA 3.0です。
- RF-DETRの検出チェックポイントはApache-2.0です。回転ボックスのチェックポイントはCC BY 4.0です。CC BY 4.0で公開されたRoboflow Universeデータセットでファインチューニングされ、そのデータセットの帰属要件が重みに引き継がれるためです。

ファミリー間ではさらに広い範囲に及び、複数の公開済みチェックポイントは商用製品で利用できません。

- SegFormerは2つの層の違いがもっとも明確な例です。実装はHugging Face TransformersのコードをApache-2.0で移植したものです。公開済みADE20KチェックポイントはNVIDIAのリリースから変換され、NVIDIA Source Code Licenseが適用されます。このライセンスは再配布を許可しますが、利用を非商用の研究または評価に制限し、その制限を派生著作物へ引き継ぎます。これらのチェックポイントにはLibreYOLOの寛容な条件は適用されません。
- OV-DEIMチェックポイントはCC BY-NC 4.0であり、アップストリームの作者によって確認されています。推論のたびにAppleのMobileCLIP-B(LT)テキストタワーも読み込まれ、そのライセンスは利用を研究に制限します。これはチェックポイント独自の条件より厳しい制限です。
- SenseNova-VisionのコードはApache-2.0で、重みはCC BY-NC 4.0です。loaderは自動ダウンロードの前に毎回、非商用である旨を表示します。

LibreYOLOがチェックポイントをまったくホストしないファミリーもあり、そのページの「重み」行に明記されています。SAM 3はHugging Face上でMeta独自のSAM Licenseによって制限され、Metaから直接ダウンロードされます。MiDaSのrelease assetは再ホストされず、公式URLから取得されてhash検証されます。Dome-DETRはアップストリームへリンクされます。model cardのメタデータにライセンスが記載されていない一方、本文ではApache-2.0を主張すると同時に利用を学術研究へ制限しており、それらが一致しないためです。TEEDとDexiNedのアーキテクチャはMITですが、作者の公開済みチェックポイントは条件が非商用であるBIPEDで学習されているため、LibreYOLOは同梱も自動ダウンロードもしません。

複数のtorchvisionチェックポイントには独自のライセンスファイルがありません。LibreYOLOは、リリース元プロジェクトが使うライセンスに基づいてmirroringし、各model cardにその根拠がチェックポイントごとの明示的許諾ではなく推定であると記載し、学習済みモデルの条件が学習データに由来する可能性があるというtorchvision独自の警告を繰り返します。

## 1つのモデルの条件を確認する

モデルページのヘッダーには **ライセンス** 行があり、`Code X, weights Y` の形式でページのライセンスセクションへリンクします。そのセクションには、元の著作物と作者、アップストリームライセンス、アップストリームソース、LibreYOLOコードのライセンス、重み、条件で許可される内容の解釈が記載されています。同じページのチェックポイント表には、公開済みファイルごとに1行の **重みのライセンス** 列があり、条件が混在するファミリーではファイルごとに表示されます。

これらはすべて、ライブラリのチェック対象と同じデータからrenderされます。そのため、このページでは表として繰り返しません。手入力のライセンスマトリックスは1リリース内でも誤りやすく、この情報の誤りには高い代償が伴います。

ソースツリー内で対応するものは、同梱コード用の `NOTICE`、アップストリームプロジェクトとそのライセンステキスト用の `THIRD_PARTY_NOTICES.txt`、公開済みチェックポイントをファミリーごとにまとめた [`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)です。

次に、ダウンロードしようとしている正確なファイルのHugging Faceリポジトリを確認してください。これが正式な情報源であり、ドキュメントページが変更されなくても変わる可能性があります。

## 商用利用

問題になるのはコードでないことがほとんどです。MIT、Apache-2.0、BSD-3-Clauseはいずれも商用およびクローズドソースでの利用を許可します。再配布するコピーにライセンステキストと帰属表示を含めることを求め、Apache-2.0は特許ライセンスも許諾します。いずれも独自のアプリケーションコードには条件を課しません。

製品が行き詰まるのはチェックポイントです。周囲のコードがどれほど寛容でも、非商用チェックポイントは非商用のままです。また、`weights/LICENSE_NOTICE.txt` に直接記載されているとおり、ファイルを変換しても適用される条件は変わりません。制限付きチェックポイントから構築されたONNXまたはTensorRT成果物は、その制限を引き継ぎます。

NVIDIA Source Code Licenseのように、ライセンスの制限が派生著作物に及ぶ場合は、ファインチューニングしても制限から外れません。利用権を持つデータを使い、同じアーキテクチャをスクラッチ学習すれば外れます。コードは寛容なライセンスなので、自身で学習したモデルは自身のものであり、学習済みチェックポイントの条件は入りません。SegFormerページでは、その重みについてこの点を明記しています。出荷予定のファミリーのページにある「解釈」行を読んでください。

ライセンスの問題は出荷時ではなく、モデル選択時に判断してください。また、1つの寛容なチェックポイントの隣に制限付きのものがあるファミリーもあるため、実際にダウンロードしたファイルの条件を読んでください。

## 法的助言ではありません

このページは関係するライセンスを説明するものです。説明であって法的助言ではなく、いかなる保証も生じさせません。商用上重要な判断である場合は、自身でライセンスを読み、専門家へ相談してください。

