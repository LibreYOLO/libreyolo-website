---
title: 身体メッシュ
seo_title: LibreYOLOでの身体メッシュ復元
description: LibreYOLOで人物ごとにパラメトリック3D身体メッシュを復元します。人物ボックスまたは検出器から推論し、頂点、関節、カメラ移動を読み取ります。
lead: >-
  身体メッシュ復元は、1枚の画像と人物ボックス集合から、人物ごとにパラメトリック3D身体を生成します。形状・姿勢パラメータ、姿勢適用済み頂点、3D関節、レンズ前に配置するカメラ移動が含まれます。
keywords:
  - 人体メッシュ 復元 python
  - body mesh
  - 3d body pose
  - SAM 3D Body
  - MHR
  - パラメトリック 人体モデル
  - libreyolo mesh
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # このファミリーはLibreYOLO()ファクトリーに登録されないため直接構築
        # model_path=Noneでアクセス制限されたHugging Faceダウンロードを開始
        # 文字列は既存のローカルチェックポイントとして扱われ取得されない
        # 推論にはCUDAが必要
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.body_model)      # これらのテンソルが使うパラメータ化
        print(meshes.vertices.shape)  # (N, V, 3)でカメラ座標系のメートル単位
        print(meshes.joints3d.shape)  # (N, J, 3)
        print(meshes.joints2d.shape)  # (N, J, 2)で元画像のピクセル単位
    - label: 人物検出器を使う
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # person_detectorは構築済みLibreYOLO検出器か通常の呼び出し可能
        # オブジェクトかPersonDetectorを受け付ける 名前の短縮指定はない
        detector = LibreYOLO("LibreYOLO9s.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## 定義

身体メッシュ復元は画像ごとに `Meshes` ペイロードを返し、`result.boxes` と行単位で対応します。行 `i` はボックス `i` の人物を表し、姿勢タスクでキーポイントに使用する契約と同じです。

すべては元画像のカメラ座標系で表されます。`transl` はメートル単位で、+zはカメラから遠ざかる方向です。`vertices` と `joints3d` はメートル単位で、すでに `transl` を含むため、さらに合成する必要はありません。`joints2d` はネットワークが見たクロップではなく、元画像キャンバス上のピクセル単位です。すべての人物が同じトポロジーを共有するため、`faces` は行ごとではなく画像全体について1回だけメッシュトポロジーを保持します。このバージョンにワールド座標系や重力座標系はなく、それらを暗黙に代用するフィールドもありません。

パラメータの配置は身体モデルごとに異なるため、形状について固定されたものはありません。`body_model` がパラメータ化の名前を示し、個数はテンソルから読み取られます。Momentum Human Rigの `"mhr"` では、回転はaxis-angleではなくラジアン単位のオイラー角で、`body_pose` は関節ごとの3値ではなく、関節ごとのパラメータを平坦化したベクトルです。`betas` は識別用blendshape係数です。骨格のスケール、手の姿勢、顔の表情は `extras` にあります。

正規のタスクキーは `mesh` です。`body-mesh`、`hmr`、`human-mesh-recovery` はこのキーへ正規化されます。

## モデル

[SAM 3D Body](/docs/models/sam-3d-body)はこのタスクを提供する唯一のファミリーで、移植版ではなくラッパーです。Metaの `sam-3d-body` パッケージはSAM Licenseの下で公開され、LibreYOLO独自のコードはそこから派生できないため、何も組み込みません。2つのバックボーンが同じMHR身体モデルを共有します。`d3` はDINOv3 ViT-H/16+エンコーダー、`h` は初代ViT-Hです。

最初の推論前に3つの要件があり、いずれも省略できません。

アップストリームパッケージはLibreYOLOではなく、自分でインストールします。

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

`sam_3d_body_path=` または `SAM_3D_BODY_PATH` 環境変数でクローンの場所をライブラリへ指定します。このファミリーを構築しない利用者にはインポートが発生しません。

チェックポイントのミラーはアクセス制限されています。Hugging Faceのモデルページでライセンスに同意し、`hf auth login` で認証してください。そうしないと最初のダウンロードが失敗します。MHR身体モデル自体は別のApache-2.0リリースで、独自の公開場所から取得され、ローカルにキャッシュされます。

推論にはCUDAデバイスが必要です。アップストリーム推定器は確認せずバッチをGPUへ移動するため、フォールバックできるCPU経路はなく、`device="cpu"` はエラーになります。

## 推論

<code-tabs name="predict" />

人物をモデルへ渡す方法は2つあります。`person_boxes` はすでに所有するボックスを1枚の画像に限って渡します。固定されたボックス集合は動画フレームをまたいで人物を追跡できないため、動画ソースとともに渡すと、最初のフレームのボックスを暗黙に再利用せずエラーになります。`person_detector` は構築済みのLibreYOLO検出器、呼び出し可能オブジェクト、または `PersonDetector` を受け付け、動画ではこの経路を使用します。`focal_length` は既知のカメラ内部パラメータを指定します。未指定の場合、モデル独自の推定値を使用し、`meshes.focal_length` で報告されます。

このファミリーは `LibreYOLO()` ファクトリーや `libreyolo predict` CLIコマンドに接続されていません。唯一のエントリポイントは `LibreSAM3DBody` です。ソース、ストリーミング、結果の処理については、[推論](/docs/predict)を参照してください。

## 学習

このタスクのファミリーはLibreYOLO内で学習できません。`LibreSAM3DBody.train()` はエラーになります。アップストリームプロジェクトで学習し、得られたチェックポイントをここで読み込んでください。

## 検証

メッシュ用のバリデーターはなく、`val()` はエラーになります。一般的なベンチマークは研究ライセンス専用なので、同梱されず、代わりに取得することもできません。

指標自体は `libreyolo.validation.mesh_metrics` として利用でき、すでに所有するデータセットに対して評価できます。予測とターゲットの関節、およびオプションで予測とターゲットの頂点を受け取り、バリデーターとまったく同じ形式のキーを持つ辞書を返します。

`metrics/mpjpe` はルート関節の位置合わせ後の平均関節位置誤差です。そのため、人物が場面内のどこに立っているかを無視して姿勢を評価します。`metrics/pa_mpjpe` は完全なProcrustes位置合わせ、すなわち回転、一様なスケール、移動の後で同じ量を計測します。全体の向きと身体サイズの誤差を除き、関節で表される姿勢だけを残します。`metrics/pve` は頂点の重心で位置合わせした後のメッシュ表面全体の平均頂点誤差です。関節指標とは異なり身体形状に敏感で、両方の頂点配列を指定した場合だけ表示されます。3つとも低いほど良い指標です。入力はメートル単位と想定され、`scale_to_mm` は結果を論文で報告されるミリメートルへ変換します。

## エクスポート

メッシュのエクスポートは実装されていません。MHRパラメータの配置をPyTorch外でどう保持するかを含め、LibreYOLOはまだこのタスクのエクスポート済みグラフのメタデータ契約を定義していません。そのため、解釈できない出力を持つグラフを生成せず、`export()` はエラーになります。
