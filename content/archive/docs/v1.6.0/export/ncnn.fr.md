---
title: ncnn
seo_title: "Exporter vers ncnn depuis LibreYOLO"
description: "Exportez un modèle LibreYOLO vers ncnn via PNNX : la paire param et bin, le canevas d'export fixe, la réécriture du Focus de YOLOX, et les familles qui convertissent."
lead: "ncnn est la bibliothèque d'inférence CPU de Tencent pour les cibles mobiles. LibreYOLO convertit via PNNX, en écrivant un graphe model.ncnn.param à côté d'un fichier de poids model.ncnn.bin et d'un metadata.yaml qui porte la famille, la tâche et les noms de classe."
keywords:
  - exporter yolo ncnn
  - pnnx
  - model.ncnn.param
  - inférence cpu mobile
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="ncnn")'
    mono: true
  - label: Écrit
    value: "Un répertoire avec model.ncnn.param, model.ncnn.bin et metadata.yaml"
  - label: Extra
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Se recharge avec
    value: 'LibreYOLO("weights/LibreYOLO9t_ncnn")'
    mono: true
  - label: Formes
    value: "Fixes. Les métadonnées enregistrent dynamic=False quel que soit le flag."
  - label: Précision
    value: "FP32 uniquement. half=True et int8=True sont refusés."
verification: "Lu depuis libreyolo/export/ncnn.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/ncnn.py et pyproject.toml sur la branche dev."
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # pnnx convertit, ncnn exécute le résultat.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit le répertoire weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Arguments
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int, ou (hauteur, largeur)
            batch=1,
            simplify=True,    # ne vaut que pour le chemin ONNX de repli
            opset=None,       # auto, ne vaut que pour le chemin ONNX de repli
            output_path=None, # None écrit weights/<stem>_ncnn
        )

        # half=True et int8=True sont refusés pendant la validation.
  run:
    - label: Via LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ncnn brut
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnn prend une seule image CHW, pas un batch.
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Le prétraitement et le postprocessing sont à votre charge ici.
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## Installation

<code-tabs name="install" />

L'extra installe les deux moitiés de la chaîne d'outils : `pnnx` effectue la
conversion et `ncnn` exécute le résultat. Aucun des deux ne passe par ONNX sur le
chemin principal.

## Export

<code-tabs name="export" />

L'artefact est un répertoire. `weights/LibreYOLO9t_ncnn` contient
`model.ncnn.param`, `model.ncnn.bin` et `metadata.yaml` ; les trois forment un
seul artefact et se déplacent ensemble.

La conversion tente d'abord PNNX directement depuis PyTorch. En cas d'échec, elle
exporte un graphe ONNX statique vers un répertoire temporaire et lui applique
l'outil en ligne de commande `pnnx`, et l'export ne lève que si les deux chemins
échouent, en rapportant les deux erreurs. `opset` et `simplify` n'affectent donc
que le repli.

YOLOX demande une réécriture pour convertir tout court. Sa couche Focus utilise
du slicing avec pas, que PNNX ne sait pas abaisser, donc l'export la remplace par
`pixel_unshuffle` et permute les canaux d'entrée de la convolution suivante pour
compenser l'ordre des canaux différent. La sortie est numériquement identique, et
les poids d'origine sont restaurés après l'export.

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` reconnaît tout répertoire contenant `model.ncnn.param` et
`model.ncnn.bin`, lit `metadata.yaml`, et renvoie le même objet `Results` que le
checkpoint.

Le second snippet est le chemin du runtime brut, et deux détails y diffèrent de
tous les autres formats de cette section. ncnn travaille sur une seule image CHW
plutôt que sur un batch, il n'y a donc pas d'axe de batch en tête. Les noms de
blob viennent du fichier `.param` ; PNNX écrit `in0` et `out0` par convention, et
le backend analyse le fichier au lieu de les supposer. Le prétraitement, le
décodage, le NMS et le rééchelonnement des coordonnées sont à votre charge sur ce
chemin.

## Contraintes

Du FP32 sur un canevas fixe. `half=True` et `int8=True` sont tous deux refusés
pendant la validation, et les métadonnées exportées enregistrent `dynamic=False`
quoi qu'ait dit le flag, pour qu'aucun backend ne suppose un axe que le graphe
n'a pas.

Toutes les familles de type DETR sont refusées au preflight : `detr`,
`deformable_detr`, `dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rfdetr` et `ec`. Le message est le même pour toutes :
le modèle demande des opérations de décodeur ou d'échantillonnage indisponibles
dans ncnn, et il oriente vers ONNX, OpenVINO, TorchScript ou TensorRT à la place.

Ce qui convertit est large du côté convolutionnel : YOLO9 et YOLO9-E2E, YOLOX,
PicoDet, YOLO-NAS détection et pose, les détecteurs plus anciens YOLO1, YOLO3,
YOLO4 et YOLO7, les quatre familles de classification CNN, la segmentation
sémantique PIDNet, la détection de points FOMO à un 96 par 96 fixe, ZipDepth,
NAFNet et Real-ESRGAN.

Les entrées bloquées nomment l'échec concret. Les graphes transformer laissent en
général derrière eux des nœuds `pnnx.Expression` non supportés, ce qui produit un
réseau sans blob d'entrée exécutable, et c'est ce qui arrête DINOv2, CLIP,
SigLIP2 et SegFormer. BiRefNet demande la convolution déformable de torchvision,
que PNNX ne sait pas abaisser. Le graphe converti de YOLO2 termine le runtime
ncnn sous Windows avec une division entière par zéro native pendant l'extraction
de la sortie.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une seule combinaison :

<code-tabs name="support" />
