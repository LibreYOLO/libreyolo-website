---
title: ONNX
seo_title: Exporter vers ONNX depuis LibreYOLO
description: "Exportez un modèle LibreYOLO vers ONNX\_: l'opset que LibreYOLO choisit selon la famille, les axes dynamiques, le NMS embarqué, l'INT8 et la façon dont le graphe se recharge."
lead: >-
  ONNX est un format de graphe portable. LibreYOLO trace le modèle avec
  torch.onnx.export, simplifie éventuellement le graphe, et écrit la famille, la
  tâche, les noms de classes et la taille d'entrée dans les métadonnées du
  fichier lui-même, de sorte que n'importe quel backend LibreYOLO peut
  reconstruire le postprocessing.
keywords:
  - exporter yolo onnx
  - onnxruntime
  - torch.onnx.export
  - onnx opset
  - axes dynamiques onnx
  - nms embarqué onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="onnx")
    mono: true
  - label: Écrit
    value: 'Un seul fichier .onnx, métadonnées embarquées dans le graphe'
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Se recharge avec
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: Formes
    value: "Batch dynamique par défaut en Python\_; exceptions par tâche ci-dessous"
  - label: Précision
    value: 'FP32, FP16 (half=True), INT8 (int8=True, détection YOLO9)'
verification: >-
  Lu depuis libreyolo/export/onnx.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/onnx.py et
  libreyolo/cli/commands/export.py sur la branche dev.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Écrit weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Arguments
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int, ou (hauteur, largeur)
            batch=1,
            dynamic=True,     # défaut Python, la CLI vaut False
            simplify=True,    # passe onnxsim sur le graphe
            opset=None,       # None choisit 13, ou 17 pour les familles DETR
            half=False,       # poids et activations FP16
            int8=False,       # INT8 QDQ, détection YOLO9 uniquement
            data=None,        # data.yaml de calibration, INT8 uniquement
            device=None,      # device de trace, None = celui du modèle
            output_path=None, # None écrit weights/<stem>.onnx
        )
  nms:
    - label: Embarquer le NMS dans le graphe
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Détection YOLO9 uniquement, batch 1. dynamic est forcé à False.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 avec données de calibration
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # quelques centaines d'images représentatives
            fraction=1.0,
        )
  run:
    - label: Via LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: ONNX Runtime seul
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # Le prétraitement et le postprocessing sont à votre charge ici.

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # Le graphe porte la famille, la tâche, les classes et la taille.

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Vérifier une famille et une tâche avant d'exporter
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## Installation

<code-tabs name="install" />

L'extra installe `onnx`, `onnxsim` et `onnxruntime`. `onnx` seul suffit à écrire
le fichier ; `onnxsim` exécute la passe de simplification, et `onnxruntime`
exécute l'artefact et réalise la calibration INT8.

## Export

<code-tabs name="export" />

Sans `output_path`, le fichier atterrit dans `weights/` sous le nom de base du
checkpoint, avec `_fp16` ou `_int8` ajouté quand cette précision a été demandée.

`dynamic` vaut `True` par défaut en Python et `False` sur la CLI. Quand il est
actif, l'axe batch devient symbolique et quelques tâches vont plus loin : la
segmentation sémantique ouvre aussi la hauteur et la largeur du masque, la
restauration Real-ESRGAN ouvre les axes spatiaux, et les détecteurs à deux
étages gardent la hauteur et la largeur source dynamiques parce que leur
redimensionnement se fait à l'intérieur du graphe.

`opset` est choisi selon la famille quand il est omis. Les familles de type DETR
(`detr`, `deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`,
`lwdetr`, `rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`) plus `deit`, `midas` et
`moge2` reçoivent l'opset 17, celui où `aten::scaled_dot_product` s'abaisse.
Tout le reste reçoit 13. Le matting passe à 19 quoi qu'il arrive, parce que le
décodeur de BiRefNet a besoin de l'opérateur `DeformConv`, qu'ONNX définit à
partir de l'opset 19.

`simplify=True` exécute `onnxsim` et conserve le graphe d'origine si la passe
échoue, si bien qu'une erreur de simplification est un avertissement plutôt
qu'un échec d'export. Sur macOS arm64 avec `onnx` 1.22 ou plus récent et
`onnxsim` 0.6.5 ou plus ancien, la passe est entièrement sautée, parce que cette
combinaison peut faire planter le processus Python.

### NMS embarqué

<code-tabs name="nms" />

`nms=True` ne concerne que la détection YOLO9 et exige batch 1 ; le demander
avec `dynamic=True` produit un avertissement et désactive dynamic. Le graphe a
alors deux sorties : `output`, de forme `(batch, max_det, 6)`, et `raw`, le
tenseur détecteur non décodé qu'utilise le backend propre à LibreYOLO pour que
le postprocessing reste identique au chemin PyTorch.

### DeepStream

`deepstream=True` est une option propre à ONNX. Elle exporte le graphe dans la
disposition qu'attend le parser de NVIDIA DeepStream et écrit deux fichiers
annexes à côté, `config_infer_primary_<stem>.txt` et `<stem>_labels.txt`, pour
que l'artefact s'insère dans un pipeline sans configuration écrite à la main.

Elle est mutuellement exclusive avec `nms=True`, et demander les deux lève une
`ValueError` : DeepStream effectue la suppression dans son propre étage de
clustering. La passer à un format autre qu'ONNX lève également. Voir
[DeepStream](/docs/export/deepstream) pour la grille des familles et des tâches
prises en charge et pour la compilation du parser.

### INT8

<code-tabs name="int8" />

`int8=True` exécute la quantification statique d'ONNX Runtime et écrit un graphe
QDQ dont les entrées et les sorties sont en float32. Seuls les nœuds `Conv` et
`Gemm` sont quantifiés. Laisser le décodage de la tête de détection en float32
est délibéré : cette concaténation mélange des coordonnées de boîtes à l'échelle
du pixel avec des scores de classe dans la plage 0 à 1, et une unique échelle
d'activation par tenseur dominée par la magnitude des boîtes ramènerait tous les
scores à zéro.

Ce flag ne s'applique pour l'instant qu'à la détection YOLO9, et tout le reste
lève `NotImplementedError` au préflight. Omettre `data` retombe sur
`coco8.yaml` avec un avertissement ; huit images ne forment pas un jeu de
calibration représentatif. Un modèle déjà quantifié dans PyTorch suit un autre
chemin, décrit sur [Quantification](/docs/export/quantization).

## Exécuter l'artefact

<code-tabs name="run" />

`LibreYOLO()` aiguille sur le suffixe `.onnx` et renvoie le même objet `Results`
qu'un checkpoint `.pt`, parce que les noms de classes, la tâche, la taille
d'entrée et le schéma de pose ont été écrits dans les `metadata_props` du graphe
au moment de l'export. Avec `device="auto"`, la session prend
`CUDAExecutionProvider` quand ONNX Runtime le signale, et retombe sur le CPU
sinon.

Le second snippet s'adresse aux lecteurs qui n'ont pas LibreYOLO installé. Le
prétraitement, le décodage, le NMS et la remise à l'échelle des coordonnées sont
alors à votre charge ; le bloc de métadonnées reste là, prêt à être lu.

## Contraintes

Les noms des tenseurs de sortie sont fixés par tâche, et c'est à eux que doit
correspondre un consommateur sans métadonnées :

| Tâche | Noms de sortie |
|---|---|
| Détection, têtes à grille et à ancres | `output` |
| Détection, type DETR | `pred_logits`, `pred_boxes` |
| Détection, RF-DETR | `dets`, `labels` |
| Classification | `output` |
| Segmentation sémantique | `semantic_logits` |
| Profondeur | `depth` |
| Normales de surface | `normal` |
| Contours | `edges` |
| Restauration | `restored` |
| Matting | `matte` |
| Regard | `yaw_logits`, `pitch_logits` |

RF-DETR est aussi la seule famille dont le tenseur d'entrée s'appelle `input`
plutôt que `images`.

Plusieurs tâches portent un contrat de runtime à résolution fixe dans cette
version. La profondeur, les normales de surface et les contours refusent
`batch != 1` et forcent `dynamic=False`. Le matting force le carré natif de
1024, parce que les tables de positions relatives du Swin de BiRefNet sont liées
à leur résolution. La restauration force un canevas fixe pour toutes les
familles sauf Real-ESRGAN, dont le générateur est entièrement convolutif.

Un `imgsz` rectangulaire fonctionne pour les familles YOLO9, HRNet, NAFNet et
Real-ESRGAN. Les familles au contrat carré fixe (`clip`, `deformable_detr`,
`detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `moge2`,
`rtdetr`, `rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`) le refusent
purement et simplement.

Deux combinaisons sont refusées avant le tracing : la segmentation YOLO9, parce
que YOLO9 ne fait que de la détection dans LibreYOLO, et la segmentation
RTMDet-Ins, dont le décodage de masques à noyaux dynamiques n'a pas de contrat
de runtime exporté.

Pour la grille complète des familles et des tâches, voir
[la matrice d'export](/docs/reference/export-matrix). Pour une seule
combinaison, interrogez directement la bibliothèque :

<code-tabs name="support" />
