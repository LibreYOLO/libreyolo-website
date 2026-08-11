---
title: CenterNet
families:
  - centernet
seo_title: "CenterNet\_: la détection d'objets dans LibreYOLO"
description: >-
  Exécutez CenterNet (Objects as Points) dans LibreYOLO avec les backbones
  ResDCN-18 et DLA-34. Prédisez, validez et exportez vers ONNX sous licence MIT.
  Aucun parcours d'entraînement.
lead: >-
  CenterNet modélise un objet comme le point central de sa bounding box et
  régresse toutes ses autres propriétés à partir d'un pic de heatmap, si bien
  qu'il n'a besoin ni d'ancres ni d'étape de non-maximum suppression. LibreYOLO
  le fournit comme détecteur en inférence seule.
keywords:
  - CenterNet
  - Objects as Points
  - détection d'objets sans ancres
  - détection par points clés python
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # L'export ONNX exige l'opset 16 ou plus, car l'upsampling par
        # convolution déformable se traduit par GridSample, arrivé en opset 16.
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme un checkpoint et renvoie le même Results.
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Installation

CenterNet ne demande aucun extra optionnel. Tout ce qu'il importe fait partie de
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face au premier usage, puis mis en
cache localement.

<code-tabs name="predict" />

L'objet `Results` renvoyé est celui que renvoient toutes les familles, si bien
que passer à un autre détecteur ne demande qu'une ligne de changement. `conf` et
`max_det` filtrent les pics de heatmap déjà classés ; `iou` est accepté par
parité d'API mais n'a aucun effet, car le décodage top-k des pics de CenterNet
n'a besoin d'aucune étape de suppression par IoU de boîtes. Voir
[la prédiction](/docs/predict) pour les sources, le streaming et le traitement
des résultats.

## Variantes

Deux backbones. `resdcn18` associe un tronc ResNet-18 à un upsampling par
convolution déformable ; `dla34` associe un tronc DLA-34 à un upsampling par
agrégation profonde itérative. Les deux alimentent les mêmes trois têtes denses
(heatmap, largeur/hauteur, offset) et le même canevas d'entrée.

## Valider

`val()` renvoie un dictionnaire de clés `metrics/` couvrant la précision, le
rappel, la mAP 50 et la mAP 50-95, mesurées sur n'importe quel dataset dans le
format avec lequel vous avez entraîné.

<code-tabs name="val" />

## Exporter

<export-matrix />

L'export ONNX exige l'opset 16 ou plus récent : dans les deux backbones, l'étage
d'upsampling par convolution déformable se traduit par l'opérateur ONNX
`GridSample`, introduit par l'opset 16. Demander un opset antérieur lève une
erreur avant le début du tracé.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box>

Le graphe ResDCN-18 crédite également human-pose-estimation.pytorch de
Microsoft, sous licence MIT, et le graphe DLA-34 crédite l'implémentation DLA de
Fisher Yu, sous licence BSD-3-Clause. LibreYOLO n'embarque pas l'extension DCNv2
d'origine qu'utilisait le projet upstream ; l'exécution native passe à la place
par le `deform_conv2d` de torchvision, sous licence BSD-3-Clause, et
l'implémentation portable réservée à l'export a été écrite séparément pour
LibreYOLO.

</provenance-box>

## Citation

<citation-block />
