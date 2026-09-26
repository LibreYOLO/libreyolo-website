---
title: DeiT
families:
  - deit
seo_title: "Classifieur d'images DeiT\_: prédire, valider, exporter"
description: "Exécutez les classifieurs d'images DeiT dans LibreYOLO\_: une famille de musée figée, en inférence seule, en tailles tiny, small et base, sous licence Apache-2.0."
lead: >-
  DeiT (Data-efficient image Transformer) est un classifieur Vision Transformer
  pur, entraîné sur ImageNet-1k seul, sans données de pré-entraînement
  supplémentaires. LibreYOLO fournit les tailles tiny, small et base en patch-16
  comme une pièce de musée figée, en inférence seule.
keywords:
  - DeiT
  - Vision Transformer
  - ViT
  - classification d'images python
  - ImageNet
  - classifieur d'images pré-entraîné
  - famille de musée
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factory s'appuie sur l'extension du fichier, donc un artefact
        # exporté se charge comme n'importe quel checkpoint et renvoie le
        # même objet Results.
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## Installation

DeiT ne demande aucun extra au-delà du paquet de base.

```bash
pip install libreyolo
```

## Prédire

Cette famille est en inférence seule : `train()` lève `NotImplementedError`,
si bien que cette page n'a pas de section Entraîner. La prédiction, la
validation et l'export sont tous pris en charge. Les poids sont téléchargés
depuis Hugging Face au premier usage, puis mis en cache localement. Le suffixe
`-cls` dans le nom de fichier est obligatoire et sélectionne la tâche de
classification.

<code-tabs name="predict" />

L'objet `Results` renvoyé porte un tenseur `probs` au lieu de `boxes` ;
`top1` et `top5` indexent les 1 000 classes d'ImageNet-1k et `top1conf` est
le score softmax de la prédiction principale. Chaque taille a une résolution
d'entrée fixe qui vient de son positional embedding : le prétraitement
redimensionne et recadre au centre à cette résolution, et passer un `imgsz`
différent lève une erreur au lieu de rééchantillonner en silence. Voir [la
prédiction](/docs/predict) pour les sources, le streaming et le traitement des
résultats.

## Valider

`val()` renvoie un dictionnaire avec l'exactitude top-1 et top-5, mesurée sur un
dataset organisé selon la structure de dossiers classique `train/<class>/` et
`val/<class>/`.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge via `LibreYOLO()` selon son extension de
fichier, si bien qu'un fichier `.onnx` ou `.engine` se comporte comme un
checkpoint et renvoie le même `Results`. Exécuter le graphe dans un runtime nu,
sans LibreYOLO installé, est également pris en charge, mais le prétraitement et
le post-traitement sont alors à votre charge.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés de cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
