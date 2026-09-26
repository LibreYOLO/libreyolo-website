---
title: ViT
families:
  - vit
seo_title: "ViT\_: exécuter les classificateurs Vision Transformer classiques dans LibreYOLO"
description: "Prédisez, validez et exportez des classificateurs ViT avec LibreYOLO. Poids AugReg sous licence Apache-2.0\_; le fine-tuning n'est pas encore pris en charge."
lead: >-
  Le Vision Transformer classique est un transformer pur appliqué à des patchs
  d'image de taille fixe, avec un token de classe appris et sans convolution.
  LibreYOLO fournit quatre tailles pré-entraînées avec AugReg pour la
  classification d'images.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - classification d'images
  - classificateur transformer
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreViTti-cls.pt")


        # data est la racine d'un répertoire avec des sous-ensembles de dossiers
        par classe

        # train/ et val/ (structure ImageFolder), et non le YAML d'un dataset.

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Utiliser le fichier exporté
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La fabrique détermine le chemin selon le suffixe du fichier ; un
        artefact exporté se charge

        # donc comme tout checkpoint et renvoie le même objet Results.

        model = LibreYOLO("LibreViTti-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Installer

ViT ne nécessite aucun extra facultatif. Tous ses imports sont inclus dans
l'installation de base.

```bash
pip install libreyolo
```

## Prédire

Les poids sont téléchargés depuis Hugging Face à la première utilisation et
mis en cache localement.

<code-tabs name="predict" />

Un classificateur renvoie `result.probs` au lieu de `result.boxes`\u00a0: `top1` et
`top5` donnent les indices de classes, tandis que `top1conf` et `top5conf`
donnent leurs confiances. Le prétraitement redimensionne l'image et la recadre
au centre pour obtenir une entrée fixe de 224\u00a0px, selon la recette d'évaluation
AugReg de timm\u00a0: interpolation bicubique avec une fraction de recadrage de 0.9.
Consultez la [prédiction](/docs/predict) pour les sources, le streaming et la
gestion des résultats.

## Variantes

Quatre tailles, de tiny à large, partagent un même graphe patch-16 fixe de
224\u00a0px et se distinguent par la largeur des embeddings et la profondeur du
transformer. LibreYOLO fournit cette famille uniquement pour l'inférence\u00a0: la
prédiction, la validation top-1/top-5 de type ImageNet et l'export sont pris en
charge, mais la recette de fine-tuning AugReg n'est pas implémentée.

## Valider

`val()` s'exécute sur un sous-ensemble de type ImageFolder (un répertoire avec
des sous-dossiers `train/` et `val/`, un dossier par classe) et renvoie
l'exactitude top-1 et top-5.

<code-tabs name="val" />

## Exporter

<export-matrix />

Un artefact exporté se recharge dans `LibreYOLO()` grâce au suffixe de son
fichier. Un fichier `.onnx` ou `.engine` se comporte donc comme un checkpoint
et renvoie les mêmes `Results`. La page [Export](/docs/export) énumère les
arguments acceptés par chaque format ainsi que les extras ajoutés par certains.

<code-tabs name="export" />

## Checkpoints

Tous les fichiers de poids publiés pour cette famille.

<checkpoint-table />

## Licence

<provenance-box></provenance-box>

## Citation

<citation-block />
